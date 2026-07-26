"""CFDPy -- an educational Computational Fluid Dynamics framework in Python.

``main.py`` is the entry point.  It defines the :class:`Simulation` orchestrator
that wires together the mesh, the physical models, the finite-volume solvers
and the visualisation back-ends, then drives the time loop.

Run a case from the command line::

    python main.py examples/natural_convection_2D/config.json
    python main.py examples/dam_break_2D/config.json

The :class:`Simulation` class owns the global state (fields + solver objects)
and exposes :meth:`initialize`, :meth:`run` and :meth:`finalize`.  Keeping the
driver in one object makes it easy to embed the framework in notebooks or
tests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Make the package importable both as `from solver...` and as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, Config
from mesh import Mesh
from physics import Fluid, Gravity, BoussinesqBuoyancy
from solver import (BoundaryCondition, LinearSolver, ProjectionMethod,
                    EnergySolver, VOFSolver)
from visualization import MatplotlibViewer, TecplotExporter, PostProcessor


@dataclass
class FieldState:
    """Container for the time-evolving fields."""
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray | None
    p: np.ndarray
    T: np.ndarray
    alpha: np.ndarray | None = None


class Simulation:
    """Top-level CFD simulation driver.

    The :class:`Simulation` is the *composition root* of the framework: it
    instantiates every subsystem from the :class:`Config`, owns the field
    state, and advances it in time.  It is deliberately the only class that
    knows about all the subsystems at once -- every other module depends only
    on its neighbours through clean interfaces (SOLID dependency inversion).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mesh = Mesh(cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz)
        self.fluid = Fluid.from_config(cfg)
        self.gravity = Gravity.from_config(cfg)
        self.buoyancy = (BoussinesqBuoyancy.from_config(cfg, cfg.gravity)
                         if cfg.boussinesq else None)
        self.bc = BoundaryCondition(cfg)
        self.linear = LinearSolver(method=cfg.linear_solver, tol=cfg.linear_tol,
                                    maxiter=cfg.linear_maxiter,
                                    use_ilu=cfg.use_ilu)
        self.poisson = LinearSolver(method=cfg.linear_solver,
                                     tol=cfg.poisson_tol,
                                     maxiter=cfg.poisson_maxiter,
                                     use_ilu=cfg.use_ilu)
        self.projection = ProjectionMethod(self.mesh, self.fluid, self.bc,
                                            cfg, self.linear)
        self.energy = EnergySolver(self.mesh, self.fluid, self.bc, cfg,
                                   self.linear)
        self.vof = VOFSolver(self.mesh, self.fluid, self.bc, cfg) if cfg.use_vof else None
        self.post = PostProcessor(self.mesh)
        self.viewer = MatplotlibViewer(
            self.mesh, output_dir=cfg.output_dir,
            save_png=cfg.save_png, save_mp4=cfg.save_mp4)
        self.tec = TecplotExporter(self.mesh, output_dir=cfg.output_dir)
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Immersed-solid obstacles (blocked cells).  Built from the list of
        # axis-aligned boxes in ``cfg.obstacles``; a cell is solid when its
        # centre lies inside any box.  The mask is registered on the boundary
        # condition so every face-flux / gradient routine zeroes the solid
        # faces (direct-forcing / no-slip no-flux on the collocated grid).
        solid = self._build_solid_mask()
        if solid is not None:
            self.bc.set_solid(solid)

        self.state: FieldState | None = None
        self.time = 0.0
        self.step_count = 0
        self.history: list[dict[str, float]] = []

    # ================================================================== #
    # Initialisation
    # ================================================================== #
    def initialize(self) -> None:
        """Allocate and initialise every field from the configuration."""
        mesh = self.mesh
        u = np.full(mesh.cell_shape, self.cfg.u0, dtype=np.float64)
        v = np.full(mesh.cell_shape, self.cfg.v0, dtype=np.float64)
        w = np.full(mesh.cell_shape, self.cfg.w0, dtype=np.float64) if not mesh.is_2d else None
        p = np.zeros(mesh.cell_shape, dtype=np.float64)
        T = np.full(mesh.cell_shape, self.cfg.t0, dtype=np.float64)

        alpha = None
        if self.cfg.use_vof:
            alpha = self._init_alpha()

        # Apply the initial temperature BCs to the boundary cells.
        self.bc.apply_velocity(u, v, w)
        # Zero the velocity inside any immersed solid (no-slip direct forcing).
        self._apply_obstacle(u, v, w)
        self.state = FieldState(u=u, v=v, w=w, p=p, T=T, alpha=alpha)

        # Hydrostatic pressure for VOF (dam-break) so the fluid starts near
        # rest.  Disabled by default because the horizontal pressure jump at
        # the dam face creates a strong collocated-gradient transient; the
        # pressure field builds dynamically from p=0 instead, which is the
        # standard dam-break initialisation.  Enable via ``hydrostatic_init``.
        if self.cfg.use_vof and alpha is not None and \
                getattr(self.cfg, "hydrostatic_init", False):
            self._init_hydrostatic(alpha)

    def _init_alpha(self) -> np.ndarray:
        """Build the initial VOF field from the configured shape."""
        mesh = self.mesh
        alpha = np.full(mesh.cell_shape, self.cfg.alpha_value, dtype=np.float64)
        mode = self.cfg.alpha_init
        if mode == "dam_break":
            # Water column on the left, occupying half the width and a
            # configurable fraction of the height.
            nx0 = mesh.Nx // 2
            ny0 = int(0.5 * mesh.Ny)
            alpha[:nx0, :ny0, :] = 1.0
        elif mode == "block":
            nx0, nx1 = mesh.Nx // 3, 2 * mesh.Nx // 3
            ny0, ny1 = mesh.Ny // 3, 2 * mesh.Ny // 3
            alpha[nx0:nx1, ny0:ny1, :] = 1.0
        elif mode == "splash_drop":
            # The classic "splash of a liquid drop" (Harlow & Shannon, 1967):
            # a circular heavy-phase drop sits in the light phase above a
            # liquid pool at the bottom of a closed tank.  Gravity accelerates
            # the drop, which impacts the free surface and splashes (crown +
            # ejecta).  Geometry defaults resolve from the domain size when
            # the corresponding config value is left <= 0.
            h_pool = self.cfg.pool_height
            if h_pool <= 0.0:
                h_pool = 0.2 * mesh.Ly
            ny_pool = int(round(h_pool / mesh.dy))
            alpha[:, :ny_pool, :] = 1.0
            cx = self.cfg.drop_x if self.cfg.drop_x > 0.0 else 0.5 * mesh.Lx
            cy = self.cfg.drop_y if self.cfg.drop_y > 0.0 else 0.8 * mesh.Ly
            r = self.cfg.drop_r if self.cfg.drop_r > 0.0 else \
                0.06 * min(mesh.Lx, mesh.Ly)
            Xc, Yc, _ = mesh.cell_grid()
            inside = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= r ** 2
            alpha[inside] = 1.0
        # uniform -> already set to alpha_value
        np.clip(alpha, 0.0, 1.0, out=alpha)
        return alpha

    def _init_hydrostatic(self, alpha: np.ndarray) -> None:
        """Initialise pressure to the hydrostatic profile ``dp/dy = rho g_y``.

        With gravity pointing downward (``g_y < 0``) the pressure *decreases*
        upward; the column is integrated from the bottom and the top-row
        reference is set to zero so the field is centred on the origin.
        """
        rho, _mu, _cp, _k = self.fluid.blend(alpha)
        g = self.cfg.gravity
        p = np.zeros(self.mesh.cell_shape)
        if abs(g[1]) > 0:
            for j in range(1, self.mesh.Ny):
                # dp/dy = rho * g_y  ->  p_j = p_{j-1} + 0.5 (rho_{j-1}+rho_j) g_y dy
                p[:, j, :] = (p[:, j - 1, :]
                              + 0.5 * (rho[:, j - 1, :] + rho[:, j, :]) * g[1]
                              * self.mesh.dy)
        self.state.p = p - p[:, -1, :].mean()

    # ================================================================== #
    # Immersed obstacles
    # ================================================================== #
    def _build_solid_mask(self) -> np.ndarray | None:
        """Build the cell-centred solid mask from the configured boxes.

        Each entry of ``cfg.obstacles`` is a tuple
        ``(x0, x1, y0, y1, z0, z1)`` describing an axis-aligned box in physical
        coordinates.  A cell is marked solid when its centre lies inside any
        box.  Returns ``None`` when no obstacles are configured.
        """
        if not self.cfg.obstacles:
            return None
        mesh = self.mesh
        Xc, Yc, Zc = mesh.cell_grid()
        solid = np.zeros(mesh.cell_shape, dtype=bool)
        for (x0, x1, y0, y1, z0, z1) in self.cfg.obstacles:
            solid |= ((Xc >= x0) & (Xc <= x1) &
                      (Yc >= y0) & (Yc <= y1) &
                      (Zc >= z0) & (Zc <= z1))
        return solid

    def _apply_obstacle(self, u: np.ndarray, v: np.ndarray,
                        w: np.ndarray | None) -> None:
        """Clamp the velocity to zero inside solid cells (direct forcing)."""
        if not self.bc.has_solid:
            return
        solid = self.bc.solid
        u[solid] = 0.0
        v[solid] = 0.0
        if w is not None:
            w[solid] = 0.0

    # ================================================================== #
    # Source terms
    # ================================================================== #
    def _body_force(self, alpha: np.ndarray | None) -> tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
        """Cell-centred body-force *acceleration* (gravity + Boussinesq).

        The momentum equation is written in acceleration form
        ``du/dt = -grad p / rho + ... + g``; the density enters only through the
        pressure projection.  Hence the body force carried by the predictor is
        the acceleration ``g`` (the same for both phases) plus, for Boussinesq
        convection, the buoyant acceleration ``-beta (T-T0) g``.
        """
        mesh = self.mesh
        gx, gy, gz = self.cfg.gravity
        src_u = np.full(mesh.cell_shape, gx, dtype=np.float64)
        src_v = np.full(mesh.cell_shape, gy, dtype=np.float64)
        src_w = np.full(mesh.cell_shape, gz, dtype=np.float64)
        if self.buoyancy is not None:
            src_u = src_u + self.buoyancy.acceleration(self.state.T, 0)
            src_v = src_v + self.buoyancy.acceleration(self.state.T, 1)
            if not mesh.is_2d:
                src_w = src_w + self.buoyancy.acceleration(self.state.T, 2)
        return src_u, src_v, src_w

    # ================================================================== #
    # Time stepping
    # ================================================================== #
    def _pick_dt(self) -> float:
        if not self.cfg.adaptive_dt:
            return self.cfg.dt
        s = self.state
        nu = self.fluid.nu
        return float(np.minimum(self.cfg.dt,
                                 _cfl_dt(s.u, s.v, s.w, self.mesh, nu,
                                        self.cfg.cfl_max, self.cfg.dt_max,
                                        self.cfg.dt_min)))

    def step(self) -> dict[str, Any]:
        """Advance the simulation by one time step; return diagnostics."""
        s = self.state
        dt = self._pick_dt()

        # density field (None for constant-density; blended for VOF)
        rho = None
        if s.alpha is not None and self.vof is not None:
            rho, _mu, _cp, _k = self.fluid.blend(s.alpha)
            rho = np.asarray(rho, dtype=np.float64)

        src = self._body_force(s.alpha)
        res = self.projection.step(s.u, s.v, s.w, s.p, dt, src, rho=rho)

        s.u, s.v, s.w, s.p = res["u"], res["v"], res["w"], res["p"]
        # re-apply wall / inlet / outlet BCs on the corrected velocity
        self.bc.apply_velocity(s.u, s.v, s.w)
        # clamp the velocity inside any immersed solid (direct forcing)
        self._apply_obstacle(s.u, s.v, s.w)

        # Energy step (only if temperature is part of the physics -- it always is
        # here, but a pure-isothermal case can keep T constant).
        s.T = self.energy.step(s.T, s.u, s.v, s.w, dt)

        # VOF transport
        if self.vof is not None and s.alpha is not None:
            s.alpha = self.vof.advect(s.alpha, s.u, s.v, s.w, dt)

        self.time += dt
        self.step_count += 1
        diag = {
            "step": self.step_count, "t": self.time, "dt": dt,
            "div": res["div"],
            "umax": float(np.abs(s.u).max()),
            "vmax": float(np.abs(s.v).max()),
        }
        if s.alpha is not None and self.vof is not None:
            diag["mass"] = self.vof.mass(s.alpha)
        return diag

    # ================================================================== #
    # Output
    # ================================================================== #
    def _snapshot(self) -> dict[str, np.ndarray]:
        s = self.state
        return {
            "u": s.u, "v": s.v, "w": (s.w if s.w is not None else np.zeros_like(s.u)),
            "p": s.p, "T": s.T,
            "alpha": s.alpha if s.alpha is not None else np.zeros_like(s.u),
        }

    def _save_frame(self, diag: dict[str, Any]) -> None:
        snap = self._snapshot()
        t = self.time
        idx = self.step_count
        # All back-ends consume the same snapshot.
        if self.cfg.save_tecplot:
            self.tec.write(t, snap["u"], snap["v"], snap["w"], snap["p"],
                           snap["T"], snap["alpha"],
                           fname=f"frame_{idx:06d}.dat")
        if self.cfg.save_csv:
            self.tec.write_csv(t, snap["u"], snap["v"], snap["w"], snap["p"],
                               snap["T"], snap["alpha"],
                               fname=f"frame_{idx:06d}.csv")
        if self.cfg.save_hdf5:
            snap["w"] = snap["w"]
            self.tec.write_hdf5(t, snap, fname=f"frame_{idx:06d}.h5")
        if self.cfg.save_png:
            self.viewer.plot_field(snap["T"], "Temperature", f"T_{idx:06d}.png",
                                   cmap="inferno",
                                   vectors=(snap["u"], snap["v"]),
                                   alpha=snap["alpha"])
            self.viewer.plot_field(snap["p"], "Pressure", f"p_{idx:06d}.png",
                                   cmap="viridis")
            # Velocity field: speed magnitude with overlaid quiver arrows and
            # streamlines -- shows the instantaneous flow direction, the
            # recirculation zones and the reattachment points.
            speed = np.sqrt(snap["u"] ** 2 + snap["v"] ** 2
                            + (snap["w"] ** 2 if snap["w"] is not None else 0.0))
            self.viewer.plot_field(speed, "Speed |u|", f"vel_{idx:06d}.png",
                                   cmap="viridis",
                                   vectors=(snap["u"], snap["v"]),
                                   streamlines=True)
        self.viewer.add_frame(t, snap)
        # Nusselt history for natural-convection cases.
        if self.cfg.boussinesq:
            Nu = self._mean_nusselt(snap["T"])
            diag["Nu"] = Nu

    def _mean_nusselt(self, T) -> float:
        # Heuristic: read the hot/cold values from the temperature BCs.
        hot = cold = None
        for patch, spec in self.bc.temperature.items():
            if spec.kind == "fixed":
                if patch == "west":
                    hot = spec.value
                elif patch == "east":
                    cold = spec.value
        if hot is None or cold is None:
            return 0.0
        return self.post.nusselt_wall(T, hot, cold, side="west")

    # ================================================================== #
    # Restart
    # ================================================================== #
    def restart_from(self, path: str) -> None:
        """Resume the simulation from a saved HDF5 field snapshot.

        Loads the cell-centred ``u,v,w,p,T,alpha`` and the recorded time from
        ``path`` (a file written by :meth:`TecplotExporter.write_hdf5`), applies
        the velocity boundary conditions and obstacle direct-forcing, and seeds
        :attr:`time` / :attr:`step_count` so the time loop and the per-frame
        output numbering continue seamlessly.  The previously saved frames in
        the output directory are pre-loaded into the viewer so the final
        animations span the whole run, not just the resumed leg.

        This is the framework's lightweight checkpoint/resume path: there is no
        separate restart file format -- every output HDF5 frame is a valid
        restart snapshot.
        """
        import glob
        import re
        import h5py
        mesh = self.mesh
        with h5py.File(path, "r") as fh:
            t0 = float(fh.attrs["time"])
            u = np.asarray(fh["u"][()], dtype=np.float64)
            v = np.asarray(fh["v"][()], dtype=np.float64)
            w = (np.asarray(fh["w"][()], dtype=np.float64)
                 if not mesh.is_2d and "w" in fh else None)
            p = np.asarray(fh["p"][()], dtype=np.float64)
            T = np.asarray(fh["T"][()], dtype=np.float64)
            alpha = (np.asarray(fh["alpha"][()], dtype=np.float64)
                     if "alpha" in fh else None)
        # Enforce wall / inlet / outlet BCs and obstacle direct-forcing on the
        # loaded velocity, exactly as initialize() does on the freshly built one.
        self.bc.apply_velocity(u, v, w)
        self._apply_obstacle(u, v, w)
        self.state = FieldState(u=u, v=v, w=w, p=p, T=T, alpha=alpha)
        self.time = t0

        # Continue the per-frame numbering past the highest existing frame so
        # new outputs (frame_{step:06d}.*) never collide with saved ones.
        existing = glob.glob(os.path.join(self.cfg.output_dir, "frame_*.h5"))
        mx = 0
        for fp in existing:
            m = re.search(r"frame_(\d+)\.h5", os.path.basename(fp))
            if m:
                mx = max(mx, int(m.group(1)))
        self.step_count = mx

        # Pre-load the saved frames + history so the animations and history.csv
        # cover the full run rather than only the resumed leg.
        self._preload_frames(existing)
        self._preload_history()

    def _preload_frames(self, files: list[str]) -> None:
        """Load saved HDF5 snapshots into the viewer in time order."""
        import h5py
        snaps: list[tuple[float, dict[str, np.ndarray]]] = []
        for fp in files:
            with h5py.File(fp, "r") as fh:
                t = float(fh.attrs["time"])
                fields = {k: fh[k][()] for k in ("u", "v", "w", "p", "T",
                                                 "alpha") if k in fh}
            snaps.append((t, fields))
        snaps.sort(key=lambda s: s[0])
        for t, fields in snaps:
            self.viewer.add_frame(t, fields)

    def _preload_history(self) -> None:
        """Load the existing history.csv rows so the resumed run appends to them."""
        import csv
        path = os.path.join(self.cfg.output_dir, "history.csv")
        if not os.path.exists(path):
            return
        with open(path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.history.append({k: (float(v) if v not in (None, "") else v)
                                     for k, v in row.items()})

    def _next_output_time(self) -> float:
        """Smallest multiple of ``output_interval`` at or past the current time.

        Keeps the output grid aligned across a restart: a fresh run starts at
        ``t = 0`` (next output 0), a resumed run continues at the next grid
        point after the loaded time.
        """
        oi = self.cfg.output_interval
        if oi <= 0.0:
            return self.time
        k = int(np.ceil(self.time / oi - 1e-9))
        return k * oi

    # ================================================================== #
    # Main loop
    # ================================================================== #
    def run(self) -> None:
        if self.cfg.restart:
            self.restart_from(self.cfg.restart)
        else:
            self.initialize()
        if self.cfg.verbose:
            print(self._header())
        next_output = self._next_output_time()
        bar = _TqdmBar(self.cfg.tfinal, enabled=self.cfg.verbose)
        while self.time < self.cfg.tfinal - 1e-12:
            diag = self.step()
            bar.update(self.time, diag)
            if self.time >= next_output - 1e-12:
                self._save_frame(diag)
                self.history.append(dict(diag))
                next_output = self.time + self.cfg.output_interval
        # final frame
        diag = {"step": self.step_count, "t": self.time, "dt": self.cfg.dt,
                "div": 0.0, "umax": float(np.abs(self.state.u).max()),
                "vmax": float(np.abs(self.state.v).max())}
        if self.vof is not None and self.state.alpha is not None:
            diag["mass"] = self.vof.mass(self.state.alpha)
        self._save_frame(diag)
        self.history.append(dict(diag))
        if isinstance(bar, _TqdmBar) and bar._bar is not None:
            bar._bar.close()
        self.finalize()

    # ================================================================== #
    def finalize(self) -> None:
        """Write animations, the history table and a summary."""
        if self.cfg.save_mp4:
            self.viewer.animate("T", f"{self.cfg.name}_T.mp4", cmap="inferno",
                                title="Temperature")
            self.viewer.animate("p", f"{self.cfg.name}_p.mp4", cmap="viridis",
                                title="Pressure")
            if self.cfg.use_vof:
                self.viewer.animate("alpha", f"{self.cfg.name}_alpha.mp4",
                                    cmap="coolwarm", title="VOF alpha")
            # Velocity field animation: speed magnitude with overlaid quiver
            # arrows and (optionally) streamlines so the flow direction,
            # recirculation and reattachment are visible in the movie.  The
            # streamline overlay is gated by ``flow_streamlines`` because
            # matplotlib's streamplot stalls on the chaotic fields produced by
            # VOF splash / dam-break cases.
            self.viewer.animate_flow(f"{self.cfg.name}_velocity.mp4",
                                     background="speed", cmap="viridis",
                                     title="Velocity |u|",
                                     with_streamlines=self.cfg.flow_streamlines)
        # history CSV
        if self.history:
            import csv
            path = os.path.join(self.cfg.output_dir, "history.csv")
            keys = sorted({k for h in self.history for k in h})
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                for h in self.history:
                    w.writerow(h)
        print(self._summary())

    # ------------------------------------------------------------------ #
    def _header(self) -> str:
        return (f"\n=== CFDPy simulation: {self.cfg.name} ===\n"
                f"{self.mesh}\n"
                f"dt={self.cfg.dt}, tfinal={self.cfg.tfinal}, "
                f"scheme={self.cfg.convection}, time={self.cfg.time_scheme}, "
                f"solver={self.cfg.linear_solver}\n"
                f"VOF={'on' if self.cfg.use_vof else 'off'}, "
                f"Boussinesq={'on' if self.cfg.boussinesq else 'off'}\n")

    def _summary(self) -> str:
        s = self.state
        return (f"\n=== {self.cfg.name} finished ===\n"
                f"steps={self.step_count}, t={self.time:.4f}s\n"
                f"max|u|={np.abs(s.u).max():.4e}, max|v|={np.abs(s.v).max():.4e}\n"
                f"outputs in: {self.cfg.output_dir}\n")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _cfl_dt(u, v, w, mesh, nu, cfl, dt_max, dt_min):
    umax = max(float(np.abs(u).max()), 1e-30)
    vmax = max(float(np.abs(v).max()), 1e-30)
    dt_conv = 1.0 / (umax / mesh.dx + vmax / mesh.dy)
    dt_diff = 1.0 / (2.0 * nu * (1.0 / mesh.dx**2 + 1.0 / mesh.dy**2))
    return max(min(cfl * dt_conv, 0.25 * dt_diff, dt_max), dt_min)


class _TqdmBar:
    """A minimal progress bar that uses tqdm if available, else prints."""

    def __init__(self, total: float, enabled: bool = True) -> None:
        self.total = total
        self.enabled = enabled
        self._bar = None
        if enabled:
            try:
                from tqdm.auto import tqdm
                self._bar = tqdm(total=total, unit="s", desc="run", smoothing=0.0)
            except Exception:
                self._bar = None
        self._last = 0.0

    def update(self, t: float, diag: dict) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.n = max(0.0, min(t, self.total))
            self._bar.refresh()
        else:
            if t - self._last > 0.1 * self.total or t >= self.total:
                print(f"  t={t:.3f}/{self.total:.3f} "
                      f"div={diag.get('div', 0):.2e} "
                      f"umax={diag.get('umax', 0):.3e}")
                self._last = t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CFDPy -- finite-volume CFD framework.")
    parser.add_argument("config", help="Path to a JSON/YAML case file.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sim = Simulation(cfg)
    sim.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())