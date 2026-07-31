"""Case-file configuration loading and validation.

The :class:`Config` dataclass aggregates *every* parameter required to drive a
simulation: mesh dimensions, time stepping, physical properties, numerical
schemes, boundary conditions, initial fields and output options.

Two on-disk layouts are accepted by :func:`load_config`:

1. **Flat** -- the compact form used in the project specification::

       {"Nx": 100, "Ny": 100, "Nz": 1, "Lx": 1.0, "rho": 1000, "mu": 0.001, ...}

2. **Nested** -- the explicit, self-documenting form::

       {"mesh": {"Nx": 100, ...}, "time": {"dt": 0.001, ...}, ...}

Both forms are merged into the same dataclass, with the flat keys taking
precedence over nested defaults when both are present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:  # YAML is optional; JSON alone is always supported.
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
@dataclass
class BoundarySpec:
    """Specification of a single boundary patch.

    A patch is identified by one of ``west|east|south|north|bottom|top``.  The
    ``kind`` selects the physics treatment (``no-slip``, ``slip``, ``inlet``,
    ``outlet``, ``symmetry``, ``periodic`` for the velocity field, and
    ``fixed``, ``heatflux``, ``adiabatic`` for the temperature field).  ``value``
    carries the prescribed quantity (velocity component, temperature or heat
    flux) and is ignored when not applicable.
    """

    kind: str = "no-slip"
    value: float = 0.0


def _parse_patch(raw: Any) -> BoundarySpec:
    if isinstance(raw, BoundarySpec):
        return raw
    if isinstance(raw, str):
        return BoundarySpec(kind=raw)
    if isinstance(raw, dict):
        return BoundarySpec(
            kind=raw.get("kind", raw.get("type", "no-slip")),
            value=float(raw.get("value", 0.0)),
        )
    raise TypeError(f"Invalid boundary specification: {raw!r}")


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Aggregate configuration of a CFD case.

    Attributes are grouped by concern for readability but the dataclass remains
    flat so downstream code can access ``cfg.Nx``, ``cfg.dt`` and so on without
    navigating sub-objects.
    """

    # -- Mesh ---------------------------------------------------------------
    # All lengths are in metres [m]; the mesh spacing dx=Lx/Nx etc. is likewise
    # in [m].  Cell counts Nx/Ny/Nz are dimensionless integers.
    Nx: int = 64                 # cells in x                     [-]
    Ny: int = 64                 # cells in y                     [-]
    Nz: int = 1                  # cells in z (1 => 2-D)          [-]
    Lx: float = 1.0              # domain length in x             [m]
    Ly: float = 1.0              # domain length in y             [m]
    Lz: float = 1.0              # domain length in z             [m]

    # -- Time integration ---------------------------------------------------
    # All times are in seconds [s].
    dt: float = 0.001            # time step                      [s]
    tfinal: float = 10.0         # final simulation time          [s]
    time_scheme: str = "crank-nicolson"  # "implicit" | "crank-nicolson" | "explicit"
    adaptive_dt: bool = False
    cfl_max: float = 0.5         # target max Courant number      [-]
    dt_min: float = 1e-7         # adaptive-dt lower bound         [s]
    dt_max: float = 1e-2         # adaptive-dt upper bound         [s]

    # -- Physical properties (single-phase defaults) -----------------------
    # Coherent SI throughout.  Defaults are liquid water at ~20 C.
    rho: float = 1000.0          # density                        [kg/m^3]
    mu: float = 1.0e-3           # dynamic viscosity              [Pa*s]
    cp: float = 4180.0           # specific heat capacity         [J/(kg*K)]
    k: float = 0.6               # thermal conductivity           [W/(m*K)]
    beta: float = 0.0            # volumetric thermal expansion   [1/K]
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)  # accel. [m/s^2]

    # -- Numerics -----------------------------------------------------------
    convection: str = "upwind"    # "upwind" | "central" | "quick" | "tvd"
    limiter: str = "vanleer"       # TVD limiter (vanleer, minmod, superbee, ...)
    linear_solver: str = "bicgstab"  # "cg" | "bicgstab" | "gmres"
    linear_tol: float = 1e-6
    linear_maxiter: int = 2000
    poisson_tol: float = 1e-7
    poisson_maxiter: int = 3000
    # ILU(0) preconditioner for the Krylov solves.  Recommended for
    # constant-coefficient (single-phase, fixed-dt) cases -- the factorisation
    # is cached once and dramatically cuts iteration counts.  For variable-
    # coefficient runs (VOF, adaptive dt) the matrix changes every step so the
    # ILU is rebuilt each step to no net benefit; set this to False there.
    use_ilu: bool = True

    # -- Multiphase ---------------------------------------------------------
    use_vof: bool = False
    rho_light: float = 1.2        # secondary (light) phase density [kg/m^3]
    mu_light: float = 1.8e-5      # secondary (light) phase viscosity [Pa*s]
    sigma: float = 0.0            # surface tension                 [N/m]
    vof_reconstruct: str = "plic"  # "upwind" | "plic"

    # -- Gravity / buoyancy -------------------------------------------------
    boussinesq: bool = False
    t_ref: float = 300.0          # Boussinesq reference temperature [K]

    # -- Non-dimensionalisation (SI-policy exception) ----------------------
    # The framework's default policy is *coherent SI everywhere*.  A case may
    # instead be posed in non-dimensional (unit-scaled) variables -- e.g. the
    # cylinder / backward-step benchmarks set rho=1, U=1, D=1 so that
    # Re = rho*U*D/mu = 1/mu -- but ONLY if it declares that intent here.  The
    # SI validator (units.validate_config) treats a case with non-physical
    # property values as an *error* unless ``nondimensional`` is True, in which
    # case it is accepted as a documented exception.  ``reference_scales`` names
    # the scales used so a reader can recover dimensional values.
    nondimensional: bool = False
    reference_scales: dict = field(default_factory=dict)  # e.g. {"L":..,"U":..,"rho":..,"dT":..}

    # -- Energy (temperature) transport ------------------------------------
    # Solve the temperature advection-diffusion equation each step.  For a
    # genuinely isothermal case -- no conduction (``k == 0``), no buoyancy
    # feedback (``boussinesq`` off) and a uniform initial temperature -- the
    # temperature carries no physics and is a passive scalar.  On a collocated
    # mesh advecting that scalar with the cell-to-face averaged velocity is not
    # discretely divergence-free (only the projected face flux is), so a uniform
    # field drifts by hundreds of K over a long run for no physical reason.
    # Set this to ``false`` for such cases (e.g. isothermal flow past a
    # cylinder): T stays at its initial value, which is the exact physical
    # answer.  Defaults to ``true`` for the thermal examples.
    solve_energy: bool = True

    # -- Boundary conditions ------------------------------------------------
    velocity_bc: dict[str, BoundarySpec] = field(default_factory=dict)
    pressure_bc: dict[str, BoundarySpec] = field(default_factory=dict)
    temperature_bc: dict[str, BoundarySpec] = field(default_factory=dict)

    # -- Initial conditions -------------------------------------------------
    u0: float = 0.0               # initial x-velocity             [m/s]
    v0: float = 0.0               # initial y-velocity             [m/s]
    w0: float = 0.0               # initial z-velocity             [m/s]
    t0: float = 300.0             # initial (uniform) temperature  [K]
    alpha_init: str = "uniform"   # "uniform" | "dam_break" | "block" | "splash_drop"
    alpha_value: float = 0.0      # background VOF fraction (0 = light, 1 = heavy) [-]

    # Splash-of-a-liquid-drop initial shape (used when alpha_init == "splash_drop"):
    # a circular heavy-phase drop suspended in the light phase above a liquid
    # pool at the bottom of the domain.  Any value <= 0 means "auto" and is
    # resolved from the domain size in Simulation._init_alpha.
    drop_x: float = 0.0           # x-centre of the drop            [m]
    drop_y: float = 0.0           # y-centre of the drop            [m]
    drop_r: float = 0.0           # radius of the drop              [m]
    pool_height: float = 0.0      # height of the bottom liquid pool [m]

    # -- Output -------------------------------------------------------------
    output_dir: str = "outputs"
    output_interval: float = 0.1  # simulation-time between output frames [s]
    save_csv: bool = True
    save_hdf5: bool = True
    save_tecplot: bool = True
    save_png: bool = True
    save_mp4: bool = True
    verbose: bool = True
    plot_interval: int = 10

    # -- Misc ---------------------------------------------------------------
    name: str = "case"
    two_d: bool = True
    # Restart: path to an HDF5 field snapshot (as written by TecplotExporter)
    # to resume from.  When set, Simulation.run loads u,v,w,p,T,alpha and the
    # recorded time from the file and continues the time loop instead of
    # calling initialize().  Empty string = start from scratch (default).
    restart: str = ""
    # Overlay streamlines on the velocity animation.  matplotlib's streamplot
    # can stall indefinitely on the chaotic velocity fields produced by VOF
    # splash / dam-break cases; set this to False there to keep finalize()
    # responsive (the speed-magnitude + quiver overlay is still rendered).
    flow_streamlines: bool = True
    # Immersed solid obstacles (blocked cells).  Each entry is a shape dict:
    # an axis-aligned box ``{"x0":..,"x1":..,"y0":..,"y1":..[,"z0":..,"z1":..]}``,
    # a circular ``{"shape":"cylinder","center":[cx,cy],"radius":r,"axis":"z"}``
    # (2-D xy-plane cylinder for ``axis="z"``), or a 3-D
    # ``{"shape":"sphere","center":[cx,cy,cz],"radius":r}``.  Cells whose centre
    # lies inside an obstacle are treated as solid (no-slip, no-flux) via direct
    # forcing on the collocated grid -- the face fluxes at solid/fluid
    # interfaces are zeroed and the velocity in solid cells is clamped to zero.
    # A curved body is a cell-resolution "staircase" (O(dx) jagged boundary,
    # convergent only under refinement).  This is how the backward-facing step
    # (a box) and the flow-past-a-cylinder case are represented on a Cartesian
    # structured mesh.
    obstacles: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a (possibly nested) mapping."""

        cfg = cls()
        merged: dict[str, Any] = {}

        # Pull nested sections up to the top level so the flat dataclass can
        # consume them.
        sections = ("mesh", "time", "physics", "numerics", "multiphase",
                    "boundary", "initial", "output", "case")
        for sec in sections:
            if isinstance(data.get(sec), dict):
                for key, val in data[sec].items():
                    merged[key] = val

        # Flat keys override / complement.
        for key, val in data.items():
            if key not in sections:
                merged[key] = val

        # Map a few aliased keys used in the project specification.
        aliases = {"T0": "t0", "t_initial": "t0"}
        for src, dst in aliases.items():
            if src in merged and dst not in merged:
                merged[dst] = merged[src]

        # Boundary conditions need structural conversion.
        bc_groups = ("velocity_bc", "pressure_bc", "temperature_bc")
        for grp in bc_groups:
            if grp in merged and isinstance(merged[grp], dict):
                merged[grp] = {
                    patch: _parse_patch(spec) for patch, spec in merged[grp].items()
                }

        # Gravity may be a list -> tuple.
        if "gravity" in merged and isinstance(merged["gravity"], (list, tuple)):
            merged["gravity"] = tuple(float(g) for g in merged["gravity"])

        # Obstacles: normalise each entry to a shape dict (box / cylinder /
        # sphere).  Kept as dicts (not tuples) so ``_build_solid_mask`` can
        # dispatch on the shape -- this matches the CFDPyGPU variant so a case
        # file runs unchanged on either.
        if "obstacles" in merged and isinstance(merged["obstacles"], list):
            norm = []
            for b in merged["obstacles"]:
                if not isinstance(b, dict):
                    continue
                shape = b.get("shape", "box")
                if shape == "box":
                    norm.append({
                        "shape": "box",
                        "x0": float(b.get("x0", 0.0)), "x1": float(b.get("x1", 0.0)),
                        "y0": float(b.get("y0", 0.0)), "y1": float(b.get("y1", 0.0)),
                        "z0": float(b.get("z0", -1e9)), "z1": float(b.get("z1", 1e9)),
                    })
                elif shape == "cylinder":
                    c = b.get("center", [0.0, 0.0])
                    norm.append({
                        "shape": "cylinder",
                        "center": (float(c[0]), float(c[1])),
                        "radius": float(b.get("radius", 0.0)),
                        "axis": str(b.get("axis", "z")),
                    })
                elif shape == "sphere":
                    c = b.get("center", [0.0, 0.0, 0.0])
                    norm.append({
                        "shape": "sphere",
                        "center": (float(c[0]), float(c[1]), float(c[2])),
                        "radius": float(b.get("radius", 0.0)),
                    })
            merged["obstacles"] = norm

        # Sanity: only assign known fields (ignore unknown keys gracefully).
        known = set(cfg.__dict__.keys())
        for key, val in merged.items():
            if key in known:
                if isinstance(val, (list, tuple)) and key != "gravity":
                    # leave scalar lists alone
                    pass
                setattr(cfg, key, val)

        cfg.two_d = cfg.Nz <= 1
        return cfg

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # BoundarySpec dataclasses become dicts.
        for grp in ("velocity_bc", "pressure_bc", "temperature_bc"):
            d[grp] = {k: (v if isinstance(v, dict) else v.__dict__)
                      for k, v in self.__dict__[grp].items()}
        return d


def default_config() -> Config:
    """Return a default configuration object (handy for tests and demos)."""
    return Config()


def load_config(path: str | os.PathLike) -> Config:
    """Load a case file from JSON or YAML.

    The file extension selects the parser; ``.yml``/``.yaml`` require PyYAML,
    everything else is parsed as JSON.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to read YAML case files.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Case file {path} must contain a mapping at top level.")
    cfg = Config.from_dict(data)
    if not cfg.name or cfg.name == "case":
        cfg.name = path.parent.name or path.stem
    # Default output directory rooted at the case name.
    if cfg.output_dir == "outputs":
        cfg.output_dir = f"outputs/{cfg.name}"
    return cfg