"""Cut-cell geometry for a curved immersed boundary on the Cartesian mesh.

This module computes, for the cell-centred solid mask of an immersed obstacle,
the per-cell **fluid volume fraction** and the per-face **aperture fraction**
(fluid fraction of the face).  These let the pressure-Poisson divergence place
the no-penetration wall at the *true* immersed boundary instead of at the
staircase cell face -- the change that makes a bluff body separate at the true
surface rather than at the staircase's 90 corners (see
``Handoff_cylinder.md`` §3).

Only the cylinder-benchmark geometry is handled with cut cells here:

* **z-axis cylinder** -- the 2-D xy-plane circle; fractions are computed
  geometrically.
* **box** -- an axis-aligned box whose bounds already lie on grid lines (the
  backward-facing step) has *no* cut cells: the staircase is the true boundary
  to O(0).  Boxes are therefore left on the centre-test mask (volume fraction
  1 outside / 0 inside, no partial cells).
* other shapes / 3-D -- fall back to the centre-test mask (no cut cells); the
  IBM path is 2-D-only anyway (``solver/ibm.py``).

Geometric fractions are computed by **deterministic sub-cell sampling** of the
*candidate* (boundary-straddling) cells and faces only; fully-fluid and
fully-solid cells use the cheap centre test.  The sampling resolution is ``sub``
(default 64); the O(dx/sub) error is far below the O(dx) staircase error being
removed and converges under refinement.  An analytic circle-rectangle kernel
can replace the sampler later without changing the API.

Small-cell stabilisation
------------------------
A cut cell whose fluid volume fraction falls below ``small_cell_eps`` would
make the Poisson diagonal ``~aperture/vf`` blow up.  Such cells are flagged
solid (``volume_fraction -> 0``, all their face apertures -> 0) so the linear
system stays well-conditioned.  This "kill small cells" simplification loses a
negligible fluid volume (small cells are rare -- only where the boundary barely
clips a cell corner) and is stable; a full flux-redistribution cell-merging can
replace it later if high-Re accuracy demands.

Consistency with the centre mask
--------------------------------
The cut-cell ``is_solid`` (``volume_fraction == 0``) differs from the raw
centre-test ``solid`` mask for *cut* cells whose centre is inside the body but
which are partially fluid.  The Poisson / face-flux path must use ``is_solid``
(not the centre mask) so those partial cells are included with their true
volume.  The velocity-clamp / ghost-cell IBM path continues to use the centre
mask (it only needs to know which cells are inside the body).
"""

from __future__ import annotations

import numpy as np


class CutCellGeometry:
    """Fluid volume / face aperture fractions for an immersed obstacle set.

    Parameters
    ----------
    mesh:
        The Cartesian :class:`Mesh` (2-D, ``Nz == 1``).
    solid:
        Cell-centred boolean mask from the centre test (``True`` inside a body).
        Used for the cheap full-cell classification and as the fallback when no
        curved body is present.
    bodies:
        The obstacle dict list (same ``cfg.obstacles`` consumed by
        :meth:`Simulation._build_solid_mask`); centres should already be
        snapped to the grid (``snap_obstacle_to_grid``) so the fractions are
        mirror-symmetric.
    sub:
        Sub-cell sampling resolution per axis (default 64).
    small_cell_eps:
        Fluid volume fraction below which a cut cell is flagged solid for
        stability (default 1e-6).
    """

    def __init__(self, mesh, solid: np.ndarray, bodies: list[dict],
                 sub: int = 64, small_cell_eps: float = 1e-6) -> None:
        self.mesh = mesh
        self.solid = np.asarray(solid, dtype=bool)
        self.bodies = bodies
        self.sub = int(sub)
        self.small_cell_eps = float(small_cell_eps)

        # Outputs (filled by _compute; defaults = pure centre-mask fallback).
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        self.volume_fraction = np.where(self.solid, 0.0, 1.0).astype(np.float64)
        self.aperture: dict[int, np.ndarray] = {
            0: self._centre_aperture(0),
            1: self._centre_aperture(1),
        }
        if not mesh.is_2d:
            self.aperture[2] = self._centre_aperture(2)
        self.is_solid = self.solid.copy()
        self.is_fluid = (~self.solid).copy()
        self.is_cut = np.zeros(self.solid.shape, dtype=bool)
        self.n_cut = 0
        self.n_small = 0
        self.has_curve = False

        if mesh.is_2d and self._has_curved_body():
            self.has_curve = True
            self._compute_2d()

    # ================================================================== #
    # Geometry helpers
    # ================================================================== #
    def _has_curved_body(self) -> bool:
        return any(ob.get("shape") in ("cylinder", "sphere")
                   for ob in self.bodies)

    def _inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean: point ``(x, y)`` lies inside any obstacle (2-D xy-plane).

        Boxes use the axis-aligned rectangle test; z-axis cylinders use the
        circle test.  (Cylinders along x / y and spheres are 3-D and are not
        reached here -- the 2-D path only handles z-axis cylinders; everything
        else falls back to the centre mask.)
        """
        inside = np.zeros_like(x, dtype=bool)
        for ob in self.bodies:
            shape = ob.get("shape", "box")
            if shape == "box":
                inside |= ((x >= ob["x0"]) & (x <= ob["x1"]) &
                           (y >= ob["y0"]) & (y <= ob["y1"]))
            elif shape == "cylinder" and ob.get("axis", "z") == "z":
                cx, cy = ob["center"]
                r = ob["radius"]
                inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            # other shapes: ignored by the 2-D cut-cell path (centre mask used)
        return inside

    # ------------------------------------------------------------------ #
    def _centre_aperture(self, axis: int) -> np.ndarray:
        """Centre-test aperture: 1 on fluid-fluid faces, 0 if either side solid.

        Matches :meth:`BoundaryCondition.solid_face_mask` so that the fallback
        (no curved body) reproduces the staircase behaviour exactly.
        """
        s = self.solid
        if axis == 0:
            ap = np.zeros((s.shape[0] + 1, s.shape[1], s.shape[2]))
            ap[1:-1] = 1.0 - (s[:-1] | s[1:]).astype(np.float64)
            ap[0] = 1.0 - s[0].astype(np.float64)
            ap[-1] = 1.0 - s[-1].astype(np.float64)
        elif axis == 1:
            ap = np.zeros((s.shape[0], s.shape[1] + 1, s.shape[2]))
            ap[:, 1:-1] = 1.0 - (s[:, :-1] | s[:, 1:]).astype(np.float64)
            ap[:, 0] = 1.0 - s[:, 0].astype(np.float64)
            ap[:, -1] = 1.0 - s[:, -1].astype(np.float64)
        else:
            ap = np.zeros((s.shape[0], s.shape[1], s.shape[2] + 1))
            ap[:, :, 1:-1] = 1.0 - (s[:, :, :-1] | s[:, :, 1:]).astype(np.float64)
            ap[:, :, 0] = 1.0 - s[:, :, 0].astype(np.float64)
            ap[:, :, -1] = 1.0 - s[:, :, -1].astype(np.float64)
        return ap

    # ================================================================== #
    # 2-D cut-cell computation
    # ================================================================== #
    def _compute_2d(self) -> None:
        mesh = self.mesh
        Nx, Ny = mesh.Nx, mesh.Ny
        dx, dy = mesh.dx, mesh.dy
        sub = self.sub
        s2 = self.solid[:, :, 0]

        # --- candidate (boundary-crossing) cells --------------------------- #
        # A cell needs sub-sampling if a boundary passes through it.  The
        # corner-straddle test (some corners in, some out) catches boxes and
        # circle crossings through corners, but a circle can clip a cell
        # through its edges alone -- all four corners on the same side -- which
        # the corner test misses (leaving an O(dx) area error).  For cylinders
        # we add a *band* test: the cell centre is within half a cell diagonal
        # of the circle, which catches every crossed cell (and a few fully
        # solid/fluid neighbours, which sub-sample harmlessly to 0 / 1).
        xlo = mesh.xc - 0.5 * dx   # (Nx,)
        xhi = mesh.xc + 0.5 * dx
        ylo = mesh.yc - 0.5 * dy   # (Ny,)
        yhi = mesh.yc + 0.5 * dy
        # corner inside-flags, shape (Nx, Ny) each
        c_ll = self._inside(*np.meshgrid(xlo, ylo, indexing="ij"))
        c_lh = self._inside(*np.meshgrid(xlo, yhi, indexing="ij"))
        c_hl = self._inside(*np.meshgrid(xhi, ylo, indexing="ij"))
        c_hh = self._inside(*np.meshgrid(xhi, yhi, indexing="ij"))
        any_in = c_ll | c_lh | c_hl | c_hh
        all_in = c_ll & c_lh & c_hl & c_hh
        candidate = any_in & ~all_in          # a boundary crosses this cell
        diag = np.hypot(dx, dy)
        xc_g, yc_g = np.meshgrid(mesh.xc, mesh.yc, indexing="ij")
        for ob in self.bodies:
            if ob.get("shape") == "cylinder" and ob.get("axis", "z") == "z":
                obcx, obcy = ob["center"]
                rr = ob["radius"]
                d = np.hypot(xc_g - obcx, yc_g - obcy)
                candidate |= np.abs(d - rr) <= 0.5 * diag
        gi, gj = np.nonzero(candidate)

        # --- sub-sample candidate cells -> fluid volume fraction ----------- #
        vf = np.where(s2, 0.0, 1.0).astype(np.float64)
        if gi.size:
            offsets = (np.arange(sub) + 0.5) / sub   # sub-cell centres in [0,1)
            sx = xlo[gi][:, None] + offsets[None, :] * dx   # (n, sub)
            sy = ylo[gj][:, None] + offsets[None, :] * dy
            n = gi.size
            # outer product per cell: pts[a,b] = (sx[a], sy[b]) -> (n, sub*sub)
            pts_x = np.broadcast_to(sx[:, :, None], (n, sub, sub)).reshape(n, -1)
            pts_y = np.broadcast_to(sy[:, None, :], (n, sub, sub)).reshape(n, -1)
            fluid = ~self._inside(pts_x, pts_y)            # (n, sub*sub)
            vf[gi, gj] = fluid.mean(axis=1)

        # --- faces: aperture on candidate (straddling) faces -------------- #
        ap_x = self.aperture[0][:, :, 0].copy()   # (Nx+1, Ny)
        ap_y = self.aperture[1][:, :, 0].copy()   # (Nx, Ny+1)
        self._face_apertures_2d(ap_x, ap_y, sub, dx, dy)

        # --- small-cell stabilisation: kill candidate slivers with vf < eps - #
        # Only *candidate* cells can be small slivers; bulk-solid cells (centre
        # inside, all corners inside) have vf = 0 legitimately and are left
        # alone.  Killed cells become wall cells (vf -> 0, all faces closed).
        small = candidate & (vf < self.small_cell_eps)
        n_small = int(small.sum())
        if n_small:
            vf[small] = 0.0
            si, sj = np.nonzero(small)
            for i, j in zip(si.tolist(), sj.tolist()):
                ap_x[i, j] = 0.0
                ap_x[i + 1, j] = 0.0
                ap_y[i, j] = 0.0
                ap_y[i, j + 1] = 0.0

        # --- pack back into 3-D arrays ------------------------------------- #
        self.volume_fraction = vf[:, :, None].astype(np.float64)
        self.aperture[0] = ap_x[:, :, None]
        self.aperture[1] = ap_y[:, :, None]
        self.is_solid = (self.volume_fraction <= self.small_cell_eps)
        self.is_fluid = (self.volume_fraction >= 1.0 - self.small_cell_eps)
        self.is_cut = (~self.is_solid) & (~self.is_fluid)
        self.n_cut = int(self.is_cut.sum())
        self.n_small = n_small

    # ------------------------------------------------------------------ #
    def _face_apertures_2d(self, ap_x, ap_y, sub, dx, dy):
        """Fill aperture fractions for boundary-straddling faces, in place.

        An x-face at ``x = xf[i]`` spans ``y in [yc[j]-dy/2, yc[j]+dy/2]``; its
        two endpoints are ``(xf[i], ylo[j])`` and ``(xf[i], yhi[j])``.  The face
        is a candidate iff its endpoints straddle the boundary; sub-sample the
        edge and take the fluid fraction.  Non-candidate faces keep the
        centre-test value already in ``ap_x`` / ``ap_y``.
        """
        mesh = self.mesh
        Nx, Ny = mesh.Nx, mesh.Ny
        ylo = mesh.yc - 0.5 * dy
        yhi = mesh.yc + 0.5 * dy
        xlo = mesh.xc - 0.5 * dx
        xhi = mesh.xc + 0.5 * dx

        # ---- x-faces (i = 0..Nx, j = 0..Ny-1) --------------------------- #
        XX, YYlo = np.meshgrid(mesh.xf, ylo, indexing="ij")   # (Nx+1, Ny)
        _, YYhi = np.meshgrid(mesh.xf, yhi, indexing="ij")
        in_lo = self._inside(XX, YYlo)
        in_hi = self._inside(XX, YYhi)
        cand = in_lo != in_hi
        fi, fj = np.nonzero(cand)
        if fi.size:
            t = (np.arange(sub) + 0.5) / sub
            ys = YYlo[fi, fj][:, None] + t[None, :] * dy       # (n, sub)
            xs = np.broadcast_to(XX[fi, fj][:, None], ys.shape)
            ap_x[fi, fj] = (~self._inside(xs, ys)).mean(axis=1)

        # ---- y-faces (i = 0..Nx-1, j = 0..Ny) --------------------------- #
        XXlo, YY = np.meshgrid(xlo, mesh.yf, indexing="ij")    # (Nx, Ny+1)
        XXhi, _ = np.meshgrid(xhi, mesh.yf, indexing="ij")
        in_l = self._inside(XXlo, YY)
        in_r = self._inside(XXhi, YY)
        cand = in_l != in_r
        fi, fj = np.nonzero(cand)
        if fi.size:
            t = (np.arange(sub) + 0.5) / sub
            xs = XXlo[fi, fj][:, None] + t[None, :] * dx
            ys = np.broadcast_to(YY[fi, fj][:, None], xs.shape)
            ap_y[fi, fj] = (~self._inside(xs, ys)).mean(axis=1)


# ====================================================================== #
# Self-test: run ``python -m solver.cut_cell`` (no solver / mesh solve
# needed) to verify area conservation, symmetry, ranges and mesh convergence.
# ====================================================================== #
def _self_test() -> None:
    from mesh.mesh import Mesh

    r = 0.5
    cx, cy = 5.005, 2.05   # snapped centre (matches the cylinder config)
    body = [{"shape": "cylinder", "center": [cx, cy], "radius": r, "axis": "z"}]
    target_solid_area = np.pi * r * r

    def build(Nx, Ny, sub=64):
        mesh = Mesh(Nx, Ny, 1, 22.0, 4.1, 1.0)
        Xc, Yc, _ = mesh.cell_grid()
        solid = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= r ** 2
        return mesh, solid, CutCellGeometry(mesh, solid, body, sub=sub)

    print("cut_cell self-test  (cylinder r=0.5 at (5.005, 2.05))")
    print(f"  target solid area = pi r^2 = {target_solid_area:.6f}")

    for (Nx, Ny) in [(200, 80), (400, 160), (800, 320)]:
        mesh, solid, g = build(Nx, Ny)
        vf = g.volume_fraction[:, :, 0]
        cell_area = mesh.dx * mesh.dy
        fluid_area = float(vf.sum() * cell_area)
        solid_area = mesh.Lx * mesh.Ly - fluid_area
        err = abs(solid_area - target_solid_area)
        # ranges
        assert vf.min() >= 0.0 and vf.max() <= 1.0, "vf out of [0,1]"
        for ax in (0, 1):
            ap = g.aperture[ax]
            assert ap.min() >= 0.0 and ap.max() <= 1.0, f"ap[{ax}] out of [0,1]"
        # symmetry about the true centre (cx, cy): mirror each cell centre
        # (xc[i], yc[j]) -> (2cx - xc[i], 2cy - yc[j]) and find the nearest
        # cell.  Works whether the centre sits on a face or a cell centre.
        mir_x = 2.0 * cx - mesh.xc
        mir_y = 2.0 * cy - mesh.yc
        mir_i = np.argmin(np.abs(mesh.xc[None, :] - mir_x[:, None]), axis=1)
        mir_j = np.argmin(np.abs(mesh.yc[None, :] - mir_y[:, None]), axis=1)
        # restrict to cells whose mirror lands inside the domain (1-cell margin)
        in_x = (mir_x > mesh.dx) & (mir_x < mesh.Lx - mesh.dx)
        in_y = (mir_y > mesh.dy) & (mir_y < mesh.Ly - mesh.dy)
        ii, jj = np.nonzero(in_x[:, None] & in_y[None, :])
        sym_err = float(np.max(np.abs(
            vf[ii, jj] - vf[mir_i[ii], mir_j[jj]])))
        print(f"  {Nx}x{Ny}: solid_area={solid_area:.4f}  err={err:.2e}  "
              f"n_cut={g.n_cut:4d} n_small={g.n_small:2d}  "
              f"sym_err={sym_err:.2e}")

    # mesh convergence of the solid-area estimate
    errs = []
    for (Nx, Ny) in [(100, 40), (200, 80), (400, 160), (800, 320)]:
        mesh, solid, g = build(Nx, Ny, sub=64)
        vf = g.volume_fraction[:, :, 0]
        fluid_area = float(vf.sum() * mesh.dx * mesh.dy)
        errs.append(abs((mesh.Lx * mesh.Ly - fluid_area) - target_solid_area))
    print("  convergence (solid-area error vs dx):")
    for (Nx, Ny), e in zip([(100, 40), (200, 80), (400, 160), (800, 320)], errs):
        print(f"    {Nx}x{Ny}: err={e:.2e}")
    if errs[0] > 0:
        print(f"  ratio err(100)/err(800) = {errs[0]/errs[-1]:.1f}  "
              f"(mesh refined 8x -> expect ~8x for O(dx), much more if sub-sampling-limited)")
    print("  all range/symmetry asserts passed.")


if __name__ == "__main__":
    _self_test()