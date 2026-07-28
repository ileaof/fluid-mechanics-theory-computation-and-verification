"""Boundary conditions for the staggered mesh.

The :class:`BoundaryCondition` object owns the boundary specification of the
three primary fields -- velocity, pressure and temperature -- for the six
patches of the Cartesian box (``west, east, south, north, bottom, top``).

Implementation notes
--------------------
* **Velocity** lives on faces; only the *normal* component has a boundary
  face that coincides with a physical patch.  The tangential component's
  no-slip condition is enforced inside the diffusion Laplacian via a ghost
  mirror (``u_ghost = -u_first`` for a zero wall value).  Therefore
  :meth:`apply_velocity` only writes the normal-component boundary faces.
* **Temperature** is cell-centred; Dirichlet/Neumann data enter the discrete
  Laplacian through a *ghost-cell* extrapolation
  (``T_ghost = 2 T_wall - T_first`` for a fixed temperature,
  ``T_ghost = T_first + q dx/k`` for a heat flux,
  ``T_ghost = T_first`` for adiabatic).  This is encapsulated by
  :meth:`temperature_ghost` (explicit path) and baked into the assembled
  matrix :meth:`scalar_laplacian_matrix` (implicit path).
* **Pressure** uses homogeneous Neumann (zero normal gradient) on all
  impermeable walls; a *pressure outlet* imposes Dirichlet ``p = p_out`` on
  the adjacent cell row (handled in the Poisson assembly).

Supported velocity kinds: ``no-slip, slip, inlet, outlet, symmetry, periodic``.
Supported temperature kinds: ``fixed, heatflux, adiabatic``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from config.config_loader import BoundarySpec


PATCHES = ("west", "east", "south", "north", "bottom", "top")
# Axis and sign of the outward normal for each patch.
PATCH_AXIS = {"west": 0, "east": 0, "south": 1, "north": 1,
              "bottom": 2, "top": 2}
PATCH_SIGN = {"west": -1, "east": +1, "south": -1, "north": +1,
              "bottom": -1, "top": +1}
# Component index whose normal face lies on each patch.
PATCH_COMP = {"west": 0, "east": 0, "south": 1, "north": 1,
              "bottom": 2, "top": 2}


def _patch_index(patch: str, N: tuple[int, int, int]) -> tuple[int, int, int]:
    """Boundary slice index along the patch axis, returned as (axis, lo, hi)."""
    axis = PATCH_AXIS[patch]
    lo, hi = 0, N[axis]  # face arrays have N[axis]+1 along axis; cells have N[axis]
    return axis, lo, hi


class BoundaryCondition:
    """Boundary-condition manager for one simulation case."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.velocity = dict(cfg.velocity_bc)
        self.pressure = dict(cfg.pressure_bc)
        self.temperature = dict(cfg.temperature_bc)

        # Default every velocity patch to no-slip if not specified.
        for p in PATCHES:
            self.velocity.setdefault(p, BoundarySpec("no-slip", 0.0))
        # Default temperature patches to adiabatic.
        for p in PATCHES:
            self.temperature.setdefault(p, BoundarySpec("adiabatic", 0.0))
        # Pressure: default Neumann (no outlet).
        for p in PATCHES:
            self.pressure.setdefault(p, BoundarySpec("neumann", 0.0))

        self.is_2d = cfg.two_d

        # Immersed solid mask (cell-centred boolean array).  ``None`` means no
        # obstacle.  Set externally via :meth:`set_solid`; the face-flux routines
        # consult :meth:`mask_solid_faces` to enforce no-penetration.
        self.solid: np.ndarray | None = None
        self._solid_face: dict[int, np.ndarray] = {}
        # Curved-boundary IBM forcing (mirror-point ghost cells).  ``None`` (or
        # an inactive forcing) means the staircase direct-forcing is used -- see
        # :meth:`apply_immersion`.  Set externally via :meth:`set_immersion`.
        self.immersion = None

    # ================================================================== #
    # Velocity BCs
    # ================================================================== #
    def patch_vel_spec(self, patch: str) -> BoundarySpec:
        return self.velocity[patch]

    # ------------------------------------------------------------------ #
    # Immersed solid (blocked cells)
    # ------------------------------------------------------------------ #
    def set_solid(self, solid: np.ndarray | None) -> None:
        """Register the cell-centred solid mask (``True`` inside an obstacle).

        Resets the cached face masks; they are rebuilt lazily by
        :meth:`solid_face_mask`.
        """
        self.solid = None if solid is None else np.asarray(solid, dtype=bool)
        self._solid_face = {}

    @property
    def has_solid(self) -> bool:
        return self.solid is not None and bool(self.solid.any())

    def solid_face_mask(self, axis: int) -> np.ndarray | None:
        """Boolean face array: ``True`` where a face is *blocked* by a solid cell.

        An interior face is blocked when either of its two neighbouring cells
        is solid; a boundary face is blocked when its only adjacent cell is
        solid.  A blocked face must carry zero normal flux (no-penetration).
        """
        if self.solid is None:
            return None
        if axis in self._solid_face:
            return self._solid_face[axis]
        s = self.solid
        if axis == 0:
            mask = np.zeros((s.shape[0] + 1, s.shape[1], s.shape[2]), dtype=bool)
            mask[1:-1] = s[:-1] | s[1:]
            mask[0], mask[-1] = s[0], s[-1]
        elif axis == 1:
            mask = np.zeros((s.shape[0], s.shape[1] + 1, s.shape[2]), dtype=bool)
            mask[:, 1:-1] = s[:, :-1] | s[:, 1:]
            mask[:, 0], mask[:, -1] = s[:, 0], s[:, -1]
        else:
            mask = np.zeros((s.shape[0], s.shape[1], s.shape[2] + 1), dtype=bool)
            mask[:, :, 1:-1] = s[:, :, :-1] | s[:, :, 1:]
            mask[:, :, 0], mask[:, :, -1] = s[:, :, 0], s[:, :, -1]
        self._solid_face[axis] = mask
        return mask

    def mask_solid_faces(self, axis: int, face: np.ndarray) -> None:
        """Zero the blocked faces of a face-centred field, in place."""
        mask = self.solid_face_mask(axis)
        if mask is not None:
            face[mask] = 0.0

    def set_immersion(self, immersion) -> None:
        """Register a curved-boundary IBM forcing object (or ``None``).

        When set and :attr:`immersion.active`, :meth:`apply_immersion` delegates
        to it; otherwise the staircase clamp is used.
        """
        self.immersion = immersion

    def apply_immersion(self, field: np.ndarray) -> None:
        """Enforce the immersed-boundary no-slip on a cell-centred velocity.

        If a mirror-point IBM is configured and active, it overrides the ghost
        cells (no-slip at the true wall) and zeroes the deep interior.  Otherwise
        fall back to the staircase direct-forcing: clamp every solid cell to
        zero.  No-op when there is no solid and no active IBM.
        """
        if self.immersion is not None and self.immersion.active:
            self.immersion.apply(field)
        elif self.has_solid:
            field[self.solid] = 0.0

    def apply_velocity(self, u: np.ndarray, v: np.ndarray,
                       w: np.ndarray | None) -> None:
        """Write the normal-component boundary face values in place.

        For ``inlet``/``no-slip``/``slip``/``symmetry`` the normal velocity is
        prescribed directly (0 for impermeable walls and symmetry).  For
        ``outlet`` a zero-gradient extrapolation is used.  Periodic patches
        are left untouched (the operators handle wrap-around).
        """

        comps = (u, v, w)
        for patch, spec in self.velocity.items():
            if patch not in PATCHES:
                continue
            axis = PATCH_AXIS[patch]
            comp = comps[axis]
            if comp is None:
                continue
            kind = spec.kind
            if kind == "periodic":
                continue
            # Face index along this axis at the boundary.
            if patch in ("west", "south", "bottom"):
                idx = 0
            else:
                idx = comp.shape[axis] - 1

            sl = [slice(None)] * 3
            sl[axis] = idx
            sl = tuple(sl)

            if kind in ("no-slip", "slip", "symmetry"):
                comp[sl] = 0.0
            elif kind == "inlet":
                comp[sl] = spec.value
            elif kind == "outlet":
                # Zero-gradient: copy the first interior face value.
                inner = list(sl)
                inner[axis] = 1 if idx == 0 else comp.shape[axis] - 2
                comp[sl] = comp[tuple(inner)]

    # ================================================================== #
    # Temperature BCs (ghost-cell form)
    # ================================================================== #
    def temperature_ghost(self, T: np.ndarray, dx: float, dy: float,
                          dz: float, k_cond: float) -> np.ndarray:
        """Return ``T`` padded with one ghost layer satisfying the BCs.

        Used by the explicit Laplacian.  For a 2D case the z-ghost is a size-1
        repeat (no z-flux).
        """

        Nx, Ny, Nz = T.shape
        pad = np.empty((Nx + 2, Ny + 2, Nz + 2), dtype=T.dtype)
        pad[1:-1, 1:-1, 1:-1] = T
        h = (dx, dy, dz)

        def fill(patch: str, spec: BoundarySpec) -> None:
            axis = PATCH_AXIS[patch]
            n = T.shape[axis]
            interior = [slice(1, -1)] * 3
            ghost = [slice(1, -1)] * 3
            if patch in ("west", "south", "bottom"):
                interior[axis] = 1       # first physical cell
                ghost[axis] = 0          # ghost layer
                sign = -1.0
                first = 1
            else:
                interior[axis] = -2
                ghost[axis] = -1
                sign = +1.0
                first = -2
            hi = h[axis]
            if spec.kind == "fixed":
                pad[tuple(ghost)] = 2.0 * spec.value - pad[tuple(interior)]
            elif spec.kind == "heatflux":
                # outward normal flux q = -k dT/dn  ->  dT/dn = -q/k
                # ghost = interior + (dT/dn) * hi, with sign of outward normal
                dTdn = -spec.value / max(k_cond, 1e-30)
                pad[tuple(ghost)] = pad[tuple(interior)] + sign * dTdn * hi
            else:  # adiabatic
                pad[tuple(ghost)] = pad[tuple(interior)]

        for patch, spec in self.temperature.items():
            if patch not in PATCHES:
                continue
            if spec.kind == "periodic":
                continue
            fill(patch, spec)

        # If a direction has size 1 (2D z), just reflect (no flux).
        if Nz == 1:
            pad[:, :, 0] = pad[:, :, 1]
            pad[:, :, -1] = pad[:, :, -2]
        return pad

    # ================================================================== #
    # Generic structured Laplacian matrix (cell-centred, collocated)
    # ================================================================== #
    @staticmethod
    def build_laplacian(shape: tuple[int, int, int],
                        h: tuple[float, float, float],
                        kinds: dict, values: dict):
        """Assemble a cell-centred Laplacian with per-axis-side BCs.

        Parameters
        ----------
        shape:
            Field shape ``(Nx, Ny, Nz)``.
        h:
            Cell sizes ``(dx, dy, dz)``.
        kinds:
            ``kinds[axis][side]`` with ``side in (-1, +1)`` (low/high) and value
            in ``{"dirichlet", "neumann", "periodic"}``.
        values:
            ``values[axis][side]`` -- the Dirichlet value (used only to build
            the known RHS contribution; 0 for Neumann/periodic).

        Returns
        -------
        (A, rhs):  ``A`` is the sparse Laplacian (size ``prod(shape)``) and
        ``rhs`` the known vector from non-homogeneous Dirichlet/flux data.

        Boundary-stencil conventions (ghost cell):

        - ``dirichlet`` (ghost ``= 2 Vw - phi``): diagonal ``-3/h^2``, RHS
          ``+2 Vw/h^2``;
        - ``neumann`` (ghost ``= phi``): diagonal ``-1/h^2``;
        - ``periodic``: the boundary cell connects to the opposite boundary
          cell (wrap-around off-diagonal).
        """

        Nx, Ny, Nz = shape
        N = Nx * Ny * Nz
        inv = [1.0 / (h[a] * h[a]) for a in range(3)]

        rows, cols, vals = [], [], []
        rhs = np.zeros(N)

        def flat(i, j, k):
            return (i * Ny + j) * Nz + k

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    r = flat(i, j, k)
                    diag = 0.0
                    for a in range(3):
                        idx = (i, j, k)[a]
                        n_a = shape[a]
                        coeff = inv[a]
                        diag += -2.0 * coeff            # -2 phi_i term
                        for side in (-1, +1):
                            at_boundary = (idx == 0 and side == -1) or \
                                          (idx == n_a - 1 and side == +1)
                            kind = kinds[a][side]
                            if at_boundary:
                                if kind == "dirichlet":
                                    diag += -coeff      # ghost contributes -phi
                                    rhs[r] += 2.0 * coeff * values[a][side]
                                elif kind == "neumann":
                                    diag += +coeff      # ghost = phi
                                else:  # periodic: wrap to opposite boundary
                                    opp = n_a - 1 if side == -1 else 0
                                    nb = [i, j, k]; nb[a] = opp
                                    rows.append(r); cols.append(flat(*nb))
                                    vals.append(coeff)
                            else:
                                nb = [i, j, k]
                                nb[a] = idx + (1 if side == +1 else -1)
                                rows.append(r); cols.append(flat(*nb))
                                vals.append(coeff)
                    rows.append(r); cols.append(r); vals.append(diag)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        return A, rhs

    # ------------------------------------------------------------------ #
    def _scalar_kinds(self) -> tuple[dict, dict]:
        """Translate the temperature BCs to (kinds, values) for the builder."""
        kinds = {a: {-1: "neumann", +1: "neumann"} for a in range(3)}
        values = {a: {-1: 0.0, +1: 0.0} for a in range(3)}
        for patch, spec in self.temperature.items():
            if patch not in PATCHES:
                continue
            a = PATCH_AXIS[patch]
            side = -1 if PATCH_SIGN[patch] < 0 else +1
            if spec.kind == "periodic":
                kinds[a][side] = "periodic"
            elif spec.kind == "fixed":
                kinds[a][side] = "dirichlet"
                values[a][side] = spec.value
            else:  # adiabatic / heatflux -> Neumann (flux handled in rhs_flux)
                kinds[a][side] = "neumann"
        return kinds, values

    def scalar_laplacian_matrix(self, shape, dx, dy, dz, k_cond=1.0):
        """Laplacian for a temperature-like scalar (Dirichlet/Neumann/periodic)."""
        h = (dx, dy, dz)
        kinds, values = self._scalar_kinds()
        A, rhs = self.build_laplacian(shape, h, kinds, values)
        rhs = rhs + self._heatflux_rhs(shape, h, k_cond)
        return A, rhs

    def _heatflux_rhs(self, shape, h, k_cond) -> np.ndarray:
        """Known RHS from prescribed heat-flux walls (Neumann with non-zero q)."""
        Nx, Ny, Nz = shape
        rhs = np.zeros(Nx * Ny * Nz)
        for patch, spec in self.temperature.items():
            if patch not in PATCHES or spec.kind != "heatflux":
                continue
            a = PATCH_AXIS[patch]
            side = -1 if PATCH_SIGN[patch] < 0 else +1
            hi = h[a]
            sign = float(side)
            dTdn = -spec.value / max(k_cond, 1e-30)
            # rhs contribution on the boundary row: sign * dTdn / h
            def flat(i, j, k):
                return (i * Ny + j) * Nz + k
            n_a = shape[a]
            idx = 0 if side == -1 else n_a - 1
            ranges = [range(shape[b]) for b in range(3)]
            for ii in ranges[0]:
                for jj in ranges[1]:
                    for kk in ranges[2]:
                        if (ii, jj, kk)[a] == idx:
                            rhs[flat(ii, jj, kk)] += sign * dTdn / hi
        return rhs

    # ------------------------------------------------------------------ #
    def velocity_laplacian(self, comp_axis: int, shape, dx, dy, dz):
        """Laplacian for a velocity *component* (collocated, cell-centred).

        For each patch the treatment depends on whether the component is the
        *normal* (use the normal BC) or *tangential* (use the tangential BC):

        - no-slip  : all components Dirichlet(0);
        - slip/symmetry: normal Dirichlet(0), tangential Neumann;
        - inlet    : normal Dirichlet(u_in), tangential Neumann;
        - outlet   : all Neumann;
        - periodic : wrap.
        """

        h = (dx, dy, dz)
        kinds = {a: {-1: "neumann", +1: "neumann"} for a in range(3)}
        values = {a: {-1: 0.0, +1: 0.0} for a in range(3)}
        for patch, spec in self.velocity.items():
            if patch not in PATCHES:
                continue
            a = PATCH_AXIS[patch]
            side = -1 if PATCH_SIGN[patch] < 0 else +1
            kind = spec.kind
            is_normal = (a == comp_axis)
            if kind == "periodic":
                kinds[a][side] = "periodic"
            elif kind == "outlet":
                kinds[a][side] = "neumann"
            elif kind == "inlet":
                if is_normal:
                    kinds[a][side] = "dirichlet"
                    values[a][side] = spec.value
                else:
                    kinds[a][side] = "neumann"
            else:  # no-slip, slip, symmetry
                if is_normal or kind == "no-slip":
                    kinds[a][side] = "dirichlet"
                    values[a][side] = 0.0
                else:
                    kinds[a][side] = "neumann"
        return self.build_laplacian(shape, h, kinds, values)

    # ================================================================== #
    # Pressure outlet / periodic helpers
    # ================================================================== #
    def pressure_outlet_patches(self) -> list[str]:
        return [p for p in PATCHES
                if self.pressure.get(p, BoundarySpec("neumann")).kind == "outlet"]

    def is_periodic(self) -> bool:
        return any(self.velocity[p].kind == "periodic" for p in PATCHES)

    # ================================================================== #
    # Boundary face values for the explicit convective flux (collocated)
    # ================================================================== #
    def set_face_boundary(self, comp_axis: int, face: np.ndarray,
                          interior_field: np.ndarray | None = None) -> None:
        """Write the prescribed normal velocity on the boundary faces, in place.

        ``face`` has shape ``mesh.face_shape(comp_axis)``; the boundary slices
        are at index ``0`` and ``-1`` along ``comp_axis``.  Walls/symmetry set
        zero, inlets set ``u_in``, outlets extrapolate from the first interior
        face, and periodic patches wrap the two ends together.
        """

        for patch, spec in self.velocity.items():
            if patch not in PATCHES or PATCH_AXIS[patch] != comp_axis:
                continue
            kind = spec.kind
            lo = patch in ("west", "south", "bottom")
            axis = comp_axis
            sl = [slice(None)] * 3
            sl[axis] = 0 if lo else face.shape[axis] - 1
            sl = tuple(sl)
            if kind == "periodic":
                # wrap: low face = interior-field cell 0 mirror handled elsewhere
                continue
            if kind in ("no-slip", "slip", "symmetry"):
                face[sl] = 0.0
            elif kind == "inlet":
                face[sl] = spec.value
            elif kind == "outlet":
                inner = list(sl)
                inner[axis] = 1 if lo else face.shape[axis] - 2
                face[sl] = face[tuple(inner)]
        # periodic wrap (last step): equate the two boundary faces
        for patch, spec in self.velocity.items():
            if patch not in PATCHES or PATCH_AXIS[patch] != comp_axis:
                continue
            if spec.kind != "periodic":
                continue
            axis = comp_axis
            face[(0, slice(None), slice(None)) if axis == 0 else
                 (slice(None), 0, slice(None)) if axis == 1 else
                 (slice(None), slice(None), 0)] = \
                face[(face.shape[axis] - 1, slice(None), slice(None)) if axis == 0
                     else (slice(None), face.shape[axis] - 1, slice(None)) if axis == 1
                     else (slice(None), slice(None), face.shape[axis] - 1)]
        # Finally, zero any face blocked by an immersed solid (no-penetration
        # through an obstacle boundary, including the inlet region covered by a
        # step).
        self.mask_solid_faces(comp_axis, face)