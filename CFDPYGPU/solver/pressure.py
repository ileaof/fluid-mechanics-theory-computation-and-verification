"""Pressure-Poisson solver and pressure-velocity coupling (Rhie-Chow).

The solver implements an **incremental** fractional step on a collocated
grid, which is robust across a sharp density jump (the water/air interface of
a VOF simulation):

1. The momentum predictor already removes the *old* pressure gradient,

   .. math::
      \\mathbf{u}^{*} \\;=\\; \\mathbf{u}^{n}
        + \\Delta t\\bigl[-(\\mathbf{u}\\!\\cdot\\!\\nabla)\\mathbf{u}
        + \\nu\\nabla^{2}\\mathbf{u}
        + \\mathbf{g} - \\tfrac{1}{\\rho}\\nabla p^{n}\\bigr].

2. The pressure **increment** :math:`\\delta p` is obtained from the
   variable-coefficient Poisson equation

   .. math::
      \\nabla\\!\\cdot\\!\\bigl(\\tfrac{1}{\\rho}\\,\\nabla\\delta p\\bigr)
        \\;=\\; \\tfrac{1}{\\Delta t}\\,\\nabla\\!\\cdot\\!\\mathbf{u}^{*},

   built with the face coefficient ``1/rho`` (arithmetic mean of the two cell
   values, equivalently the harmonic mean of rho).  At equilibrium
   (hydrostatic, ``u=0``) the RHS vanishes so :math:`\\delta p` is a constant
   and the old pressure is preserved -- no spurious interface current.

3. The correction

   .. math::
      \\mathbf{u}^{n+1} \\;=\\; \\mathbf{u}^{*}
        - \\tfrac{\\Delta t}{\\rho}\\,\\nabla\\delta p,
      \\qquad
      p^{n+1} \\;=\\; p^{n} + \\delta p

   makes the velocity divergence-free.

For a single-phase (constant ``rho``) run the same path is used with a uniform
density; it reduces to the standard kinematic-pressure projection.

**Null-space handling.**  The Poisson operator has a one-dimensional null
space (the constant pressure field) because only Neumann (zero-gradient)
boundary conditions appear.  Rather than pinning a single matrix row -- which
would alter the sparsity pattern every time the density changes and force an
expensive rebuild -- we keep the *pure* symmetric operator (constant sparsity,
cacheable) and remove the null space analytically: the right-hand side is
projected onto the zero-mean subspace before the solve, and the solution is
mean-subtracted afterwards.  This keeps the matrix structure invariant so only
the coefficient values need to be refreshed when ``rho`` changes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from numerics.gradients import face_gradient, cell_gradient
from numerics.divergence import divergence


class PressureSolver:
    """Incremental pressure-Poisson solve with variable-coefficient matrix.

    The matrix sparsity pattern is assembled once and cached; on every step
    only the coefficient values (the face ``1/rho`` terms) are recomputed and
    written into a fresh CSR matrix.  This makes the variable-density VOF path
    nearly as cheap as the constant-density path.
    """

    def __init__(self, mesh, boundary, cfg, linear_solver) -> None:
        self.mesh = mesh
        self.bc = boundary
        self.cfg = cfg
        self.ls = linear_solver
        self.rhie_chow: bool = bool(getattr(cfg, "rhie_chow", False))
        # Cached sparsity pattern (built on first use).
        self._rows: np.ndarray | None = None
        self._cols: np.ndarray | None = None
        # Per-axis coefficient buffers and their flatten order, used to rebuild
        # only the data array each step.
        self._axis_info: list = []
        # Identity of the rho array last used to build the matrix: when rho is
        # the same object (single-phase uniform density, cached upstream) we
        # reuse the assembled matrix wholesale.
        self._rho_id: int | None = None
        self._A: sp.csr_matrix | None = None
        # Pressure-outlet (Dirichlet p = p_out) cells, derived once from the
        # boundary-condition specs.  When non-empty the Poisson operator has
        # its outlet-cell rows replaced by identity (a Dirichlet pin) and the
        # constant null space disappears, so the mean-projection in solve()
        # is skipped.  See :meth:`BoundaryCondition.pressure_outlet_patches`.
        self._outlet_idx, self._outlet_val = self._collect_outlet_cells()
        # Cut-cell geometry for a curved-body IBM (experimental; see
        # :class:`solver.cut_cell.CutCellGeometry`).  ``None`` -> the standard
        # staircase path (face-flux masking, symmetric Poisson) is used.  When
        # set and active the Poisson operator and divergence use per-cell fluid
        # volume fractions and per-face aperture fractions so the no-penetration
        # wall lands at the true immersed surface.
        self.cc = None
        self._cc_vf: np.ndarray | None = None
        self._cc_is_solid: np.ndarray | None = None
        self._cc_ap_eff: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def set_cut_cell(self, cc) -> None:
        """Register a :class:`CutCellGeometry` for the cut-cell Poisson path.

        Precomputes the per-face *open fraction* ``ap_eff`` -- the aperture
        zeroed on any face that borders a fully-solid (``is_solid``) cell, i.e.
        a wall face -- and stashes the cell volume fraction and solid mask used
        by :meth:`_matrix` and :meth:`solve`.  No-op when ``cc`` is inactive
        (no curved body); the standard staircase path then runs unchanged.
        """
        if cc is None or not cc.has_curve:
            self.cc = None
            self._cc_vf = None
            self._cc_is_solid = None
            self._cc_ap_eff = {}
            return
        self.cc = cc
        self._cc_vf = cc.volume_fraction
        self._cc_is_solid = cc.is_solid
        solid = cc.is_solid
        ap_eff = {}
        # axis 0: face array (Nx+1, Ny, Nz); interior face f couples f-1 & f.
        ax0 = cc.aperture[0].copy()
        ax0[1:-1] = ax0[1:-1] * ~(solid[:-1] | solid[1:])
        ax0[0] = ax0[0] * ~solid[0]
        ax0[-1] = ax0[-1] * ~solid[-1]
        ap_eff[0] = ax0
        # axis 1
        ax1 = cc.aperture[1].copy()
        ax1[:, 1:-1] = ax1[:, 1:-1] * ~(solid[:, :-1] | solid[:, 1:])
        ax1[:, 0] = ax1[:, 0] * ~solid[:, 0]
        ax1[:, -1] = ax1[:, -1] * ~solid[:, -1]
        ap_eff[1] = ax1
        if not self.mesh.is_2d and 2 in cc.aperture:
            ax2 = cc.aperture[2].copy()
            ax2[:, :, 1:-1] = ax2[:, :, 1:-1] * ~(solid[:, :, :-1] | solid[:, :, 1:])
            ax2[:, :, 0] = ax2[:, :, 0] * ~solid[:, :, 0]
            ax2[:, :, -1] = ax2[:, :, -1] * ~solid[:, :, -1]
            ap_eff[2] = ax2
        self._cc_ap_eff = ap_eff
        # the cached matrix (built for the old mask) is no longer valid.
        self._A = None
        self._rho_id = None

    @property
    def cut_cell_active(self) -> bool:
        return self.cc is not None

    # ------------------------------------------------------------------ #
    def _inv_rho_faces(self, rho: np.ndarray):
        """Face values of ``1/rho`` (arithmetic mean of cell 1/rho)."""
        mesh = self.mesh
        ir = 1.0 / rho
        irx = np.zeros(mesh.face_shape(0))
        irx[1:-1, :, :] = 0.5 * (ir[:-1, :, :] + ir[1:, :, :])
        iry = np.zeros(mesh.face_shape(1))
        iry[:, 1:-1, :] = 0.5 * (ir[:, :-1, :] + ir[:, 1:, :])
        irz = None
        if not mesh.is_2d:
            irz = np.zeros(mesh.face_shape(2))
            irz[:, :, 1:-1] = 0.5 * (ir[:, :, :-1] + ir[:, :, 1:])
        return irx, iry, irz

    # ------------------------------------------------------------------ #
    @staticmethod
    def _patch_axis(patch: str) -> int:
        return {"west": 0, "east": 0, "south": 1, "north": 1,
                "bottom": 2, "top": 2}[patch]

    def _collect_outlet_cells(self):
        """Flat (C-order) indices of the boundary-cell row on each pressure
        outlet patch, plus the prescribed ``p_out`` per cell.

        Returns ``(None, 0.0)`` when no ``pressure_bc`` patch is of kind
        ``outlet``.  The Dirichlet rows these indices identify are pinned in
        :meth:`_matrix`; in :meth:`solve` the RHS there is set to
        ``p_out - p_old`` so the pressure *increment* brings the cell to
        ``p_out`` and ``p^{n+1} = p_out``.
        """
        bc = self.bc
        mesh = self.mesh
        patches = bc.pressure_outlet_patches()
        if not patches:
            return None, np.array(0.0)
        low = {"west", "south", "bottom"}
        N = (mesh.Nx, mesh.Ny, mesh.Nz)
        idx_list, val_list = [], []
        for p in patches:
            axis = self._patch_axis(p)
            spec = bc.pressure.get(p)
            p_out = spec.value if spec is not None else 0.0
            fixed = (N[axis] - 1) if p not in low else 0
            mask = np.zeros(mesh.cell_shape, dtype=bool)
            if axis == 0:
                mask[fixed, :, :] = True
            elif axis == 1:
                mask[:, fixed, :] = True
            else:
                mask[:, :, fixed] = True
            flat = np.flatnonzero(mask)  # C-order flat indices
            idx_list.append(flat)
            val_list.append(np.full(flat.shape, p_out, dtype=np.float64))
        idx = np.concatenate(idx_list)
        vals = np.concatenate(val_list)
        return idx, vals

    @property
    def has_outlet(self) -> bool:
        return self._outlet_idx is not None and len(self._outlet_idx) > 0

    # ------------------------------------------------------------------ #
    def _build_pattern(self) -> None:
        """Assemble the constant sparsity pattern (rows/cols) of the operator.

        For every interior face along an axis we add four entries to the
        symmetric stencil -- ``+c`` on the off-diagonals (L, R) and ``-c`` on
        the two diagonals -- where ``c = (1/rho)_face / h^2``.  The values are
        filled later by :meth:`_matrix`; here we only record where they go.
        """
        mesh = self.mesh
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        shape = (Nx, Ny, Nz)

        def rav(i, j, k):
            return np.ravel_multi_index((i, j, k), shape)

        rows, cols = [], []
        self._axis_info = []

        def add_axis(axis):
            n = shape[axis]
            if n < 2:
                return
            idx = np.arange(1, n)
            if axis == 0:
                I, J, K = np.meshgrid(idx, np.arange(Ny), np.arange(Nz),
                                      indexing="ij")
                L = rav(I - 1, J, K).ravel()
                R = rav(I, J, K).ravel()
            elif axis == 1:
                I, J, K = np.meshgrid(np.arange(Nx), idx, np.arange(Nz),
                                      indexing="ij")
                L = rav(I, J - 1, K).ravel()
                R = rav(I, J, K).ravel()
            else:
                I, J, K = np.meshgrid(np.arange(Nx), np.arange(Ny), idx,
                                      indexing="ij")
                L = rav(I, J, K - 1).ravel()
                R = rav(I, J, K).ravel()
            # store the flat (L, R) pair for this axis so we can rebuild the
            # data values cheaply each step.
            self._axis_info.append((L, R))
            # +c on (L,R) and (R,L); -c on (L,L) and (R,R).
            rows.append(L); cols.append(R)
            rows.append(R); cols.append(L)
            rows.append(L); cols.append(L)
            rows.append(R); cols.append(R)

        add_axis(0)
        add_axis(1)
        if not mesh.is_2d:
            add_axis(2)

        self._rows = np.concatenate(rows).astype(np.int64)
        self._cols = np.concatenate(cols).astype(np.int64)
        # The pure symmetric operator has a one-dimensional null space (the
        # constant pressure).  We keep it -- the matrix structure is then
        # invariant across steps so only the coefficient data need to be
        # refreshed -- and remove the null space analytically in :meth:`solve`
        # by mean-projecting the RHS before the solve and the solution after.
        # The symmetric PSD system is solved reliably with BiCGSTAB on the
        # mean-free subspace (the predicted velocity field is divergence-free up
        # to the projection tolerance, so the RHS is already nearly mean-free).

    # ------------------------------------------------------------------ #
    def _matrix(self, rho: np.ndarray) -> sp.csr_matrix:
        """Build the variable-coefficient Poisson matrix for the current rho.

        The sparsity pattern is cached; only the coefficient values are
        recomputed (vectorised, no meshgrid) and assembled into a CSR matrix.
        When ``rho`` is the same object as on the previous call (single-phase
        uniform density) the previously assembled matrix is reused wholesale.
        """
        if id(rho) == self._rho_id and self._A is not None:
            return self._A
        self._rho_id = id(rho)
        if self._rows is None:
            self._build_pattern()
        mesh = self.mesh
        hx2, hy2, hz2 = mesh.dx ** 2, mesh.dy ** 2, mesh.dz ** 2
        irx, iry, irz = self._inv_rho_faces(rho)

        vals = []
        for k, (L, R) in enumerate(self._axis_info):
            if k == 0:
                c = (irx[1:mesh.Nx, :, :] / hx2).ravel()
            elif k == 1:
                c = (iry[:, 1:mesh.Ny, :] / hy2).ravel()
            else:
                c = (irz[:, :, 1:mesh.Nz] / hz2).ravel()
            if self.cut_cell_active:
                coeff_L, coeff_R = self._cc_face_coeffs(k, c)
                vals.append(coeff_L)    # + (L,R)
                vals.append(coeff_R)    # + (R,L)
                vals.append(-coeff_L)   # - (L,L)
                vals.append(-coeff_R)   # - (R,R)
            else:
                vals.append(c)
                vals.append(c)
                vals.append(-c)
                vals.append(-c)
        data = np.concatenate(vals)
        N = mesh.Nx * mesh.Ny * mesh.Nz
        A = sp.coo_matrix((data, (self._rows, self._cols)),
                          shape=(N, N)).tocsr()
        # Pressure-outlet Dirichlet: replace the outlet-cell rows by identity
        # (A[i,:] = e_i).  The column couplings (interior rows referencing the
        # outlet node) are left intact -- they carry the neighbour value, which
        # the Dirichlet row now fixes.  Done in LIL, once per matrix build (the
        # single-phase cylinder matrix is built once and cached).
        if self.has_outlet:
            A = A.tolil()
            for r in self._outlet_idx:
                A.rows[int(r)] = [int(r)]
                A.data[int(r)] = [1.0]
            A = A.tocsr()
        self._A = A
        return A

    # ------------------------------------------------------------------ #
    def _cc_face_coeffs(self, axis: int, c: np.ndarray):
        """Per-side Poisson face coefficient for the cut-cell operator.

        For an interior face between cell ``L`` and cell ``R`` with base face
        coefficient ``c = (1/rho)_face / h^2`` and open fraction ``ap_eff``:

        * a **solid** side (``is_solid``) keeps the standard coefficient ``c``
          so its (passive) row stays a well-conditioned Laplacian with
          ``delta_p -> 0``;
        * a **fluid / cut** side takes ``c * ap_eff`` -- the aperture weights
          the open face area, placing the no-penetration wall at the true
          boundary.

        This is the **volume-fraction-free** (flux / aperture-weighted) form:
        the equation is *not* divided by the cell volume fraction ``vf``, so
        the coefficient never carries the ``1/vf`` factor that makes a tiny
        cut cell (``vf -> 0``) ill-conditioned.  The constant null space is
        preserved (every row still sums to zero) and the mean-projection in
        :meth:`solve` is unchanged; the consistent flux-form velocity
        correction in :meth:`solver.projection.ProjectionMethod._correct_cut_cell`
        matches this operator exactly (``D' F^{n+1} = D' F* - dt A' dp``).

        On a fluid-solid face ``ap_eff`` is 0 (zeroed in :meth:`set_cut_cell`),
        so the fluid side couples 0 (a wall) while the solid side still
        references its neighbour (``c``) -- the body is no longer a leakage
        path.  The operator is non-symmetric (``ap_L != ap_R`` on a cut face)
        -> requires BiCGSTAB/GMRES, not CG.
        """
        solid = self._cc_is_solid
        ap = self._cc_ap_eff[axis]
        if axis == 0:
            ap = ap[1:-1, :, :].ravel()
            sL = solid[:-1, :, :].ravel()
            sR = solid[1:, :, :].ravel()
        elif axis == 1:
            ap = ap[:, 1:-1, :].ravel()
            sL = solid[:, :-1, :].ravel()
            sR = solid[:, 1:, :].ravel()
        else:
            ap = ap[:, :, 1:-1].ravel()
            sL = solid[:, :, :-1].ravel()
            sR = solid[:, :, 1:].ravel()
        coeff_L = np.where(sL, c, c * ap)
        coeff_R = np.where(sR, c, c * ap)
        return coeff_L, coeff_R

    # ------------------------------------------------------------------ #
    def _face_fluxes(self, us, vs, ws):
        """Plain interpolated face fluxes of the predicted velocity (with BC)."""
        mesh = self.mesh

        def avg(comp, axis):
            uf = np.zeros(mesh.face_shape(axis), dtype=comp.dtype)
            if axis == 0:
                uf[1:-1, :, :] = 0.5 * (comp[:-1, :, :] + comp[1:, :, :])
            elif axis == 1:
                uf[:, 1:-1, :] = 0.5 * (comp[:, :-1, :] + comp[:, 1:, :])
            else:
                uf[:, :, 1:-1] = 0.5 * (comp[:, :, :-1] + comp[:, :, 1:])
            self.bc.set_face_boundary(axis, uf)
            # No flux through an immersed-solid face (no-penetration).
            self.bc.mask_solid_faces(axis, uf)
            return uf

        Fx = avg(us, 0)
        Fy = avg(vs, 1)
        Fz = avg(ws, 2) if not mesh.is_2d else None
        return Fx, Fy, Fz

    # ------------------------------------------------------------------ #
    def _rhie_chow(self, Fx, Fy, Fz, p_old, dt, rho):
        """Momentum-interpolation (Rhie-Chow) correction to the face fluxes.

        On a collocated grid the face flux ``F = 0.5*(u_L + u_R)`` decouples the
        pressure: a checkerboard pressure has a *cell-centred* central-difference
        gradient of (near) zero, so it never feeds back into the momentum
        predictor and survives.  Rhie-Chow replaces the averaged cell-centre
        pressure gradient that is implicit in the predictor by the *direct* face
        pressure difference when building the face flux used for the divergence.

        For each interior face along axis ``a`` the correction is

        .. math::
           F_{i+1/2} \\;\\mathrel{-}=\\;
             \\frac{\\Delta t}{\\rho_f}\\Bigl[
                \\tfrac{p_R - p_L}{h}
                - \\tfrac{1}{2}(g^p_{L,a} + g^p_{R,a})
             \\Bigr],

        where ``g^p`` is the cell-centre pressure gradient of the *old* pressure
        ``p^n`` (the increment of which has already been removed from the
        predictor ``u*``).  For a smooth pressure the bracket is ~0 (the two
        estimates agree); for a checkerboard pressure the central-difference
        cell gradient vanishes while the face difference is large, so the term
        injects a flux that damps the decoupled mode.

        ``rho`` is uniform for the single-phase cylinder case; the face
        ``1/rho`` (from :meth:`_inv_rho_faces`) is used so the path is also valid
        for a variable-density run.  Solid-blocked faces are re-zeroed at the
        end (the correction may otherwise reintroduce a flux through them).
        """
        mesh = self.mesh
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        gpx, gpy, gpz = cell_gradient(p_old, dx, dy, dz)
        irx, iry, irz = self._inv_rho_faces(rho)
        dt_rho = dt  # multiply by face 1/rho below

        # axis 0 (x-faces), interior faces 1..Nx-1
        direct_x = (p_old[1:, :, :] - p_old[:-1, :, :]) / dx
        avg_grad_x = 0.5 * (gpx[:-1, :, :] + gpx[1:, :, :])
        corr_x = dt_rho * irx[1:-1, :, :] * (direct_x - avg_grad_x)
        Fx[1:-1, :, :] -= corr_x

        # axis 1 (y-faces), interior faces 1..Ny-1
        direct_y = (p_old[:, 1:, :] - p_old[:, :-1, :]) / dy
        avg_grad_y = 0.5 * (gpy[:, :-1, :] + gpy[:, 1:, :])
        corr_y = dt_rho * iry[:, 1:-1, :] * (direct_y - avg_grad_y)
        Fy[:, 1:-1, :] -= corr_y

        # axis 2 (z-faces), interior faces 1..Nz-1 -- 3-D only
        if Fz is not None and irz is not None and mesh.Nz > 1:
            direct_z = (p_old[:, :, 1:] - p_old[:, :, :-1]) / dz
            avg_grad_z = 0.5 * (gpz[:, :, :-1] + gpz[:, :, 1:])
            corr_z = dt_rho * irz[:, :, 1:-1] * (direct_z - avg_grad_z)
            Fz[:, :, 1:-1] -= corr_z

        # The correction can set a non-zero flux on a solid-blocked face; the
        # divergence must still see zero flux there, so re-apply the masks.
        self.bc.mask_solid_faces(0, Fx)
        self.bc.mask_solid_faces(1, Fy)
        if Fz is not None:
            self.bc.mask_solid_faces(2, Fz)

    # ------------------------------------------------------------------ #
    def _cc_divergence(self, Fx, Fy, Fz) -> np.ndarray:
        """Aperture-weighted divergence for the cut-cell path (vf-free form).

        ``div = (ap_eff_x[+] F_x[+] - ap_eff_x[-] F_x[-])/dx + y [+ z]``,
        zeroed on passive ``is_solid`` cells.  No ``1/vf`` volume scaling: this
        is the flux / aperture-weighted divergence that matches the
        vf-free Poisson operator :meth:`_cc_face_coeffs`, so the two share the
        same conditioning and the flux-form velocity correction cancels the
        divergence exactly (``D' F^{n+1} = D' F* - dt A' dp``).  Zeroing the
        *flux* divergence (``D' F = 0``) is equivalent to zeroing the
        per-volume divergence (``D F = D' F / vf = 0``); the net mass flux
        through the cut faces is what is conserved.  For full cells
        (``ap_eff = 1``) this reduces to the standard
        :func:`numerics.divergence.divergence`.
        """
        mesh = self.mesh
        apx = self._cc_ap_eff[0]
        apy = self._cc_ap_eff[1]
        div = ((apx[1:] * Fx[1:] - apx[:-1] * Fx[:-1]) / mesh.dx
               + (apy[:, 1:] * Fy[:, 1:] - apy[:, :-1] * Fy[:, :-1]) / mesh.dy)
        if not mesh.is_2d and Fz is not None and 2 in self._cc_ap_eff:
            apz = self._cc_ap_eff[2]
            div = div + ((apz[:, :, 1:] * Fz[:, :, 1:] - apz[:, :, :-1] * Fz[:, :, :-1])
                         / mesh.dz)
        div[self._cc_is_solid] = 0.0
        return div

    # ------------------------------------------------------------------ #
    def solve(self, us, vs, ws, dt, rho, p_old=None):
        """Solve for the pressure increment ``delta_p``.

        Returns ``(delta_p, face_fluxes)``.  The caller adds ``delta_p`` to the
        old pressure and applies the velocity correction.

        With no pressure outlet the operator is pure-Neumann and has a
        one-dimensional null space (the constant pressure); the constant is
        removed by mean-projecting the RHS before the solve and the solution
        afterwards.  With one or more pressure-outlet (Dirichlet) patches the
        null space is gone -- the outlet rows are pinned in :meth:`_matrix` and
        the RHS there is set to ``p_out - p_old`` so the increment brings the
        cell to ``p_out`` (i.e. ``p^{n+1} = p_out``); the mean projection is
        then skipped.
        """

        mesh = self.mesh
        A = self._matrix(rho)
        Fx, Fy, Fz = self._face_fluxes(us, vs, ws)
        # Rhie-Chow: replace the averaged cell-centre pressure gradient in the
        # face flux with the *direct* face pressure difference.  For a smooth
        # pressure the two agree (no effect); for a checkerboard pressure the
        # central-difference cell gradient vanishes while the face difference is
        # large, so the correction injects a flux that damps the decoupled mode.
        # Uses the old pressure p^n (already removed from the predictor us).
        if self.rhie_chow and p_old is not None:
            self._rhie_chow(Fx, Fy, Fz, p_old, dt, rho)
        if self.cut_cell_active:
            div_us = self._cc_divergence(Fx, Fy, Fz)
        else:
            div_us = divergence(Fx, Fy, Fz, mesh.dx, mesh.dy, mesh.dz,
                               two_d=mesh.is_2d)
        rhs = div_us.ravel() / dt
        if self.has_outlet and p_old is not None:
            # Dirichlet: dp = p_out - p_old  ->  p^{n+1} = p_old + dp = p_out.
            rhs = rhs.copy()
            rhs[self._outlet_idx] = (self._outlet_val
                                     - p_old.ravel()[self._outlet_idx])
            dp_vec = self.ls.solve(A, rhs, x0=np.zeros_like(rhs))
            dp = dp_vec.reshape(mesh.cell_shape)
        else:
            # Project the RHS onto the zero-mean subspace (remove the constant
            # null space) so the singular symmetric system is consistent.
            rhs = rhs - rhs.mean()
            dp_vec = self.ls.solve(A, rhs, x0=np.zeros_like(rhs))
            dp = dp_vec.reshape(mesh.cell_shape)
            # The solution is determined up to a constant; fix the mean to zero.
            dp = dp - dp.mean()
        return dp, (Fx, Fy, Fz)