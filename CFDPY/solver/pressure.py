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

from numerics.gradients import face_gradient
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
            vals.append(c)
            vals.append(c)
            vals.append(-c)
            vals.append(-c)
        data = np.concatenate(vals)
        N = mesh.Nx * mesh.Ny * mesh.Nz
        A = sp.coo_matrix((data, (self._rows, self._cols)),
                          shape=(N, N)).tocsr()
        self._A = A
        return A

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
    def solve(self, us, vs, ws, dt, rho):
        """Solve for the pressure increment ``delta_p``.

        Returns ``(delta_p, face_fluxes)``.  The caller adds ``delta_p`` to the
        old pressure and applies the velocity correction.  The constant null
        space of the operator is removed by mean-projecting the RHS before the
        solve and the solution afterwards.
        """

        mesh = self.mesh
        A = self._matrix(rho)
        Fx, Fy, Fz = self._face_fluxes(us, vs, ws)
        div_us = divergence(Fx, Fy, Fz, mesh.dx, mesh.dy, mesh.dz,
                           two_d=mesh.is_2d)
        rhs = div_us.ravel() / dt
        # Project the RHS onto the zero-mean subspace (remove the constant null
        # space) so the singular symmetric system has a consistent solution.
        rhs = rhs - rhs.mean()
        dp_vec = self.ls.solve(A, rhs, x0=np.zeros_like(rhs))
        dp = dp_vec.reshape(mesh.cell_shape)
        # The solution is determined up to a constant; fix the mean to zero.
        dp = dp - dp.mean()
        return dp, (Fx, Fy, Fz)