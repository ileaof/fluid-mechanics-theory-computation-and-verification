"""Post-processing of derived quantities.

:class:`PostProcessor` computes diagnostic fields and scalar metrics from the
raw solver state:

* **vorticity** ``omega = d v/dx - d u/dy`` (cell-centred);
* **streamfunction** ``psi`` obtained by integrating ``-omega`` with a Poisson
  solve (handy for streamline plotting in 2D);
* **Nusselt number** on a hot wall -- the dimensionless heat-transfer rate,

  .. math::
     \\overline{Nu} \\;=\\; -\\int_{0}^{H}\\!\\frac{\\partial T^{*}}{\\partial x}
        \\Big|_{x=0}\\!dy,
     \\qquad T^{*}=\\frac{T-T_c}{T_h-T_c},

  reduced to a discrete sum over the wall-adjacent cells.

All methods are pure functions of the field state and the mesh.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numerics.gradients import cell_gradient
from numerics.laplacian import poisson_matrix


class PostProcessor:
    """Derive vorticity, streamfunction and Nusselt number from the state."""

    def __init__(self, mesh) -> None:
        self.mesh = mesh
        self._psi_A = None

    # ------------------------------------------------------------------ #
    def vorticity(self, u, v) -> np.ndarray:
        """2D vorticity ``omega = dv/dx - du/dy`` (cell-centred)."""
        gx, gy, _ = cell_gradient(v, self.mesh.dx, self.mesh.dy, self.mesh.dz)
        gxu, gy_u, _ = cell_gradient(u, self.mesh.dx, self.mesh.dy, self.mesh.dz)
        return gx - gy_u

    # ------------------------------------------------------------------ #
    def streamfunction(self, u, v) -> np.ndarray:
        """Solve  nabla^2 psi = -omega  with psi=0 on the boundary (2D)."""
        mesh = self.mesh
        if self._psi_A is None:
            A = poisson_matrix(mesh.Nx, mesh.Ny, mesh.Nz, mesh.dx, mesh.dy,
                               mesh.dz, two_d=True)
            # Dirichlet psi = 0 on all walls: replace boundary rows by identity.
            Nx, Ny = mesh.Nx, mesh.Ny
            A = A.tolil()
            for i in range(Nx):
                for j in (0, Ny - 1):
                    r = (i * Ny + j)
                    A.rows[r] = [r]; A.data[r] = [1.0]
            for j in range(Ny):
                for i in (0, Nx - 1):
                    r = (i * Ny + j)
                    A.rows[r] = [r]; A.data[r] = [1.0]
            self._psi_A = A.tocsr()
        omega = self.vorticity(u, v)
        rhs = -omega.ravel()
        # enforce homogeneous Dirichlet on boundary cells
        Nx, Ny = mesh.Nx, mesh.Ny
        for i in range(Nx):
            rhs[i * Ny + 0] = 0.0
            rhs[i * Ny + (Ny - 1)] = 0.0
        for j in range(Ny):
            rhs[0 * Ny + j] = 0.0
            rhs[(Nx - 1) * Ny + j] = 0.0
        psi = spla.spsolve(self._psi_A, rhs)
        return psi.reshape(mesh.cell_shape)

    # ------------------------------------------------------------------ #
    def nusselt_wall(self, T, T_hot: float, T_cold: float, side: str = "west"
                     ) -> float:
        """Mean Nusselt number on the hot wall (default: west).

        Uses the one-sided temperature gradient at the wall-adjacent cell row
        and the non-dimensional temperature ``T* = (T-T_cold)/(T_hot-T_cold)``.
        """
        mesh = self.mesh
        dT = T_hot - T_cold
        if abs(dT) < 1e-30:
            return 0.0
        Tstar = (T - T_cold) / dT
        if side == "west":
            # -dT*/dx at x=0, one-sided (forward) over the first cell
            grad = (Tstar[1, :, 0] - Tstar[0, :, 0]) / mesh.dx
            Nu_local = -grad
            length = mesh.Ly
            return float(np.sum(Nu_local) * mesh.dy / length)
        if side == "east":
            grad = (Tstar[-1, :, 0] - Tstar[-2, :, 0]) / mesh.dx
            Nu_local = grad
            length = mesh.Ly
            return float(np.sum(Nu_local) * mesh.dy / length)
        # generic axis mapping
        raise ValueError(f"Unsupported wall side: {side}")