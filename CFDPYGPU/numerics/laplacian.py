"""Sparse Laplacian / Poisson matrix for the pressure equation.

The projection method requires the solution of the pressure-Poisson equation

.. math::
    \\nabla^{2} p \\;=\\; \\frac{1}{\\Delta t}\\,\\nabla\\!\\cdot\\!\\mathbf{u}^{*}

on a uniform Cartesian staggered grid.  Because the grid is structured the
Laplacian is a sparse matrix with at most 5 (2D) or 7 (3D) non-zeros per row,
written with the standard second-order centred stencil:

.. math::
    \\nabla^{2} p_{i,j,k} \\approx
       \\frac{p_{i+1}-2p_i+p_{i-1}}{\\Delta x^{2}} +
       \\frac{p_{j+1}-2p_j+p_{j-1}}{\\Delta y^{2}} +
       \\frac{p_{k+1}-2p_k+p_{k-1}}{\\Delta z^{2}}.

This module assembles that operator once as a :class:`scipy.sparse.csr_matrix`
with homogeneous Neumann (zero normal gradient) boundaries by default, which
makes the matrix singular.  A single row is replaced by the discrete
mean-pressure constraint ``(1/V)\\int p = 0`` to pin the null space, yielding a
non-singular system.  The velocity projection step enforces the divergence-free
condition exactly at the faces regardless of the boundary treatment.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _flat(i: int, j: int, k: int, Ny: int, Nz: int) -> int:
    return (i * Ny + j) * Nz + k


def poisson_matrix(Nx: int, Ny: int, Nz: int,
                   dx: float, dy: float, dz: float,
                   two_d: bool = False) -> sp.csr_matrix:
    """Assemble the pressure-Laplacian with a pinned mean.

    The returned matrix is square of size ``Nx*Ny*Nz``.  It is built directly in
    COO format for clarity and converted to CSR.
    """

    N = Nx * Ny * Nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 0.0 if two_d else 1.0 / (dz * dz)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                row = _flat(i, j, k, Ny, Nz)
                diag = 0.0
                # x-neighbours
                if i > 0:
                    cols.append(_flat(i - 1, j, k, Ny, Nz)); rows.append(row)
                    vals.append(inv_dx2); diag -= inv_dx2
                if i < Nx - 1:
                    cols.append(_flat(i + 1, j, k, Ny, Nz)); rows.append(row)
                    vals.append(inv_dx2); diag -= inv_dx2
                # y-neighbours
                if j > 0:
                    cols.append(_flat(i, j - 1, k, Ny, Nz)); rows.append(row)
                    vals.append(inv_dy2); diag -= inv_dy2
                if j < Ny - 1:
                    cols.append(_flat(i, j + 1, k, Ny, Nz)); rows.append(row)
                    vals.append(inv_dy2); diag -= inv_dy2
                # z-neighbours
                if not two_d:
                    if k > 0:
                        cols.append(_flat(i, j, k - 1, Ny, Nz)); rows.append(row)
                        vals.append(inv_dz2); diag -= inv_dz2
                    if k < Nz - 1:
                        cols.append(_flat(i, j, k + 1, Ny, Nz)); rows.append(row)
                        vals.append(inv_dz2); diag -= inv_dz2
                rows.append(row); cols.append(row); vals.append(diag)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # Pin the mean pressure to remove the constant null space (Neumann
    # Laplacian is singular).  Replace one row by the averaging operator.
    pin = _flat(Nx // 2, Ny // 2, max(Nz // 2, 0), Ny, Nz)
    # zero the row
    A = A.tolil()
    for c in range(N):
        A[pin, c] = 0.0
    A[pin, :] = np.ones(N) / N          # mean-pressure constraint
    A = A.tocsr()
    return A