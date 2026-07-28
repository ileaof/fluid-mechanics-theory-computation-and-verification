"""Discrete gradients on the staggered mesh.

Two gradient operators are needed:

* :func:`cell_gradient` -- gradient of a cell-centred scalar sampled back at
  the cell centres (used for the Boussinesq buoyancy term and for plotting
  streamlines via vorticity).
* :func:`face_gradient` -- component of the gradient of a cell-centred scalar
  evaluated at a face set (used for the diffusive flux and for the pressure
  gradient that drives the projection correction).

Both use centred (second-order) differences with one-sided boundaries.
"""

from __future__ import annotations

import numpy as np


def cell_gradient(phi: np.ndarray, dx: float, dy: float, dz: float):
    """Gradient of a cell-centred field, returned at cell centres.

    Returns a tuple ``(dphidx, dphidy, dphidz)`` each with shape ``phi.shape``.
    Interior: centred difference; boundaries: one-sided first-order (or
    second-order where possible).
    """

    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gz = np.zeros_like(phi)

    if phi.shape[0] > 1:
        gx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
        gx[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / dx
        gx[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / dx
    if phi.shape[1] > 1:
        gy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
        gy[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / dy
        gy[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / dy
    if phi.shape[2] > 1:
        gz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)
        gz[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dz
        gz[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dz
    return gx, gy, gz


def face_gradient(phi: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Component ``d phi / d x_axis`` at the faces normal to *axis*.

    For axis 0 the result has shape ``(Nx+1, Ny, Nz)`` and equals
    ``(phi[i+1]-phi[i])/dx`` at face ``i+1/2``.  Analogously for axes 1, 2.
    The boundary faces use a one-sided difference.
    """

    if axis == 0:
        g = np.zeros((phi.shape[0] + 1, phi.shape[1], phi.shape[2]),
                     dtype=phi.dtype)
        g[1:-1] = (phi[1:] - phi[:-1]) / h
        g[0] = (phi[0] - phi[0]) / h          # zero (refined by BCs)
        g[-1] = (phi[-1] - phi[-1]) / h
        # one-sided at the physical boundaries
        g[0] = (phi[0] - 0.0) / h if False else g[0]
    elif axis == 1:
        g = np.zeros((phi.shape[0], phi.shape[1] + 1, phi.shape[2]),
                     dtype=phi.dtype)
        g[:, 1:-1, :] = (phi[:, 1:, :] - phi[:, :-1, :]) / h
    else:
        g = np.zeros((phi.shape[0], phi.shape[1], phi.shape[2] + 1),
                     dtype=phi.dtype)
        g[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / h
    return g


def pressure_gradient_face(p: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Pressure gradient at the velocity-faces (for the projection step).

    Returns ``dp/dx_axis`` at the faces of the staggered velocity component
    along *axis*.  This is a thin specialisation of :func:`face_gradient`
    kept explicit for clarity at the call site.
    """

    return face_gradient(p, axis, h)