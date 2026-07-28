"""Discrete divergence on the staggered mesh.

For a cell ``(i, j, k)`` the conservative divergence of a face flux field is

.. math::
    (\\nabla\\!\\cdot\\!\\mathbf{F})_{i,j,k} \\;=\\;
    \\frac{F^x_{i+1/2} - F^x_{i-1/2}}{\\Delta x} +
    \\frac{F^y_{j+1/2} - F^y_{j-1/2}}{\\Delta y} +
    \\frac{F^z_{k+1/2} - F^z_{k-1/2}}{\\Delta z}

where ``F^x`` lives on the x-faces ``(Nx+1,Ny,Nz)``, etc.  This is the natural
complement of :func:`numerics.interpolation.face_interpolate`: the convective
term is reconstructed as ``divergence(face_flux)`` with the flux equal to the
face velocity times the interpolated scalar.
"""

from __future__ import annotations

import numpy as np


def divergence(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
               dx: float, dy: float, dz: float,
               two_d: bool = False) -> np.ndarray:
    """Divergence of a face-flux field, returned at cell centres.

    Parameters
    ----------
    fx, fy, fz:
        Flux components on the x-, y-, z-faces respectively.  In 2D ``fz`` may
        be ``None``.
    dx, dy, dz:
        Cell sizes.
    two_d:
        If ``True`` the z-contribution is skipped.
    """

    shape = (fx.shape[0] - 1, fy.shape[1] - 1, fx.shape[2])
    out = np.zeros(shape, dtype=fx.dtype)

    out = (fx[1:, :, :] - fx[:-1, :, :]) / dx
    out = out + (fy[:, 1:, :] - fy[:, :-1, :]) / dy
    if not two_d and fz is not None:
        out = out + (fz[:, :, 1:] - fz[:, :, :-1]) / dz
    return out