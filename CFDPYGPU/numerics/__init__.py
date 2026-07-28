"""Numerical operators package.

Discrete finite-volume operators for the staggered mesh:

* :mod:`interpolation` -- face-value reconstruction (convection schemes);
* :mod:`gradients`     -- cell & face gradients (centred differences);
* :mod:`divergence`    -- divergence of face fluxes;
* :mod:`laplacian`     -- sparse Poisson matrix assembly;
* :mod:`timestep`      -- CFL-based adaptive time-step control.

All operators are pure functions of the mesh plus the field(s); they hold no
state and are reused by every solver module.
"""

from .interpolation import face_interpolate, limiter, tvd_face_value
from .gradients import cell_gradient, face_gradient
from .divergence import divergence
from .laplacian import poisson_matrix
from .timestep import compute_dt

__all__ = [
    "face_interpolate", "limiter", "tvd_face_value",
    "cell_gradient", "face_gradient",
    "divergence", "poisson_matrix", "compute_dt",
]