"""Time-step selection.

For explicit convection the stability limit is the CFL condition

.. math::
    \\Delta t \\;<\\; \\mathrm{CFL}\\,
       \\min_{f}\\!\\left(\\frac{\\Delta x}{|u|},
                        \\frac{\\Delta y}{|v|},
                        \\frac{\\Delta z}{|w|}\\right),

combined with the diffusion (Fourier) limit

.. math::
    \\Delta t \\;<\\; \\frac{1}{2\\nu\\,(1/\\Delta x^{2}+1/\\Delta y^{2}+1/\\Delta z^{2})} .

:func:`compute_dt` returns a stable time-step bounded by
``[dt_min, dt_max]`` and the user target ``dt``.  It is only consulted when
``adaptive_dt`` is enabled; otherwise the fixed user ``dt`` is used.
"""

from __future__ import annotations

import numpy as np


def compute_dt(u, v, w, dx, dy, dz, nu, *,
              cfl: float = 0.5, fo: float = 0.25,
              dt_target: float = 1e-3,
              dt_min: float = 1e-7, dt_max: float = 1e-2) -> float:
    """Return a stable adaptive time-step.

    Parameters
    ----------
    u, v, w:
        Velocity components (w may be ``None`` in 2D).
    dx, dy, dz:
        Cell sizes.
    nu:
        Kinematic diffusivity used for the Fourier limit.
    cfl, fo:
        Safety coefficients on the convective and diffusive limits.
    dt_target:
        User-preferred time-step (never exceeded).
    dt_min, dt_max:
        Hard bounds.
    """

    umax = max(float(np.abs(u).max()), 1e-30)
    vmax = max(float(np.abs(v).max()), 1e-30)
    wmax = max(float(np.abs(w).max()), 1e-30) if w is not None else 0.0

    inv_conv = umax / dx + vmax / dy + (wmax / dz if w is not None else 0.0)
    dt_conv = 1.0 / max(inv_conv, 1e-30)

    inv_diff = 2.0 * max(nu, 1e-30) * (
        1.0 / dx**2 + 1.0 / dy**2 + (1.0 / dz**2 if w is not None else 0.0))
    dt_diff = 1.0 / max(inv_diff, 1e-30)

    dt = min(cfl * dt_conv, fo * dt_diff, dt_target, dt_max)
    return max(dt, dt_min)