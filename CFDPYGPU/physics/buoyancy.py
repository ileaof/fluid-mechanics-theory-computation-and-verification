"""Boussinesq buoyancy source term.

In the Boussinesq approximation the density is constant everywhere *except*
in the gravitational body force, where the temperature perturbation produces a
buoyant source:

.. math::
    \\rho \\approx \\rho_{0}\\bigl[1 - \\beta\\,(T - T_{0})\\bigr],
    \\qquad
    \\mathbf{f}_{b} \\;=\\; \\rho_{0}\\,\\beta\\,(T - T_{0})\\,(-\\mathbf{g}).

Equivalently the buoyant acceleration added to the momentum equation is
``-beta (T - T0) g_vec``.  The sign is chosen so that a *hot* fluid in a
downward gravity field rises.
"""

from __future__ import annotations

import numpy as np


class BoussinesqBuoyancy:
    """Buoyancy source from the Boussinesq approximation."""

    def __init__(self, beta: float, t0: float, gravity: tuple[float, float, float]) -> None:
        self.beta = float(beta)
        self.t0 = float(t0)
        self.g = np.asarray(gravity, dtype=np.float64)

    def acceleration(self, T: np.ndarray, axis: int) -> np.ndarray:
        """Buoyant acceleration along *axis* at cell centres.

        ``a = -beta (T - T0) g[axis]``  (positive when hot fluid rises
        against gravity).
        """

        return -self.beta * (T - self.t0) * self.g[axis]

    @classmethod
    def from_config(cls, cfg, gravity_vec) -> "BoussinesqBuoyancy":
        return cls(beta=cfg.beta, t0=cfg.t_ref, gravity=gravity_vec)