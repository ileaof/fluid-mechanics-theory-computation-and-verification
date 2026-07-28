"""Gravity body force.

Stores the gravity vector ``g`` and offers a per-axis accessor used by the
momentum predictor.  In the one-fluid (VOF) formulation the *net* gravity
source on the momentum equation is

.. math::
    \\mathbf{f}_{g} \\;=\\; \\rho_{\\mathrm{eff}}\\,\\mathbf{g},

i.e. the full weight, because the pressure equation already carries the
hydrostatic reference.  For Boussinesq convection the density perturbation is
handled separately by :class:`physics.buoyancy.BoussinesqBuoyancy`.
"""

from __future__ import annotations

import numpy as np


class Gravity:
    """Constant gravitational acceleration vector."""

    def __init__(self, gx: float = 0.0, gy: float = -9.81, gz: float = 0.0) -> None:
        self.vec = np.array([float(gx), float(gy), float(gz)], dtype=np.float64)

    @property
    def x(self) -> float:
        return float(self.vec[0])

    @property
    def y(self) -> float:
        return float(self.vec[1])

    @property
    def z(self) -> float:
        return float(self.vec[2])

    def component(self, axis: int) -> float:
        return float(self.vec[axis])

    def body_force(self, rho_eff: np.ndarray, axis: int) -> np.ndarray:
        """Per-cell gravitational source ``rho_eff * g_axis``."""
        return rho_eff * self.vec[axis]

    @classmethod
    def from_config(cls, cfg) -> "Gravity":
        g = tuple(cfg.gravity) if hasattr(cfg, "gravity") else (0.0, -9.81, 0.0)
        return cls(g[0], g[1], g[2] if len(g) > 2 else 0.0)