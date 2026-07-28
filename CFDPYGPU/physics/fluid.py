"""Fluid (single-phase and two-phase) thermophysical model.

The :class:`Fluid` class wraps the properties required by the momentum and
energy equations: density ``rho``, dynamic viscosity ``mu``, specific heat
``cp`` and thermal conductivity ``k``.  When a VOF simulation is run two sets
of properties are stored (heavy/light phases) and the cell-centred effective
properties are blended linearly with the VOF marker:

.. math::
    \\rho_{\\mathrm{eff}} = \\alpha\\,\\rho_{\\mathrm{heavy}} +
                           (1-\\alpha)\\,\\rho_{\\mathrm{light}}.

This linear blending is exact for the mixture density and is the standard
one-fluid formulation for the momentum coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Fluid:
    """Thermophysical state of the working fluid(s).

    The heavy phase (``rho``) is the primary fluid; the light phase
    (``rho_light``) is the secondary fluid used in VOF runs (e.g. water+air).
    """

    rho: float = 1000.0
    mu: float = 1.0e-3
    cp: float = 4180.0
    k: float = 0.6
    beta: float = 0.0           # Boussinesq thermal-expansion coefficient
    t_ref: float = 300.0

    # Secondary (light) phase -- air by default.
    rho_light: float = 1.2
    mu_light: float = 1.8e-5
    cp_light: float = 1005.0
    k_light: float = 0.026

    # ------------------------------------------------------------------ #
    @property
    def nu(self) -> float:
        """Kinematic viscosity of the heavy phase ``mu/rho``."""
        return self.mu / self.rho

    @property
    def nu_light(self) -> float:
        return self.mu_light / self.rho_light

    @property
    def alpha_thermal(self) -> float:
        """Thermal diffusivity ``k/(rho cp)`` of the heavy phase."""
        return self.k / (self.rho * self.cp)

    # ------------------------------------------------------------------ #
    def blend(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """Return effective (rho, mu, cp, k) blended by the VOF marker.

        Parameters
        ----------
        alpha:
            Cell-centred volume fraction of the heavy phase in ``[0, 1]``.
        """

        a = np.clip(alpha, 0.0, 1.0)
        rho = a * self.rho + (1.0 - a) * self.rho_light
        mu = a * self.mu + (1.0 - a) * self.mu_light
        cp = a * self.cp + (1.0 - a) * self.cp_light
        k = a * self.k + (1.0 - a) * self.k_light
        return rho, mu, cp, k

    @classmethod
    def from_config(cls, cfg) -> "Fluid":
        return cls(
            rho=cfg.rho, mu=cfg.mu, cp=cfg.cp, k=cfg.k,
            beta=getattr(cfg, "beta", 0.0), t_ref=getattr(cfg, "t_ref", 300.0),
            rho_light=cfg.rho_light, mu_light=cfg.mu_light,
        )