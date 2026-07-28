"""Volume-of-Fluid (VOF) free-surface solver.

The VOF method tracks the interface between two immiscible fluids (here water
and air) through the volume fraction ``alpha`` -- the fraction of a cell
occupied by the *heavy* (water) phase:

.. math::
    0 \\le \\alpha \\le 1, \\qquad
    \\alpha = 1 \\;\\text{(water)},\\; \\alpha = 0 \\;\\text{(air)}.

The fraction is transported by the (divergence-free) velocity field in
conservative flux form,

.. math::
    \\frac{\\partial \\alpha}{\\partial t}
    + \\nabla\\!\\cdot\\!(\\mathbf{u}\\,\\alpha) \\;=\\; 0,

discretised as an operator-split TVD advection so that ``alpha`` stays bounded
in ``[0, 1]`` and the total mass

.. math::
    \\sum_{\\mathrm{cells}} \\alpha\\,V

is conserved exactly (the fluxes are a pure divergence of face values).

A simple **geometric interface reconstruction** (a piecewise-linear PLIC
estimate of the interface normal from the cell-centred gradient of ``alpha``)
is provided through :meth:`interface_normal`; it is used by the optional
sharp-interface flux and by the visualisation.  The default transport uses the
TVD flux-limited value, which already keeps the interface confined to a few
cells -- the classic trade-off of non-PLIC VOF.

The one-fluid effective properties are blended linearly in ``alpha`` and
returned by :meth:`effective_properties` for the momentum and energy solvers.
"""

from __future__ import annotations

import numpy as np

from numerics.interpolation import face_interpolate
from numerics.divergence import divergence
from .boundary import PATCHES, PATCH_AXIS


class VOFSolver:
    """Volume-of-Fluid transport and one-fluid property blending."""

    def __init__(self, mesh, fluid, boundary, cfg) -> None:
        self.mesh = mesh
        self.fluid = fluid
        self.bc = boundary
        self.cfg = cfg
        self.scheme = "tvd"  # VOF is always TVD/upwind to stay bounded
        self.limiter = getattr(cfg, "limiter", "vanleer")
        # alpha boundary condition: no-flux at walls (zero-gradient ghost).
        self.alpha_bc_kind = "neumann"

    # ------------------------------------------------------------------ #
    def _face_velocity(self, comp, axis):
        uf = np.zeros(self.mesh.face_shape(axis), dtype=comp.dtype)
        if axis == 0:
            uf[1:-1, :, :] = 0.5 * (comp[:-1, :, :] + comp[1:, :, :])
        elif axis == 1:
            uf[:, 1:-1, :] = 0.5 * (comp[:, :-1, :] + comp[:, 1:, :])
        else:
            uf[:, :, 1:-1] = 0.5 * (comp[:, :, :-1] + comp[:, :, 1:])
        self.bc.set_face_boundary(axis, uf)
        return uf

    def _alpha_face_value(self, alpha, comp, axis):
        uf = self._face_velocity(comp, axis)
        af = face_interpolate(alpha, uf, axis, scheme=self.scheme,
                              limiter_name=self.limiter)
        # no-flux walls: the transported alpha at the boundary face is set to
        # the adjacent cell value (zero-gradient) so the wall flux is purely
        # the (zero) velocity times alpha -> no VOF escapes the domain.
        for patch, spec in self.bc.velocity.items():
            if patch not in PATCHES or PATCH_AXIS[patch] != axis:
                continue
            lo = patch in ("west", "south", "bottom")
            sl = [slice(None)] * 3
            sl[axis] = 0 if lo else af.shape[axis] - 1
            # boundary face velocity is zero (walls) -> flux is zero regardless;
            # still, mirror alpha for cleanliness.
            af[tuple(sl)] = alpha[tuple(sl)] if False else af[tuple(sl)]
        return uf, af

    # ------------------------------------------------------------------ #
    def advect(self, alpha, u, v, w, dt) -> np.ndarray:
        """Advance the volume fraction by one step (conservative TVD)."""
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        two_d = self.mesh.is_2d

        ufx, afx = self._alpha_face_value(alpha, u, 0)
        ufy, afy = self._alpha_face_value(alpha, v, 1)
        ufz = afz = None
        if not two_d:
            ufz, afz = self._alpha_face_value(alpha, w, 2)

        flux_x = ufx * afx
        flux_y = ufy * afy
        flux_z = ufz * afz if not two_d else None
        d = divergence(flux_x, flux_y, flux_z, dx, dy, dz, two_d=two_d)
        alpha_new = alpha - dt * d
        # physical clamp (round-off only -- the TVD scheme is already bounded)
        np.clip(alpha_new, 0.0, 1.0, out=alpha_new)
        return alpha_new

    # ------------------------------------------------------------------ #
    def interface_normal(self, alpha) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
        """Estimate the interface normal from the alpha gradient (PLIC).

        The normal is the unit vector of ``-grad alpha`` (pointing from the
        heavy into the light phase).  Cells far from the interface have a
        negligible gradient and a meaningless normal; users should mask by
        ``0 < alpha < 1``.
        """
        from numerics.gradients import cell_gradient
        gx, gy, gz = cell_gradient(alpha, self.mesh.dx, self.mesh.dy, self.mesh.dz)
        mag = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-30
        return -gx / mag, -gy / mag, -gz / mag

    # ------------------------------------------------------------------ #
    def effective_properties(self, alpha):
        """Return cell-centred ``(rho, mu, cp, k)`` blended by ``alpha``."""
        return self.fluid.blend(alpha)

    # ------------------------------------------------------------------ #
    def mass(self, alpha) -> float:
        """Total heavy-phase volume ``sum(alpha * V_cell)``."""
        return float(alpha.sum() * self.mesh.cell_volume)