"""Energy (heat-transport) solver.

Solves the advection-diffusion equation for the temperature:

.. math::
    \\rho c_p\\,\\frac{\\partial T}{\\partial t}
    + \\rho c_p\\,(\\mathbf{u}\\!\\cdot\\!\\nabla) T
    \\;=\\; \\nabla\\!\\cdot\\!(k\\,\\nabla T) .

Dividing by ``rho cp`` and writing the thermal diffusivity
``alpha = k/(rho cp)``:

.. math::
    \\frac{\\partial T}{\\partial t}
    + (\\mathbf{u}\\!\\cdot\\!\\nabla) T
    \\;=\\; \\alpha\\,\\nabla^{2} T .

The same semi-implicit structure as the momentum predictor is used: explicit
advection (with the configured convection scheme) and implicit diffusion,
solved with the linear solver.  The boundary conditions for the temperature
(Dirichlet / heat-flux / adiabatic) are baked into the diffusion matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from numerics.interpolation import face_interpolate
from numerics.divergence import divergence
from .boundary import PATCHES, PATCH_AXIS


class EnergySolver:
    """Advection-diffusion solver for the temperature field."""

    def __init__(self, mesh, fluid, boundary, cfg, linear_solver) -> None:
        self.mesh = mesh
        self.fluid = fluid
        self.bc = boundary
        self.cfg = cfg
        self.ls = linear_solver
        self.scheme = cfg.convection.lower()
        self.limiter = getattr(cfg, "limiter", "vanleer")
        self.time_scheme = cfg.time_scheme.lower()
        self.theta = 1.0 if "implicit" in self.time_scheme else 0.5
        self._L: sp.csr_matrix | None = None
        self._Lrhs: np.ndarray | None = None
        # Cache M = I - theta*dt*alpha*L keyed on the scalar coefficient so the
        # matrix (and its ILU factorisation) is built once for fixed-dt runs.
        self._M_cache: dict[float, sp.csr_matrix] = {}

    # ------------------------------------------------------------------ #
    def _diff_matrix(self):
        if self._L is None:
            k = self.fluid.k
            A, rhs = self.bc.scalar_laplacian_matrix(
                self.mesh.cell_shape, self.mesh.dx, self.mesh.dy,
                self.mesh.dz, k_cond=k)
            self._L = A
            self._Lrhs = rhs
        return self._L, self._Lrhs

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

    def _advective_flux(self, T, comp, axis):
        uf = self._face_velocity(comp, axis)
        Tf = face_interpolate(T, uf, axis, scheme=self.scheme,
                              limiter_name=self.limiter)
        # boundary face values: fixed-temperature walls impose the wall T on
        # the ghost so the transported value at the boundary face reflects it.
        for patch, spec in self.bc.temperature.items():
            if patch not in PATCHES or PATCH_AXIS[patch] != axis:
                continue
            lo = patch in ("west", "south", "bottom")
            sl = [slice(None)] * 3
            sl[axis] = 0 if lo else Tf.shape[axis] - 1
            if spec.kind == "fixed":
                Tf[tuple(sl)] = spec.value
            # adiabatic / heatflux: zero advective flux (wall velocity = 0)
        return uf * Tf

    # ------------------------------------------------------------------ #
    def advection(self, T, u, v, w):
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        two_d = self.mesh.is_2d
        Fx = self._advective_flux(T, u, 0)
        Fy = self._advective_flux(T, v, 1)
        Fz = self._advective_flux(T, w, 2) if not two_d else None
        return divergence(Fx, Fy, Fz, dx, dy, dz, two_d=two_d)

    # ------------------------------------------------------------------ #
    def step(self, T, u, v, w, dt) -> np.ndarray:
        """Advance the temperature by one time step."""
        alpha = self.fluid.alpha_thermal  # k/(rho cp)
        adv = self.advection(T, u, v, w)
        L, Lrhs = self._diff_matrix()
        N = L.shape[0]
        coeff = self.theta * dt * alpha
        if np.ndim(coeff) == 0:
            key = float(coeff)
            M = self._M_cache.get(key)
            if M is None:
                I = sp.eye(N, format="csr")
                M = I - key * L
                self._M_cache[key] = M
        else:
            I = sp.eye(N, format="csr")
            M = I - coeff * L
        rhs = T.ravel().copy() - dt * adv.ravel()
        rhs += self.theta * dt * alpha * Lrhs
        if self.theta < 1.0:
            rhs += (1.0 - self.theta) * dt * alpha * (L @ T.ravel())
        x = self.ls.solve(M, rhs, x0=T.ravel())
        return x.reshape(self.mesh.cell_shape)