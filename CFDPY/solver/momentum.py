"""Momentum (incompressible Navier-Stokes) predictor.

The semi-implicit fractional-step predictor advances the cell-centred
velocity from time ``n`` to an intermediate ``u*``:

.. math::
    \\frac{\\mathbf{u}^{*} - \\mathbf{u}^{n}}{\\Delta t} \\;=\\;
    -\\,(\\mathbf{u}^{n}\\!\\cdot\\!\\nabla)\\,\\mathbf{u}^{n}
    \\;+\\; \\theta\\,\\nu\\,\\nabla^{2}\\mathbf{u}^{*}
    \\;+\\; (1-\\theta)\\,\\nu\\,\\nabla^{2}\\mathbf{u}^{n}
    \\;+\\; \\mathbf{f}^{n}.

The convective term is treated **explicitly** (the chosen scheme -- upwind,
central, QUICK or TVD -- only affects the face-value reconstruction, so the
explicit treatment does not change the linear-algebra structure).  The
diffusive term is treated **implicitly** with the time-stepping coefficient

* ``theta = 1``       -- implicit (backward) Euler,
* ``theta = 0.5``      -- Crank-Nicolson.

The implicit diffusion is a sparse linear solve per component handled by the
:class:`~solver.linear_solver.LinearSolver` (CG / BiCGSTAB / GMRES + ILU).

The predictor *excludes* the pressure gradient: that enters in the projection
correction (see :mod:`solver.projection`), which is the standard Chorin
splitting.  The gravity / Boussinesq body forces are added here as
``f``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from numerics.interpolation import face_interpolate
from numerics.divergence import divergence
from .boundary import PATCHES, PATCH_AXIS


class MomentumSolver:
    """Semi-implicit momentum predictor on the collocated mesh."""

    def __init__(self, mesh, fluid, boundary, cfg, linear_solver) -> None:
        self.mesh = mesh
        self.fluid = fluid
        self.bc = boundary
        self.cfg = cfg
        self.ls = linear_solver

        scheme = cfg.convection.lower()
        self.scheme = scheme
        self.limiter = getattr(cfg, "limiter", "vanleer")
        self.time_scheme = cfg.time_scheme.lower()
        self.theta = 1.0 if "implicit" in self.time_scheme else 0.5

        # Cache the diffusion Laplacian per component (independent of dt).
        self._L: dict[int, sp.csr_matrix] = {}
        self._L_rhs: dict[int, np.ndarray] = {}
        # Cache the assembled implicit-diffusion matrix M = I - theta*dt*nu*L
        # keyed on the scalar coefficient theta*dt*nu.  For fixed-dt,
        # constant-viscosity runs (the common case) this is invariant across
        # steps, so the matrix -- and its ILU factorisation -- is built once.
        self._M_cache: dict[tuple[int, float], sp.csr_matrix] = {}

    # ------------------------------------------------------------------ #
    # Diffusion operator (implicit)
    # ------------------------------------------------------------------ #
    def _diff_matrix(self, comp_axis: int) -> tuple[sp.csr_matrix, np.ndarray]:
        if comp_axis not in self._L:
            shape = self.mesh.cell_shape
            A, rhs = self.bc.velocity_laplacian(
                comp_axis, shape, self.mesh.dx, self.mesh.dy, self.mesh.dz)
            self._L[comp_axis] = A
            self._L_rhs[comp_axis] = rhs
        return self._L[comp_axis], self._L_rhs[comp_axis]

    # ------------------------------------------------------------------ #
    # Convective term  (u . grad) u   at cell centres
    # ------------------------------------------------------------------ #
    def _face_velocity(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Interpolate a cell-centred field to the faces normal to *axis*.

        Uses centred interpolation in the interior and the BC-prescribed value
        on the boundary faces (the convective flux through a wall is zero, an
        inlet carries its prescribed value, an outlet extrapolates).
        """

        # centred interior face value
        phif = np.zeros(self.mesh.face_shape(axis), dtype=field.dtype)
        if axis == 0:
            phif[1:-1, :, :] = 0.5 * (field[:-1, :, :] + field[1:, :, :])
        elif axis == 1:
            phif[:, 1:-1, :] = 0.5 * (field[:, :-1, :] + field[:, 1:, :])
        else:
            phif[:, :, 1:-1] = 0.5 * (field[:, :, :-1] + field[:, :, 1:])
        self.bc.set_face_boundary(axis, phif)
        # Zero the faces blocked by an immersed solid (no-penetration): this
        # also kills the centred-interpolated value at solid/fluid interfaces,
        # which a plain cell clamp would leave as a half-value leak.
        self.bc.mask_solid_faces(axis, phif)
        return phif

    def _convective_flux(self, field: np.ndarray, axis: int) -> np.ndarray:
        """Face flux ``F = (mass flux normal to face) * (field at face)``.

        The mass flux normal to a face of axis *a* is the *a*-component of
        velocity interpolated to that face.  The transported value is rebuilt
        with the selected convection scheme.
        """

        # mass flux (face velocity of the same component)
        mass = np.zeros(self.mesh.face_shape(axis), dtype=field.dtype)
        comp = (self._u, self._v, self._w)[axis]
        mass = self._face_velocity(comp, axis)

        # transported value at the face, chosen by the convection scheme
        phif = face_interpolate(field, mass, axis, scheme=self.scheme,
                                limiter_name=self.limiter)
        # overwrite the boundary face transported value with the BC value of
        # the field itself (so wall/inlet fluxes use the prescribed value).
        self._set_field_face_boundary(axis, phif, field)
        return mass * phif

    def _set_field_face_boundary(self, axis: int, phif: np.ndarray,
                                 field: np.ndarray) -> None:
        """Impose field boundary values on the convective face (no-slip=0)."""
        for patch, spec in self.bc.velocity.items():
            if patch not in PATCHES:
                continue
            if PATCH_AXIS[patch] != axis:
                continue
            kind = spec.kind
            lo = patch in ("west", "south", "bottom")
            sl = [slice(None)] * 3
            sl[axis] = 0 if lo else phif.shape[axis] - 1
            if kind == "inlet":
                phif[tuple(sl)] = spec.value
            elif kind in ("no-slip", "slip", "symmetry"):
                phif[tuple(sl)] = 0.0
            elif kind == "outlet":
                inner = list(sl); inner[axis] = 1 if lo else phif.shape[axis] - 2
                phif[tuple(sl)] = phif[tuple(inner)]

    def convective_term(self, u, v, w) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
        """Return ``(u.grad) u, (u.grad) v, (u.grad) w`` at cell centres."""
        self._u, self._v, self._w = u, v, w
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        two_d = self.mesh.is_2d

        Fx_u = self._convective_flux(u, 0)
        Fy_u = self._convective_flux(u, 1)
        Fz_u = self._convective_flux(u, 2) if not two_d else None
        cu = divergence(Fx_u, Fy_u, Fz_u, dx, dy, dz, two_d=two_d)

        Fx_v = self._convective_flux(v, 0)
        Fy_v = self._convective_flux(v, 1)
        Fz_v = self._convective_flux(v, 2) if not two_d else None
        cv = divergence(Fx_v, Fy_v, Fz_v, dx, dy, dz, two_d=two_d)

        cw = np.zeros_like(u)
        if not two_d:
            Fx_w = self._convective_flux(w, 0)
            Fy_w = self._convective_flux(w, 1)
            Fz_w = self._convective_flux(w, 2)
            cw = divergence(Fx_w, Fy_w, Fz_w, dx, dy, dz, two_d=False)
        return cu, cv, cw

    # ------------------------------------------------------------------ #
    # Diffusion solve per component
    # ------------------------------------------------------------------ #
    def _diffuse_implicit(self, comp_axis: int, field, conv, src, nu, dt):
        """Solve  (I - theta dt nu L) u* = u + dt(-conv + src) + theta dt nu rhs_bc."""
        L, Lrhs = self._diff_matrix(comp_axis)
        N = L.shape[0]
        # Assemble (or fetch the cached) implicit-diffusion matrix.  Caching is
        # only safe for a scalar viscosity; a spatially-varying nu (VOF blend)
        # is rebuilt every call.
        coeff = self.theta * dt * nu
        if np.ndim(coeff) == 0:
            key = (comp_axis, float(coeff))
            M = self._M_cache.get(key)
            if M is None:
                I = sp.eye(N, format="csr")
                M = I - float(coeff) * L
                self._M_cache[key] = M
        else:
            I = sp.eye(N, format="csr")
            M = I - coeff * L
        rhs = field.ravel().copy()
        rhs += dt * (-conv.ravel() + src.ravel())
        # implicit-diffusion boundary contribution (Dirichlet walls/inlets).  The
        # known boundary data (Lrhs) is time-independent, so it enters at both
        # time levels and is scaled by the full dt*nu -- NOT by theta.  Scaling
        # it by theta would under-apply a nonzero Dirichlet velocity (e.g. an
        # inlet) by the implicit fraction.  No-slip walls have Lrhs == 0 so this
        # only matters for prescribed-velocity (inlet) boundaries.
        rhs += dt * nu * Lrhs
        # explicit part of CN diffusion: + (1-theta) dt nu L u^n  added to rhs
        if self.theta < 1.0:
            rhs += (1.0 - self.theta) * dt * nu * (L @ field.ravel())
        x = self.ls.solve(M, rhs, x0=field.ravel())
        return x.reshape(field.shape)

    # ------------------------------------------------------------------ #
    # Public predictor
    # ------------------------------------------------------------------ #
    def predict(self, u, v, w, nu, dt, src_u, src_v, src_w):
        """Advance velocity to the intermediate state ``u*``.

        Parameters
        ----------
        u, v, w:
            Velocity components at time ``n`` (cell-centred).
        nu:
            Effective kinematic viscosity (scalar or array).
        dt:
            Time step.
        src_u, src_v, src_w:
            Cell-centred body-force sources (gravity + Boussinesq).

        Returns
        -------
        (u*, v*, w*)
        """

        nu_eff = nu if np.ndim(nu) == 0 else nu
        cu, cv, cw = self.convective_term(u, v, w)

        us = self._diffuse_implicit(0, u, cu, src_u, nu_eff, dt)
        vs = self._diffuse_implicit(1, v, cv, src_v, nu_eff, dt)
        ws = self._diffuse_implicit(2, w, cw, src_w, nu_eff, dt) if not self.mesh.is_2d else None
        # Direct forcing: clamp the predicted velocity to zero inside any
        # immersed solid so the pressure Poisson sees zero divergence there
        # (the obstacle acts as a no-slip wall).
        if self.bc.has_solid:
            solid = self.bc.solid
            us[solid] = 0.0
            vs[solid] = 0.0
            if ws is not None:
                ws[solid] = 0.0
        return us, vs, ws