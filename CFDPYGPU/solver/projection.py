"""Pressure-velocity coupling: the incremental projection method.

The :class:`ProjectionMethod` class orchestrates the incremental
fractional-step algorithm:

1. **Predict** the intermediate velocity ``u*`` with the *old* pressure
   gradient removed (so ``u*`` is the pressure-free predictor);
2. **Solve** the variable-coefficient pressure-Poisson equation for the
   pressure increment ``delta_p`` via :class:`solver.pressure.PressureSolver`;
3. **Correct** the velocity and accumulate the new pressure,

   .. math::
      \\mathbf{u}^{n+1} \\;=\\; \\mathbf{u}^{*} - \\tfrac{\\Delta t}{\\rho}\\,\\nabla\\delta p,
      \\qquad
      p^{n+1} \\;=\\; p^{n} + \\delta p.

This is the natural extension point for future **SIMPLE**-type segregated
coupling: the same predict / solve / correct skeleton is reused, only the
coefficients and the under-relaxation change.
"""

from __future__ import annotations

import numpy as np

from numerics.gradients import cell_gradient
from numerics.divergence import divergence as div_op
from .momentum import MomentumSolver
from .pressure import PressureSolver


class ProjectionMethod:
    """Incremental projection coupling on the collocated mesh."""

    def __init__(self, mesh, fluid, boundary, cfg, linear_solver) -> None:
        self.mesh = mesh
        self.fluid = fluid
        self.bc = boundary
        self.cfg = cfg
        self.momentum = MomentumSolver(mesh, fluid, boundary, cfg, linear_solver)
        self.pressure = PressureSolver(mesh, boundary, cfg, linear_solver)
        self._rho_uniform: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def step(self, u, v, w, p, dt, sources, rho=None):
        """Perform one incremental projection step.

        Parameters
        ----------
        u, v, w, p:
            Fields at time ``n`` (``w`` is ``None`` in 2D).
        dt:
            Time step.
        sources:
            Tuple ``(src_u, src_v, src_w)`` of cell-centred body-force
            accelerations (gravity + Boussinesq).
        rho:
            Cell-centred density field (uniform array for single-phase).

        Returns
        -------
        dict with the corrected fields, the pressure increment, the face
        fluxes and the divergence residual.
        """

        src_u, src_v, src_w = sources
        if w is None:
            w = np.zeros(self.mesh.cell_shape)
        nu = self.fluid.nu

        # --- 1. remove the old pressure gradient from the predictor ---------
        gpx, gpy, gpz = cell_gradient(p, self.mesh.dx, self.mesh.dy, self.mesh.dz)
        if rho is None:
            if self._rho_uniform is None:
                self._rho_uniform = np.full(self.mesh.cell_shape,
                                            self.fluid.rho, dtype=np.float64)
            rho = self._rho_uniform
        ir = 1.0 / rho
        src_u = src_u - ir * gpx
        src_v = src_v - ir * gpy
        src_w = src_w - ir * gpz

        us, vs, ws = self.momentum.predict(u, v, w, nu, dt, src_u, src_v, src_w)

        # --- 2. pressure increment -----------------------------------------
        # ``p`` is passed so a pressure-outlet (Dirichlet) patch can fix the
        # gauge: the increment there is set to ``p_out - p`` so p^{n+1}=p_out.
        dp, fluxes = self.pressure.solve(us, vs, ws, dt, rho, p_old=p)

        # --- 3. correction -------------------------------------------------
        u_n, v_n, w_n = self._correct(us, vs, ws, dp, dt, ir, rho=rho,
                                      fluxes=fluxes)
        p_new = p + dp

        div = self._divergence_residual(u_n, v_n, w_n)
        return {"u": u_n, "v": v_n, "w": w_n, "p": p_new,
                "dp": dp, "fluxes": fluxes, "div": div}

    # ------------------------------------------------------------------ #
    def _correct(self, us, vs, ws, dp, dt, ir, rho=None, fluxes=None):
        """Velocity correction  u = u* - (dt/rho) grad(delta_p).

        Inside an immersed solid the corrected velocity is re-clamped to the
        no-slip: the pressure correction there only enforces the no-flux
        condition, and any spurious pressure gradient leaking across the
        solid/fluid interface would otherwise re-energise the solid cells.
        With the staircase this is a zero-clamp; with
        ``immersed_method:"ibm"`` it re-applies the mirror-point ghost forcing.

        For the cut-cell path the central-difference gradient is **not** the
        adjoint of the aperture-weighted cut-cell divergence, so applying it
        uniformly leaves an O(1/vf) mass source in cut cells and the field
        blows up; :meth:`_correct_cut_cell` is used instead.
        """
        if self.pressure.cut_cell_active and fluxes is not None:
            return self._correct_cut_cell(us, vs, ws, dp, dt, ir, rho, fluxes)
        gpx, gpy, gpz = cell_gradient(dp, self.mesh.dx, self.mesh.dy, self.mesh.dz)
        u = us - dt * ir * gpx
        v = vs - dt * ir * gpy
        w = ws - dt * ir * gpz if not self.mesh.is_2d else None
        if self.bc.has_solid:
            self.bc.apply_immersion(u)
            self.bc.apply_immersion(v)
            if w is not None:
                self.bc.apply_immersion(w)
        return u, v, w

    # ------------------------------------------------------------------ #
    def _correct_cut_cell(self, us, vs, ws, dp, dt, ir, rho, fluxes):
        """Velocity correction for the cut-cell path.

        The collocated central-difference gradient ``G_central`` is the
        adjoint of the *standard* (staircase) divergence -- consistent for full
        cells, but only approximately consistent with the aperture-weighted
        cut-cell divergence ``D'``.  Two alternatives were tried and rejected:

        * **Flux-form face recovery** (``u = mean(F_target)`` over the whole
          field) applies a ``(1/4,1/2,1/4)`` smoother every step -- ~2.4x the
          physical viscosity on the 200x80 mesh, collapsing the effective Re.
        * **Cut-cell override** (``u_cut = 2 F_target - u_neighbour``) enforces
          the divergence-free face flux exactly but drives the cut-cell
          velocity to large values where the full-cell neighbour's
          central-diff correction disagrees with the face target -- the run
          blows up (``|v|`` spike, adaptive dt -> dt_min).

        The pragmatic choice here is the **standard** correction
        ``u = u* - dt/rho * G_central(dp)`` everywhere.  The cut-cell win is
        carried by the Poisson side: the vf-free aperture-weighted operator
        places the no-penetration wall at the true surface, and the solid
        mask becomes the deep-solid (``vf == 0``) ring so the face-flux
        no-penetration (``mask_solid_faces``) and the IBM ghost forcing sit at
        the inner edge of the cut ring -- a cell closer to the true circle
        than the centre-inside staircase.  The residual ``D'`` in cut cells is
        ``O(1)`` (no ``1/vf``), comparable to the staircase's own residual, and
        does not accumulate (each projection resets it).  IBM ghost forcing is
        re-applied to the solid layer afterwards.
        """
        mesh = self.mesh
        gpx, gpy, gpz = cell_gradient(dp, mesh.dx, mesh.dy, mesh.dz)
        u = us - dt * ir * gpx
        v = vs - dt * ir * gpy
        w = ws - dt * ir * gpz if not mesh.is_2d else None
        if self.bc.has_solid:
            self.bc.apply_immersion(u)
            self.bc.apply_immersion(v)
            if w is not None:
                self.bc.apply_immersion(w)
        return u, v, w

    # ------------------------------------------------------------------ #
    def _divergence_residual(self, u, v, w):
        mesh = self.mesh
        Fx = np.zeros(mesh.face_shape(0))
        Fx[1:-1, :, :] = 0.5 * (u[:-1, :, :] + u[1:, :, :])
        self.bc.set_face_boundary(0, Fx)
        self.bc.mask_solid_faces(0, Fx)
        Fy = np.zeros(mesh.face_shape(1))
        Fy[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        self.bc.set_face_boundary(1, Fy)
        self.bc.mask_solid_faces(1, Fy)
        Fz = None
        if not mesh.is_2d:
            Fz = np.zeros(mesh.face_shape(2))
            Fz[:, :, 1:-1] = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
            self.bc.set_face_boundary(2, Fz)
            self.bc.mask_solid_faces(2, Fz)
        d = div_op(Fx, Fy, Fz, mesh.dx, mesh.dy, mesh.dz, two_d=mesh.is_2d)
        return float(np.abs(d).max())