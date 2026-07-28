"""Force integration on an immersed obstacle (drag / lift coefficients).

The :class:`ForcesCalculator` integrates the pressure and viscous traction
that the fluid exerts on an immersed body over the **fluid/solid interface** of
the cell-centred solid mask (see :meth:`BoundaryCondition.solid`).  The body is
represented on the Cartesian mesh as a cell-resolution "staircase" (direct
forcing), so the integration is a discrete surface sum over the staircase
facets -- the fluid cell touching a solid cell contributes one facet.  The
estimate is first-order in the mesh spacing and converges under refinement.

Surface convention
------------------
Let ``n_body`` be the outward normal of the *body* (pointing from the solid
cell into the fluid cell).  The force the fluid exerts on the body is

.. math::
    \\mathbf{F} \\;=\\; \\int_S \\bigl(-p\\,\\mathbf{n}_\\mathrm{body}
        + \\boldsymbol{\\tau}\\cdot\\mathbf{n}_\\mathrm{body}\\bigr)\\,dA .

For a facet on axis ``a`` (an x-, y- or z-face) with cell spacing ``h`` and
face area ``A``:

* **pressure** contribution ``-p_fluid * n_body * A``;
* **viscous** contribution ``(tau . n_body) * A``, approximated with a one-sided
  wall gradient (the wall sits at the fluid/solid face, half a cell from the
  fluid centre, and ``u_wall = 0``):

  .. math::
     (\\boldsymbol{\\tau}\\cdot\\mathbf{n})_i \\;\\approx\\;
        \\mu\\,\\frac{u^{\\mathrm{fluid}}_i - 0}{h/2}
        \\;=\\; \\frac{2\\mu}{h}\\,u^{\\mathrm{fluid}}_i ,

  i.e. each velocity component contributes ``2*mu/h * u_fluid * A`` (the
  symmetric cross term ``mu * du_a/dx_i`` is dropped -- it is second-order small
  next to the staircase geometry error and keeps the estimate robust).

Coefficients (2-D, per unit span)::

    Cd = 2*Fx / (rho * U_inf**2 * D),   Cl = 2*Fy / (rho * U_inf**2 * D),
    Cp = p_fluid / (0.5*rho*U_inf**2),  Cf = tau_wall / (0.5*rho*U_inf**2).

The class also derives the recirculation length (centreline velocity sign
change behind the body) and the surface-angle distributions of Cp / Cf used to
locate the separation point.
"""

from __future__ import annotations

import numpy as np


class ForcesCalculator:
    """Drag / lift / pressure / skin-friction on an immersed obstacle.

    Parameters
    ----------
    mesh:
        The Cartesian mesh (provides coordinates, spacing, face areas).
    bc:
        The :class:`BoundaryCondition` carrying the solid mask (``bc.solid``).
    fluid:
        The :class:`Fluid` (provides ``rho`` and ``mu`` for the coefficients).
    U_inf:
        Free-stream velocity magnitude used to non-dimensionalise.
    D:
        Reference length (cylinder diameter) for the coefficients.
    """

    def __init__(self, mesh, bc, fluid, U_inf: float, D: float) -> None:
        self.mesh = mesh
        self.bc = bc
        self.fluid = fluid
        self.U_inf = float(U_inf)
        self.D = float(D)
        self.rho = float(fluid.rho)
        self.mu = float(fluid.mu)
        self.q_dyn = 0.5 * self.rho * self.U_inf ** 2 * self.D  # per-span dynamic scale
        # Body centroid (used as the angular reference for the surface plots).
        s = bc.solid
        if s is not None and s.any():
            Xc, Yc, _ = mesh.cell_grid()
            self.cx = float(Xc[s].mean())
            self.cy = float(Yc[s].mean())
        else:
            self.cx = 0.5 * mesh.Lx
            self.cy = 0.5 * mesh.Ly

    # ------------------------------------------------------------------ #
    def forces(self, u, v, p, mu: float | None = None) -> dict[str, float]:
        """Total force on the body and the drag / lift coefficients.

        Returns a dict with ``Fx, Fy, Fp_x, Fp_y, Fv_x, Fv_y, Cd, Cl``.
        """
        mesh = self.mesh
        s = self.bc.solid
        mu = self.mu if mu is None else float(mu)
        Fx = Fy = 0.0
        Fp_x = Fp_y = 0.0
        Fv_x = Fv_y = 0.0
        if s is None or not s.any():
            return {"Fx": 0.0, "Fy": 0.0, "Fp_x": 0.0, "Fp_y": 0.0,
                    "Fv_x": 0.0, "Fv_y": 0.0, "Cd": 0.0, "Cl": 0.0}

        # ---- axis 0 (x-faces): between cell (f-1) and cell (f) -------------
        # left_solid:  solid on (f-1), fluid on (f) -> n_body = +x, fluid cell f
        # right_solid: solid on (f),   fluid on (f-1) -> n_body = -x, fluid cell f-1
        L = s[:-1, :, :]            # cell (f-1), face index f = 1..Nx-1
        R = s[1:, :, :]             # cell (f)
        left_solid = L & ~R
        right_solid = ~L & R
        A = mesh.area_x              # area of an x-normal face (dy * dz)
        h = mesh.dx
        coef = 2.0 * mu / h
        # pressure: F_p = -p_fluid * n_body * A ; n_body_x = +1 (left) / -1 (right)
        Fp_x += float(-p[1:, :, :][left_solid].sum() * A)               # fluid cell f
        Fp_x += float(+p[:-1, :, :][right_solid].sum() * A)            # fluid cell f-1
        # viscous: F_v = (2 mu / h) * u_fluid * A  (per component)
        Fv_x += float(coef * u[1:, :, :][left_solid].sum() * A)
        Fv_x += float(coef * u[:-1, :, :][right_solid].sum() * A)
        Fv_y += float(coef * v[1:, :, :][left_solid].sum() * A)
        Fv_y += float(coef * v[:-1, :, :][right_solid].sum() * A)

        # ---- axis 1 (y-faces): between cell (j-1) and cell (j) ------------
        Dn = s[:, :-1, :]           # cell (j-1)
        Up = s[:, 1:, :]            # cell (j)
        low_solid = Dn & ~Up        # solid below, fluid above -> n_body = +y, fluid cell j
        high_solid = ~Dn & Up       # solid above, fluid below -> n_body = -y, fluid cell j-1
        A = mesh.area_y             # area of a y-normal face (dx * dz)
        h = mesh.dy
        coef = 2.0 * mu / h
        Fp_y += float(-p[:, 1:, :][low_solid].sum() * A)               # fluid cell j, n_body=+y
        Fp_y += float(+p[:, :-1, :][high_solid].sum() * A)            # fluid cell j-1, n_body=-y
        Fv_x += float(coef * u[:, 1:, :][low_solid].sum() * A)
        Fv_x += float(coef * u[:, :-1, :][high_solid].sum() * A)
        Fv_y += float(coef * v[:, 1:, :][low_solid].sum() * A)
        Fv_y += float(coef * v[:, :-1, :][high_solid].sum() * A)

        Fx = Fp_x + Fv_x
        Fy = Fp_y + Fv_y
        denom = self.q_dyn if self.q_dyn > 1e-30 else 1.0
        return {"Fx": Fx, "Fy": Fy, "Fp_x": Fp_x, "Fp_y": Fp_y,
                "Fv_x": Fv_x, "Fv_y": Fv_y,
                "Cd": Fx / denom, "Cl": Fy / denom}

    # ------------------------------------------------------------------ #
    def surface(self, u, v, p) -> dict[str, np.ndarray]:
        """Per-facet surface distributions for Cp / Cf vs angle.

        Returns arrays ``theta`` (geometric angle from +x, rad), ``Cp``,
        ``Cf`` (streamwise skin-friction, the x-component of the wall
        traction -- changes sign at separation), and the facet coordinates.
        Facets are the fluid/solid interface faces (the staircase surface).
        """
        mesh = self.mesh
        s = self.bc.solid
        if s is None or not s.any():
            return {"theta": np.array([]), "Cp": np.array([]),
                     "Cf": np.array([]), "x": np.array([]), "y": np.array([])}
        p_scale = 0.5 * self.rho * self.U_inf ** 2
        p_scale = p_scale if p_scale > 1e-30 else 1.0
        mu = self.mu
        xs, ys, theta, Cp, Cf = [], [], [], [], []

        # axis 0 (x-faces)
        L = s[:-1, :, :]
        R = s[1:, :, :]
        A = mesh.area_x
        h = mesh.dx
        coef = 2.0 * mu / h
        xf = mesh.xf                      # face-centre x for x-faces, length Nx+1
        yc = mesh.yc                      # cell-centre y, length Ny
        # left_solid: fluid cell (f), facet at x=xf[f], y=yc[j]; n_body=+x
        left_solid = L & ~R
        if left_solid.any():
            f_idx, j_idx, k_idx = np.nonzero(left_solid)   # f_idx in [0,Nx-2] -> face f=f_idx+1
            face_x = xf[f_idx + 1]
            face_y = yc[j_idx]
            pf = p[1:, :, :][left_solid]
            uf = u[1:, :, :][left_solid]
            vf = v[1:, :, :][left_solid]
            xs.append(face_x); ys.append(face_y)
            theta.append(np.arctan2(face_y - self.cy, face_x - self.cx))
            Cp.append(pf / p_scale)
            Cf.append(coef * uf / p_scale)   # streamwise (x) wall shear
        # right_solid: fluid cell (f-1), facet at x=xf[f], y=yc[j]; n_body=-x
        right_solid = ~L & R
        if right_solid.any():
            f_idx, j_idx, k_idx = np.nonzero(right_solid)
            face_x = xf[f_idx + 1]
            face_y = yc[j_idx]
            pf = p[:-1, :, :][right_solid]
            uf = u[:-1, :, :][right_solid]
            xs.append(face_x); ys.append(face_y)
            theta.append(np.arctan2(face_y - self.cy, face_x - self.cx))
            Cp.append(pf / p_scale)
            Cf.append(coef * uf / p_scale)

        # axis 1 (y-faces)
        Dn = s[:, :-1, :]
        Up = s[:, 1:, :]
        A = mesh.area_y
        h = mesh.dy
        coef = 2.0 * mu / h
        xc = mesh.xc
        yf = mesh.yf
        # low_solid: solid below, fluid above (cell j); facet at x=xc[i], y=yf[j]
        low_solid = Dn & ~Up
        if low_solid.any():
            i_idx, j_idx, k_idx = np.nonzero(low_solid)   # j_idx in [0,Ny-2] -> face j_idx+1
            face_x = xc[i_idx]
            face_y = yf[j_idx + 1]
            pf = p[:, 1:, :][low_solid]
            uf = u[:, 1:, :][low_solid]
            xs.append(face_x); ys.append(face_y)
            theta.append(np.arctan2(face_y - self.cy, face_x - self.cx))
            Cp.append(pf / p_scale)
            Cf.append(coef * uf / p_scale)
        # high_solid: solid above, fluid below (cell j-1); facet at x=xc[i], y=yf[j]
        high_solid = ~Dn & Up
        if high_solid.any():
            i_idx, j_idx, k_idx = np.nonzero(high_solid)
            face_x = xc[i_idx]
            face_y = yf[j_idx + 1]
            pf = p[:, :-1, :][high_solid]
            uf = u[:, :-1, :][high_solid]
            xs.append(face_x); ys.append(face_y)
            theta.append(np.arctan2(face_y - self.cy, face_x - self.cx))
            Cp.append(pf / p_scale)
            Cf.append(coef * uf / p_scale)

        theta = np.concatenate(theta)
        order = np.argsort(theta)
        return {
            "theta": theta[order],
            "Cp": np.concatenate(Cp)[order],
            "Cf": np.concatenate(Cf)[order],
            "x": np.concatenate(xs)[order],
            "y": np.concatenate(ys)[order],
        }

    # ------------------------------------------------------------------ #
    def recirculation_length(self, u) -> float:
        """Length of the near-wake recirculation bubble behind the body.

        Along the centreline row (y closest to the body centroid) and
        downstream of the body, find the x where the streamwise velocity
        crosses from negative (reversed) back to non-negative.  Returns the
        length in metres measured from the rear of the body (``x_rear =
        cx + D/2``), or 0.0 if no reversed-flow region is found.
        """
        mesh = self.mesh
        s = self.bc.solid
        if s is None or not s.any():
            return 0.0
        j_c = int(np.argmin(np.abs(mesh.yc - self.cy)))
        uc = u[:, j_c, 0]
        x = mesh.xc
        x_rear = self.cx + 0.5 * self.D
        # scan from just behind the body downstream
        started = False
        for i in range(mesh.Nx):
            if x[i] <= x_rear:
                continue
            if uc[i] < 0.0:
                started = True
            elif started and uc[i] >= 0.0:
                # linear-interpolate the zero crossing between i-1 and i
                x0, x1 = x[i - 1], x[i]
                u0, u1 = uc[i - 1], uc[i]
                if u1 - u0 != 0.0:
                    xc = x0 - u0 * (x1 - x0) / (u1 - u0)
                else:
                    xc = x1
                return float(max(xc - x_rear, 0.0))
        return 0.0

    # ------------------------------------------------------------------ #
    def separation_angle_deg(self, u, v) -> float:
        """Approximate separation angle (degrees from the front stagnation).

        Uses the surface distribution of the streamwise wall shear (Cf) and
        reports the angle, measured from the forward stagnation point, where Cf
        changes sign on the upper half of the body.  Returns ``nan`` if no
        sign change is found.  Staircase noise makes this approximate.
        """
        surf = self.surface(u, v, np.zeros_like(u))
        if surf["theta"].size == 0:
            return float("nan")
        theta = surf["theta"]
        Cf = surf["Cf"]
        # upper half: 0 < theta < pi  (y > cy); angle from front stagnation =
        # pi - theta (front stagnation is at theta = pi, i.e. -x).
        upper = (theta > 0.0) & (theta < np.pi)
        th_u = theta[upper]
        cf_u = Cf[upper]
        order = np.argsort(th_u)
        th_u = th_u[order]
        cf_u = cf_u[order]
        # sign change + -> - scanning from the front (pi) backward to the top (pi/2)
        # i.e. decreasing theta from pi; reverse so we scan front->top.
        for k in range(len(th_u) - 1):
            if cf_u[k] > 0.0 and cf_u[k + 1] <= 0.0:
                ang_from_front = np.degrees(np.pi - th_u[k + 1])
                return float(ang_from_front)
        return float("nan")