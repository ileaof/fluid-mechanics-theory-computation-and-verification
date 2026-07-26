#!/usr/bin/env python3
"""
Example 1.1 -- Analytical solution: plane Couette-Poiseuille flow.

Steady, fully developed, incompressible flow of a Newtonian fluid between two
infinite parallel plates a distance H apart.  The lower plate (y = 0) is fixed;
the upper plate (y = H) moves in its own plane at speed U.  A constant pressure
gradient dp/dx may also be imposed.  The x-momentum equation reduces to

        d/dy ( mu du/dy ) = dp/dx ,      u(0)=0 ,  u(H)=U ,

whose closed-form solution is

        u(y) = U (y/H) + (1/(2 mu)) (dp/dx) (y^2 - H y).

With Y = y/H and the dimensionless pressure gradient
        P = -(H^2/(2 mu U)) dp/dx
the profile becomes  u/U = Y + P Y (1 - Y).

This program evaluates the exact profile for several P, verifies the closed-form
wall shear stress against a finite-difference estimate, verifies the closed-form
volume flow rate against numerical quadrature, and saves a figure.  Only numpy
and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Physical data (SI units)
H  = 2.0e-3        # plate gap                       [m]
U  = 0.50          # upper-plate speed               [m/s]
mu = 1.0e-3        # dynamic viscosity (water, 20 C) [Pa s]

def u_profile(y, dpdx):
    return U * (y / H) + (1.0 / (2.0 * mu)) * dpdx * (y**2 - H * y)

def tau_wall_exact(dpdx):
    # du/dy at y=0 = U/H - (H/(2 mu)) dpdx
    return mu * (U / H) - (H / 2.0) * dpdx

def flow_rate_exact(dpdx):
    return U * H / 2.0 - dpdx * H**3 / (12.0 * mu)

P_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
def dpdx_from_P(P):
    return -P * 2.0 * mu * U / H**2

print("Example 1.1  Plane Couette-Poiseuille flow -- analytical verification")
print(f"  H = {H*1e3:.3f} mm,  U = {U:.3f} m/s,  mu = {mu:.3e} Pa s\n")
print(f"  {'P':>5} {'tau_w exact':>14} {'tau_w FD':>14} {'Q exact':>13}"
      f" {'Q quad':>13} {'max|err|':>11}")

y_fine = np.linspace(0.0, H, 20001)
worst = 0.0
for P in P_values:
    dpdx = dpdx_from_P(P)
    u = u_profile(y_fine, dpdx)
    dy = y_fine[1] - y_fine[0]
    dudy0 = (-3*u[0] + 4*u[1] - u[2]) / (2*dy)
    tau_fd = mu * dudy0
    tau_ex = tau_wall_exact(dpdx)
    Q_quad = trapezoid(u, y_fine)
    Q_ex   = flow_rate_exact(dpdx)
    # mixed absolute/relative error (guards the P=-1 case where tau_w ~ 0)
    def relerr(a, b, scale):
        return abs(a - b) / (abs(b) + scale)
    err = max(relerr(tau_fd, tau_ex, mu * U / H),
              relerr(Q_quad, Q_ex, U * H))
    worst = max(worst, err)
    print(f"  {P:5.1f} {tau_ex:14.6e} {tau_fd:14.6e} {Q_ex:13.6e}"
          f" {Q_quad:13.6e} {err:11.2e}")

print(f"\n  Worst relative discrepancy (closed form vs numerics): {worst:.2e}")
assert worst < 1e-4, "analytical formulas failed their numerical check"
print("  PASS: closed-form tau_w and Q agree with independent numerics.\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
Y = np.linspace(0.0, 1.0, 400)
for P in P_values:
    ax1.plot(Y + P * Y * (1 - Y), Y, lw=2, label=f"P = {P:+.0f}")
ax1.axvline(0.0, color="0.6", lw=0.8)
ax1.set_xlabel(r"$u/U$"); ax1.set_ylabel(r"$y/H$")
ax1.set_title("Velocity profiles")
ax1.legend(frameon=False, fontsize=9); ax1.grid(alpha=0.3)
for P in P_values:
    dpdx = dpdx_from_P(P)
    dudy = U / H + (1.0 / (2.0 * mu)) * dpdx * (2 * Y * H - H)
    ax2.plot(mu * dudy, Y, lw=2, label=f"P = {P:+.0f}")
ax2.axvline(0.0, color="0.6", lw=0.8)
ax2.set_xlabel(r"$\tau = \mu\,du/dy$  [Pa]"); ax2.set_ylabel(r"$y/H$")
ax2.set_title("Shear-stress distribution"); ax2.grid(alpha=0.3)
fig.suptitle("Plane Couette-Poiseuille flow of a Newtonian fluid", y=1.05, fontsize=13)
fig.savefig("fig1_1_couette.png", dpi=150, bbox_inches="tight")
print("  Wrote fig1_1_couette.png")
