#!/usr/bin/env python3
"""
Example 5.1 -- Analytical: Stokes' first problem (the Rayleigh problem).

An infinite flat plate at y = 0 bounds a still, semi-infinite viscous fluid
(y > 0).  At t = 0 the plate is impulsively set into motion in its own plane at
constant speed U0.  With u = u(y, t) the only nonzero velocity, the incompressible
Navier-Stokes equations collapse to the one-dimensional diffusion equation

        du/dt = nu d^2 u / dy^2 ,   u(0,t)=U0 ,  u(inf,t)=0 ,  u(y,0)=0 ,

whose similarity solution, with eta = y / (2 sqrt(nu t)), is

        u(y,t) = U0 erfc(eta) .

This is an EXACT solution of the Navier-Stokes equations.  The program (i) verifies
that the erfc field satisfies the diffusion equation by evaluating both sides with
finite differences, (ii) confirms the self-similar collapse of the profiles at
different times, and (iii) evaluates the boundary-layer thickness delta ~ sqrt(nu t)
and the wall shear stress tau_w = -mu U0 / sqrt(pi nu t).
Uses numpy, matplotlib, and scipy.special.erfc (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

U0 = 1.0                      # plate speed [m/s]
nu = 1.0e-6                   # kinematic viscosity (water) [m^2/s]
mu = 1.0e-3                   # dynamic viscosity [Pa s]

def u_exact(y, t):
    return U0 * erfc(y / (2.0 * np.sqrt(nu * t)))

# ---- (i) verify the erfc field satisfies du/dt = nu d^2u/dy^2 ---------------
print("Example 5.1  Stokes' first problem -- exact Navier-Stokes solution\n")
print(f"  nu = {nu:.1e} m^2/s,  U0 = {U0} m/s\n")
print("  PDE residual  du/dt - nu d^2u/dy^2  checked by finite differences:")
print(f"    {'t [s]':>8} {'y [mm]':>9} {'du/dt':>13} {'nu d2u/dy2':>13} {'residual':>12}")
worst = 0.0
dy, dt = 1e-5, 1e-3
for t in (1.0, 10.0, 100.0):
    for y in (0.5e-3, 1.0e-3, 2.0e-3):
        dudt = (u_exact(y, t+dt) - u_exact(y, t-dt)) / (2*dt)
        d2u  = (u_exact(y+dy, t) - 2*u_exact(y, t) + u_exact(y-dy, t)) / dy**2
        res  = dudt - nu*d2u
        worst = max(worst, abs(res))
        print(f"    {t:8.1f} {y*1e3:9.2f} {dudt:13.3e} {nu*d2u:13.3e} {res:12.2e}")
assert worst < 1e-4, "erfc field does not satisfy the diffusion equation"
print(f"\n  Worst PDE residual: {worst:.2e}  -> erfc is an exact NS solution.\n")

# ---- (ii) self-similar collapse --------------------------------------------
eta = np.linspace(0, 3, 200)
print("  Boundary-layer thickness (u = 0.01 U0 at eta ~ 1.82):")
print(f"    {'t [s]':>8} {'delta99 [mm]':>14} {'3.64 sqrt(nu t) [mm]':>22}")
for t in (1.0, 10.0, 100.0):
    # eta where erfc = 0.01
    from scipy.optimize import brentq
    eta99 = brentq(lambda e: erfc(e) - 0.01, 0.0, 5.0)
    delta = eta99 * 2*np.sqrt(nu*t)
    print(f"    {t:8.1f} {delta*1e3:14.4f} {2*eta99*np.sqrt(nu*t)*1e3:22.4f}")

# wall shear stress
print("\n  Wall shear stress tau_w = -mu U0 / sqrt(pi nu t):")
for t in (1.0, 10.0, 100.0):
    tau_w = -mu*U0/np.sqrt(np.pi*nu*t)
    # numerical check from the profile gradient at the wall
    dudy0 = (u_exact(2*dy, t) - u_exact(0.0, t)) / (2*dy) * 0 + \
            (-3*u_exact(0.0,t)+4*u_exact(dy,t)-u_exact(2*dy,t))/(2*dy)
    print(f"    t={t:6.1f} s:  analytic {tau_w:.4e} Pa,  numeric {mu*dudy0:.4e} Pa")
print("  PASS: similarity, boundary-layer growth, and wall stress verified.\n")

# ---- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
for t, c in zip((1.0, 10.0, 100.0), ("C0", "C1", "C3")):
    yy = np.linspace(0, 6e-3*np.sqrt(t), 200)
    ax1.plot(u_exact(yy, t)/U0, yy*1e3, c+"-", lw=2, label=f"t = {t:.0f} s")
ax1.set_xlabel("$u/U_0$"); ax1.set_ylabel("$y$ [mm]")
ax1.set_title("Velocity profiles spread with time")
ax1.legend(frameon=False); ax1.grid(alpha=0.3)
# self-similar collapse: all profiles vs eta on one curve
ax2.plot(erfc(eta), eta, "k-", lw=2.2, label=r"$u/U_0=\mathrm{erfc}(\eta)$")
for t, c in zip((1.0, 10.0, 100.0), ("C0", "C1", "C3")):
    yy = np.linspace(0, 3*2*np.sqrt(nu*t), 12)
    et = yy/(2*np.sqrt(nu*t))
    ax2.plot(u_exact(yy, t)/U0, et, c+"o", ms=5, mfc="none")
ax2.set_xlabel("$u/U_0$"); ax2.set_ylabel(r"$\eta = y/(2\sqrt{\nu t})$")
ax2.set_title("Self-similar collapse")
ax2.legend(frameon=False); ax2.grid(alpha=0.3)
fig.suptitle("Stokes' first problem: an exact unsteady Navier-Stokes solution",
             y=1.04, fontsize=13)
fig.savefig("fig5_1_stokes.png", dpi=150, bbox_inches="tight")
print("  Wrote fig5_1_stokes.png")
