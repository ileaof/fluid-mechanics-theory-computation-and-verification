#!/usr/bin/env python3
"""
Example 7.1 -- Analytical: Hagen-Poiseuille flow in a circular pipe.

For steady, fully developed laminar flow of a Newtonian fluid in a pipe of radius
R, the Navier-Stokes equations reduce (Chapter 5) to

        (1/r) d/dr( r mu du/dr ) = dp/dx ,   u(R)=0 , du/dr(0)=0 ,

whose solution is the parabolic profile

        u(r) = (-dp/dx) (R^2 - r^2) / (4 mu) .

Integrating over the cross-section gives the Hagen-Poiseuille law and the standard
laminar results:

        Q       = pi R^4 (-dp/dx) / (8 mu)              (fourth-power law)
        V_mean  = Q/(pi R^2) = (-dp/dx) R^2/(8 mu) = u_max/2
        tau_w   = (-dp/dx) R / 2
        f * Re  = 64            (Darcy friction factor, Re based on diameter)

The program evaluates these closed forms and verifies the flow rate against direct
numerical integration of the profile, and confirms f*Re = 64.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

rho = 1000.0
mu  = 1.0e-3
R   = 0.01                     # pipe radius [m]
dpdx = -4.0                    # pressure gradient [Pa/m] -> laminar (Re~1000)

def u_profile(r):
    return (-dpdx) * (R**2 - r**2) / (4*mu)

u_max  = (-dpdx) * R**2 / (4*mu)
Q_exact = np.pi * R**4 * (-dpdx) / (8*mu)
V_mean  = Q_exact / (np.pi * R**2)
tau_w   = (-dpdx) * R / 2
Re      = rho * V_mean * (2*R) / mu
f       = (-dpdx) * (2*R) / (0.5 * rho * V_mean**2)

print("Example 7.1  Hagen-Poiseuille flow -- analytical verification\n")
print(f"  R = {R*1e3:.1f} mm, dp/dx = {dpdx} Pa/m, mu = {mu:.1e} Pa s")
print(f"  u_max = {u_max:.5f} m/s,  V_mean = {V_mean:.5f} m/s  (u_max/2 = {u_max/2:.5f})")
print(f"  Q = {Q_exact:.6e} m^3/s,  tau_w = {tau_w:.5f} Pa")
print(f"  Re = {Re:.2f},  f = {f:.5f},  f*Re = {f*Re:.4f}\n")

# verify Q by numerical integration of u(r) * 2 pi r
r = np.linspace(0, R, 200001)
Q_num = trapezoid(u_profile(r) * 2*np.pi*r, r)
err_Q = abs(Q_num - Q_exact)/Q_exact
print(f"  Q by integration = {Q_num:.6e} m^3/s,  rel. error vs Hagen-Poiseuille = {err_Q:.2e}")
# verify V_mean = u_max/2 and f*Re = 64
assert abs(V_mean - u_max/2) < 1e-12
assert abs(f*Re - 64.0) < 1e-9, "f*Re must equal 64 for laminar pipe flow"
assert err_Q < 1e-6, "Hagen-Poiseuille flow rate failed its integral check"
print("  PASS: Q verified, V_mean = u_max/2, and f*Re = 64.\n")

# demonstrate the fourth-power law Q ~ R^4
Rs = np.linspace(0.002, 0.02, 50)
Qs = np.pi * Rs**4 * (-dpdx) / (8*mu)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
rr = np.linspace(-R, R, 200)
ax1.plot(u_profile(np.abs(rr))/u_max, rr/R, "C0-", lw=2)
ax1.fill_betweenx(rr/R, 0, u_profile(np.abs(rr))/u_max, color="C0", alpha=0.15)
ax1.set_xlabel("$u/u_{max}$"); ax1.set_ylabel("$r/R$")
ax1.set_title("Parabolic velocity profile"); ax1.grid(alpha=0.3)
ax2.loglog(Rs*1e3, Qs, "C3-", lw=2, label=r"$Q=\pi R^4(-dp/dx)/(8\mu)$")
ax2.loglog(Rs*1e3, Qs[0]*(Rs/Rs[0])**4, "k:", lw=1.3, label="slope 4")
ax2.set_xlabel("pipe radius $R$ [mm]"); ax2.set_ylabel("flow rate $Q$ [m$^3$/s]")
ax2.set_title("Hagen-Poiseuille fourth-power law"); ax2.legend(frameon=False)
ax2.grid(alpha=0.3, which="both")
fig.suptitle("Hagen-Poiseuille flow in a circular pipe", y=1.04, fontsize=13)
fig.savefig("fig7_1_hagen.png", dpi=150, bbox_inches="tight")
print("  Wrote fig7_1_hagen.png")
