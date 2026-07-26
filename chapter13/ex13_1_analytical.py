#!/usr/bin/env python3
"""
Example 13.1 -- Natural convection in a differentially heated vertical channel.

Two vertical plates a distance 2b apart are held at different temperatures, the left
(y=-b) hot at T1 and the right (y=+b) cold at T2.  The air between them rises on the
hot side and sinks on the cold side, driven by BUOYANCY through the Boussinesq
approximation.  For fully developed flow the energy equation gives a linear
temperature profile, and the momentum equation, balancing viscous forces against the
buoyancy of the temperature excess,

        nu d^2u/dy^2 + g beta (T - T_m) = 0 ,   T(y) = T_m + (dT/2)(y/b) ,

integrates to the exact cubic velocity profile

        u(y) = (g beta dT b^2)/(12 nu) [ (y/b) - (y/b)^3 ] ,

an antisymmetric flow with zero net discharge -- hot fluid up, cold fluid down.  The
program evaluates this closed form, verifies it against the governing equation, the
no-slip boundary conditions, and the zero-net-flow condition, and reports the Grashof
number.  Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

g, beta, nu = 9.81, 3.4e-3, 1.5e-5      # air ~300 K
b, dT = 0.02, 20.0                        # half-gap 20 mm, temperature difference 20 K

def u_exact(y):
    return (g*beta*dT*b**2)/(12*nu) * ((y/b) - (y/b)**3)

def T_profile(y):
    return 0.5*dT*(y/b)                    # temperature excess above the mean

print("Example 13.1  Natural convection in a vertical channel\n")
Gr = g*beta*dT*(2*b)**3/nu**2
print(f"  Gap 2b = {2*b*1e3:.0f} mm, dT = {dT} K, Grashof number Gr = {Gr:.3e}\n")

# verify governing equation nu u'' + g beta (T - T_m) = 0
y = np.linspace(-b, b, 40001); dy = y[1]-y[0]
u = u_exact(y)
upp = np.gradient(np.gradient(u, dy), dy)
residual = nu*upp + g*beta*T_profile(y)
interior = slice(50, -50)
print("  Verification of the exact solution:")
print(f"    max |nu u'' + g beta (T-T_m)| = {np.max(np.abs(residual[interior])):.2e}  (governing eq.)")
print(f"    u(-b) = {u_exact(-b):.2e},  u(+b) = {u_exact(b):.2e}  (no-slip)")
Q = trapezoid(u, y)
print(f"    net discharge  integral u dy = {Q:.2e}  (should be 0)")
u_max = u_exact(b/np.sqrt(3))
print(f"    peak velocity at y = b/sqrt(3): u_max = {u_max:.4f} m/s")
assert np.max(np.abs(residual[interior])) < 1e-2
assert abs(u_exact(b)) < 1e-12 and abs(u_exact(-b)) < 1e-12
assert abs(Q) < 1e-8
print("  PASS: cubic profile satisfies the equation, no-slip, and zero net flow.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.0,4.5),constrained_layout=True)
yy=np.linspace(-b,b,200)
ax1.plot(u_exact(yy)*1e3, yy/b, "C0-", lw=2)
ax1.axvline(0,color="0.6",lw=0.8); ax1.fill_betweenx(yy/b,0,u_exact(yy)*1e3,color="C0",alpha=0.15)
ax1.set_xlabel("velocity $u$ [mm/s]"); ax1.set_ylabel("$y/b$")
ax1.set_title("Buoyant velocity profile (cubic)"); ax1.grid(alpha=0.3)
ax1.text(u_max*1e3*0.4,0.6,"hot side\n(rising)",fontsize=8,color="C3")
ax1.text(-u_max*1e3*0.7,-0.6,"cold side\n(sinking)",fontsize=8,color="C0")
ax2.plot(T_profile(yy), yy/b, "C3-", lw=2)
ax2.set_xlabel(r"temperature excess $T-T_m$ [K]"); ax2.set_ylabel("$y/b$")
ax2.set_title("Linear temperature profile"); ax2.grid(alpha=0.3)
fig.suptitle("Natural convection between vertical plates", y=1.04, fontsize=13)
fig.savefig("fig13_1_natconv.png", dpi=150, bbox_inches="tight")
print("  Wrote fig13_1_natconv.png")
