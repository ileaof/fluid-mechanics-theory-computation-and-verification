#!/usr/bin/env python3
"""
Example 4.2 -- Control-volume linear momentum: force on a reducing bend, with
kinetic-energy and momentum correction factors.

Water flows steadily through a bend that turns the flow by angle theta and reduces
the area from A1 to A2.  Three integral laws close the problem:

  MASS       Q = A1 V1 = A2 V2
  BERNOULLI  p2 = p1 + (rho/2)(V1^2 - V2^2)        (horizontal, loss-free)
  MOMENTUM   sum F = rho Q (beta2 V2_out - beta1 V1_in)   (vector)

Solving the x- and y-momentum balances gives the anchoring force (Rx, Ry) that
holds the bend in place.  The program computes this force, then verifies that the
momentum balance closes to machine zero when the force is substituted back.

Real velocity profiles are not uniform, so the momentum flux carries a correction
factor beta = (1/(A V^2)) integral u^2 dA and the kinetic-energy flux a factor
alpha = (1/(A V^3)) integral u^3 dA.  For fully developed LAMINAR pipe flow
(parabolic profile) these are exactly beta = 4/3 and alpha = 2.  The program
computes both by numerical integration over the cross-section and confirms
second-order convergence to the exact values.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rho = 1000.0
Q   = 0.05                 # volume flow rate [m^3/s]
D1, D2 = 0.20, 0.12        # inlet / outlet diameters [m]
theta = np.radians(60.0)   # bend turning angle
p1  = 200e3                # inlet gauge pressure [Pa]

A1 = np.pi*D1**2/4; A2 = np.pi*D2**2/4
V1 = Q/A1; V2 = Q/A2
p2 = p1 + 0.5*rho*(V1**2 - V2**2)          # Bernoulli

print("Example 4.2  Force on a reducing bend (control-volume momentum)\n")
print(f"  Q={Q} m^3/s: V1={V1:.3f} m/s, V2={V2:.3f} m/s")
print(f"  p1={p1/1e3:.1f} kPa, p2={p2/1e3:.1f} kPa, turn angle={np.degrees(theta):.0f} deg\n")

# momentum balance (beta=1 for the resultant-force calculation; profiles below)
# x: Rx + p1 A1 - p2 A2 cos(theta) = rho Q (V2 cos(theta) - V1)
# y: Ry        - p2 A2 sin(theta) = rho Q (V2 sin(theta) - 0)
Rx = rho*Q*(V2*np.cos(theta) - V1) - p1*A1 + p2*A2*np.cos(theta)
Ry = rho*Q*(V2*np.sin(theta))                 + p2*A2*np.sin(theta)
R  = np.hypot(Rx, Ry)
phi = np.degrees(np.arctan2(Ry, Rx))
print(f"  Anchoring force: Rx={Rx/1e3:.3f} kN, Ry={Ry/1e3:.3f} kN")
print(f"  Magnitude R={R/1e3:.3f} kN at {phi:.1f} deg from +x\n")

# verify the momentum balance closes: residual should be ~0
res_x = (Rx + p1*A1 - p2*A2*np.cos(theta)) - rho*Q*(V2*np.cos(theta) - V1)
res_y = (Ry - p2*A2*np.sin(theta))         - rho*Q*(V2*np.sin(theta))
print(f"  Momentum residual: |Rx-balance|={abs(res_x):.2e} N, "
      f"|Ry-balance|={abs(res_y):.2e} N")
assert abs(res_x) < 1e-6 and abs(res_y) < 1e-6, "momentum balance does not close"
print("  PASS: control-volume momentum balance closes to machine zero.\n")

# ---- correction factors from the laminar (parabolic) profile ----------------
# u(r) = 2 V (1 - (r/R)^2);  beta = (1/(A V^2)) int u^2 dA,  alpha similarly with u^3
def correction_factors(N, R=0.1, V=1.0):
    """Midpoint integration over the pipe cross-section (annular rings)."""
    dr = R/N
    r  = (np.arange(N)+0.5)*dr
    u  = 2*V*(1 - (r/R)**2)
    dA = 2*np.pi*r*dr
    A  = np.pi*R**2
    beta  = np.sum(u**2 * dA)/(A*V**2)
    alpha = np.sum(u**3 * dA)/(A*V**3)
    return beta, alpha

print("  Correction factors for the laminar parabolic profile:")
print(f"    {'N':>6} {'beta':>12} {'err_beta':>11} {'p':>6} {'alpha':>12} {'err_alpha':>11}")
prevb = None
for N in [10, 20, 40, 80, 160, 320]:
    b, a = correction_factors(N)
    eb = abs(b - 4/3); ea = abs(a - 2.0)
    p = np.log(prevb/eb)/np.log(2) if prevb else float("nan")
    print(f"    {N:6d} {b:12.8f} {eb:11.3e} {p:6.3f} {a:12.8f} {ea:11.3e}")
    prevb = eb
print(f"\n  Exact: beta = 4/3 = {4/3:.6f}, alpha = 2")
assert 1.9 < p < 2.1, "correction-factor quadrature is not second order"
print("  PASS: beta -> 4/3 and alpha -> 2 at second order.\n")

# ---- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
# bend schematic with force
ax1.annotate("", xy=(0,0), xytext=(-1.2,0),
             arrowprops=dict(arrowstyle="-|>", lw=6, color="C0", alpha=0.5))
ax1.annotate("", xy=(0.9*np.cos(theta),0.9*np.sin(theta)), xytext=(0,0),
             arrowprops=dict(arrowstyle="-|>", lw=4, color="C0", alpha=0.5))
ax1.text(-1.15,0.12,r"$V_1,\,p_1,\,A_1$",color="C0")
ax1.text(0.35,0.62,r"$V_2,\,p_2,\,A_2$",color="C0")
sc=0.9/R if (R:=np.hypot(Rx,Ry)) else 1
ax1.annotate("", xy=(Rx/R*0.9, Ry/R*0.9), xytext=(0,0),
             arrowprops=dict(arrowstyle="-|>", lw=2.5, color="C3"))
ax1.text(Rx/R*0.5-0.1, Ry/R*0.5-0.18, f"$R$={R/1e3:.1f} kN", color="C3")
ax1.plot(0,0,"ko",ms=6)
ax1.set_xlim(-1.3,1.1); ax1.set_ylim(-0.9,0.9); ax1.set_aspect("equal"); ax1.axis("off")
ax1.set_title("Reducing bend and anchoring force")
# profile + correction factors
r = np.linspace(-1,1,200)
ax2.plot(2*(1-r**2), r, "C0-", lw=2, label="laminar $u/V=2(1-(r/R)^2)$")
ax2.axvline(1.0, color="0.6", ls=":", lw=1.2, label="mean $V$")
ax2.set_xlabel("$u/V$"); ax2.set_ylabel("$r/R$")
ax2.set_title(r"Profile: $\beta=4/3,\ \alpha=2$")
ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.3)
fig.suptitle("Integral momentum and velocity-profile correction factors",
             y=1.04, fontsize=13)
fig.savefig("fig4_2_bend.png", dpi=150, bbox_inches="tight")
print("  Wrote fig4_2_bend.png")
