#!/usr/bin/env python3
"""
Example 2.1 -- Analytical: hydrostatic force on an inclined plane gate.

A flat rectangular gate of width b and height (along the plane) Lg is submerged
in a static liquid of density rho.  The gate is inclined at angle theta to the
horizontal free surface.  The gauge pressure at depth z is p = rho g z, so the
pressure grows linearly down the plane.  The resultant force and its point of
application (the centre of pressure) follow in closed form:

    F     = rho g h_c A              (A = b Lg,  h_c = centroid depth)
    y_cp  = y_c + I_xc / (y_c A)     (y measured along the plane from the surface)
    h_cp  = y_cp sin(theta)

with I_xc = b Lg^3 / 12 the second moment of the area about its centroidal axis.
This program evaluates the closed forms for several inclinations and verifies each
against a direct numerical integration of the pressure distribution over the gate.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

rho = 1000.0        # water density        [kg/m^3]
g   = 9.81          # gravity              [m/s^2]
b   = 2.0           # gate width           [m]
Lg  = 3.0           # gate length (plane)  [m]
d_top = 1.0         # depth of the top edge of the gate [m]

def analytic(theta):
    """Closed-form resultant force, centre of pressure, both along-plane & depth."""
    A   = b * Lg
    # along-plane distance from the free-surface line to the top edge
    y_top = d_top / np.sin(theta)
    y_c   = y_top + Lg / 2.0                 # centroid, along plane
    h_c   = y_c * np.sin(theta)              # centroid depth
    F     = rho * g * h_c * A
    Ixc   = b * Lg**3 / 12.0
    y_cp  = y_c + Ixc / (y_c * A)
    h_cp  = y_cp * np.sin(theta)
    return F, y_c, y_cp, h_cp

def numeric(theta, n=200000):
    """Direct integration of p = rho g z over the gate area."""
    y_top = d_top / np.sin(theta)
    y = np.linspace(y_top, y_top + Lg, n)    # along-plane coordinate
    z = y * np.sin(theta)                     # depth of each strip
    p = rho * g * z                           # gauge pressure
    dF = p * b                                # force per unit along-plane length
    F  = trapezoid(dF, y)
    y_cp = trapezoid(y * dF, y) / F           # first moment / force
    return F, y_cp, y_cp * np.sin(theta)

print("Example 2.1  Hydrostatic force on an inclined plane gate -- verification")
print(f"  rho={rho:.0f} kg/m^3, gate {b:.1f} m x {Lg:.1f} m, top edge at {d_top:.1f} m depth\n")
print(f"  {'theta[deg]':>10} {'F_exact[kN]':>13} {'F_num[kN]':>12}"
      f" {'h_cp_exact[m]':>14} {'h_cp_num[m]':>13} {'max relerr':>12}")

worst = 0.0
for deg in (30.0, 45.0, 60.0, 90.0):
    th = np.radians(deg)
    F_a, y_c, y_cp_a, h_cp_a = analytic(th)
    F_n, y_cp_n, h_cp_n = numeric(th)
    e = max(abs(F_a - F_n) / F_a, abs(h_cp_a - h_cp_n) / h_cp_a)
    worst = max(worst, e)
    print(f"  {deg:10.1f} {F_a/1e3:13.4f} {F_n/1e3:12.4f}"
          f" {h_cp_a:14.5f} {h_cp_n:13.5f} {e:12.2e}")

print(f"\n  Worst relative discrepancy (closed form vs integration): {worst:.2e}")
assert worst < 1e-5, "analytical hydrostatic-force formulas failed their check"
# The centre of pressure always lies below the centroid:
for deg in (30.0, 45.0, 60.0, 90.0):
    th = np.radians(deg)
    _, y_c, y_cp_a, _ = analytic(th)
    assert y_cp_a > y_c, "centre of pressure must be below the centroid"
print("  PASS: F and h_cp verified; centre of pressure lies below the centroid.\n")

# ---- figure: vertical gate, pressure prism, resultant at the centre of pressure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)

th = np.radians(90.0)
z_top, z_bot = d_top, d_top + Lg
zz = np.linspace(0, z_bot + 0.5, 200)
ax1.plot(rho*g*zz/1e3, zz, "C0-", lw=2)
ax1.fill_betweenx([z_top, z_bot], 0,
                  [rho*g*z_top/1e3, rho*g*z_bot/1e3], color="C0", alpha=0.25)
F_a, y_c, y_cp_a, h_cp_a = analytic(th)
ax1.axhline(h_cp_a, color="C3", ls="--", lw=1.5)
ax1.plot(rho*g*h_cp_a/1e3, h_cp_a, "C3o", ms=8)
ax1.text(rho*g*h_cp_a/1e3*0.5, h_cp_a+0.15, "centre of pressure", color="C3")
ax1.text(rho*g*(z_c:=(z_top+z_bot)/2)/1e3*0.5, z_c-0.2, "centroid", color="0.3")
ax1.invert_yaxis()
ax1.set_xlabel("gauge pressure  $p=\\rho g z$  [kPa]")
ax1.set_ylabel("depth $z$ [m]")
ax1.set_title("Pressure distribution on a vertical gate")
ax1.grid(alpha=0.3)

degs = np.linspace(20, 90, 120)
Fs = [analytic(np.radians(d))[0]/1e3 for d in degs]
hcps = [analytic(np.radians(d))[3] for d in degs]
ax2.plot(degs, Fs, "C0-", lw=2, label="resultant force $F$ [kN]")
ax2b = ax2.twinx()
ax2b.plot(degs, hcps, "C3--", lw=2, label="$h_{cp}$ [m]")
ax2.set_xlabel("inclination $\\theta$ [deg]")
ax2.set_ylabel("$F$ [kN]", color="C0"); ax2.tick_params(axis="y", colors="C0")
ax2b.set_ylabel("$h_{cp}$ [m]", color="C3"); ax2b.tick_params(axis="y", colors="C3")
ax2.set_title("Force and centre of pressure vs inclination")
ax2.grid(alpha=0.3)

fig.suptitle("Hydrostatic force on a submerged plane gate", y=1.05, fontsize=13)
fig.savefig("fig2_1_gate.png", dpi=150, bbox_inches="tight")
print("  Wrote fig2_1_gate.png")
