#!/usr/bin/env python3
"""
Example 9.1 -- Isentropic flow and the converging-diverging nozzle.

For steady, adiabatic, reversible (isentropic) flow of a perfect gas, the local
flow properties are tied to the Mach number M through the stagnation relations

        T0/T   = 1 + (g-1)/2 M^2
        p0/p   = (T0/T)^( g/(g-1) )
        rho0/rho = (T0/T)^( 1/(g-1) )

and the cross-sectional area is tied to M by the area-Mach relation

        A/A* = (1/M) [ (2/(g+1))(1 + (g-1)/2 M^2) ] ^ ((g+1)/(2(g-1))) ,

where A* is the sonic throat area.  For a given area ratio A/A* > 1 this equation
has TWO solutions -- one subsonic, one supersonic -- corresponding to the converging
and diverging operation of a nozzle.  The program solves the area-Mach relation on
each branch with a bracketed root finder, verifies the round trip
M -> A/A* -> M to machine precision, and confirms the stagnation relations.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

g = 1.4

def T0_T(M):   return 1 + 0.5*(g-1)*M**2
def p0_p(M):   return T0_T(M)**(g/(g-1))
def rho0_rho(M): return T0_T(M)**(1/(g-1))
def area_ratio(M):
    return (1/M)*((2/(g+1))*(1+0.5*(g-1)*M**2))**((g+1)/(2*(g-1)))

def mach_from_area(AR, branch):
    """Invert A/A* = area_ratio(M) on the 'sub' or 'super' branch."""
    if branch == "sub":
        return brentq(lambda M: area_ratio(M)-AR, 1e-6, 1.0, xtol=1e-13)
    else:
        return brentq(lambda M: area_ratio(M)-AR, 1.0, 50.0, xtol=1e-13)

print("Example 9.1  Isentropic flow and the C-D nozzle (gamma = 1.4)\n")
# round-trip verification across a range of Mach numbers
print("  Round-trip verification  M -> A/A* -> M:")
print(f"    {'M':>6} {'A/A*':>10} {'branch':>7} {'M recovered':>13} {'|err|':>10}")
worst = 0.0
for M in (0.2, 0.5, 0.8, 1.5, 2.5, 4.0):
    AR = area_ratio(M)
    br = "sub" if M < 1 else "super"
    Mr = mach_from_area(AR, br)
    e = abs(Mr - M)
    worst = max(worst, e)
    print(f"    {M:6.2f} {AR:10.4f} {br:>7} {Mr:13.8f} {e:10.2e}")
assert worst < 1e-8, "area-Mach inversion failed"

# verify the stagnation relations are mutually consistent (p0/p = (rho0/rho)^g)
print("\n  Consistency of stagnation relations  p0/p = (rho0/rho)^gamma:")
for M in (0.5, 1.0, 2.0, 3.0):
    lhs, rhs = p0_p(M), rho0_rho(M)**g
    print(f"    M={M:.1f}:  p0/p={lhs:10.4f},  (rho0/rho)^g={rhs:10.4f},  diff={abs(lhs-rhs):.2e}")
    assert abs(lhs-rhs) < 1e-9
print("\n  PASS: area-Mach inversion and isentropic relations verified.\n")

# --- solve a full converging-diverging nozzle (supersonic design) -----------
# area distribution A(x)/A* : converge to throat at x=0.5, diverge to Ae/A*=4
xs = np.linspace(0, 1, 400)
At = 1.0
Ae_As = 4.0
# smooth area: min at throat
Ax = 1.0 + 3.0*( (xs-0.5)/0.5 )**2 * np.where(xs>0.5, Ae_As/4.0, 1.0)
Ax = 1.0 + (Ae_As-1.0)*((xs-0.5)/0.5)**2
M_sub = np.array([mach_from_area(a, "sub") for a in Ax])
M_sup = np.array([mach_from_area(a, "super") for a in Ax])
# design (fully supersonic) operation: subsonic converging, supersonic diverging
M_noz = np.where(xs<=0.5, M_sub, M_sup)
Me = mach_from_area(Ae_As, "super")
print(f"  C-D nozzle Ae/A*={Ae_As}: design exit Mach Me = {Me:.4f}, "
      f"pe/p0 = {1/p0_p(Me):.5f}")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
ax1.plot(xs, np.sqrt(Ax), "C0", lw=2); ax1.plot(xs, -np.sqrt(Ax), "C0", lw=2)
ax1.fill_between(xs, np.sqrt(Ax), 3, color="0.9"); ax1.fill_between(xs,-np.sqrt(Ax),-3,color="0.9")
ax1.axvline(0.5, color="0.6", ls=":", lw=1); ax1.text(0.5,0.1,"throat",ha="center",fontsize=8)
ax1.set_title("Converging-diverging nozzle"); ax1.set_xlabel("x"); ax1.axis("equal"); ax1.axis("off")
ax2.plot(xs, M_noz, "C0-", lw=2, label="Mach number")
ax2b=ax2.twinx()
ax2b.plot(xs, 1/p0_p(M_noz), "C3--", lw=2, label="$p/p_0$")
ax2.axhline(1.0,color="0.6",ls=":",lw=1); ax2.text(0.02,1.05,"M=1",fontsize=8)
ax2.set_xlabel("x"); ax2.set_ylabel("M",color="C0"); ax2b.set_ylabel("$p/p_0$",color="C3")
ax2.set_title("Isentropic nozzle flow (design)")
fig.suptitle("Isentropic converging-diverging nozzle flow", y=1.04, fontsize=13)
fig.savefig("fig9_1_nozzle.png", dpi=150, bbox_inches="tight")
print("  Wrote fig9_1_nozzle.png")
