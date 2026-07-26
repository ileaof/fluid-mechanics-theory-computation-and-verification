#!/usr/bin/env python3
"""
Example 9.2 -- Normal shock relations and a shock inside a nozzle.

A normal shock is a thin, irreversible compression across which the flow jumps from
supersonic to subsonic.  For a perfect gas the downstream Mach number and property
ratios are given by the Rankine-Hugoniot relations

        M2^2   = ( 1 + (g-1)/2 M1^2 ) / ( g M1^2 - (g-1)/2 )
        p2/p1  = 1 + 2g/(g+1) (M1^2 - 1)
        rho2/rho1 = (g+1) M1^2 / ( 2 + (g-1) M1^2 )
        T2/T1  = (p2/p1)(rho1/rho2) .

These algebraic relations are VERIFIED here against the three fundamental
conservation laws -- mass, momentum and energy -- solved independently across the
shock, which must agree to machine precision.  The stagnation-pressure ratio
p02/p01 < 1 and the entropy rise are computed, and finally the position of a shock
inside a converging-diverging nozzle is found by matching the exit pressure to a
specified back pressure with a bracketed root finder.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

g, R = 1.4, 287.0

def M2_of(M1):   return np.sqrt((1+0.5*(g-1)*M1**2)/(g*M1**2-0.5*(g-1)))
def p_ratio(M1): return 1 + 2*g/(g+1)*(M1**2-1)
def rho_ratio(M1): return (g+1)*M1**2/(2+(g-1)*M1**2)
def T_ratio(M1): return p_ratio(M1)/rho_ratio(M1)
def p0_p(M):     return (1+0.5*(g-1)*M**2)**(g/(g-1))
def p02_p01(M1): return (p0_p(M2_of(M1))/1.0) * (p_ratio(M1)) / (p0_p(M1))
def area_ratio(M): return (1/M)*((2/(g+1))*(1+0.5*(g-1)*M**2))**((g+1)/(2*(g-1)))

print("Example 9.2  Normal shock relations (gamma = 1.4)\n")
print("  Verify Rankine-Hugoniot against independent mass/momentum/energy balance:")
print(f"    {'M1':>5} {'M2':>8} {'p2/p1':>8} {'cons.resid':>12} {'p02/p01':>9} {'ds/R':>8}")
worst=0.0
# upstream reference state
T1, p1 = 300.0, 1.0e5
for M1 in (1.5, 2.0, 3.0, 4.0):
    rho1 = p1/(R*T1); a1 = np.sqrt(g*R*T1); u1 = M1*a1
    # algebraic (Rankine-Hugoniot) downstream state
    p2 = p1*p_ratio(M1); rho2 = rho1*rho_ratio(M1); T2 = T1*T_ratio(M1)
    u2 = u1*rho1/rho2                              # from mass conservation
    M2 = M2_of(M1)
    # independent check: residuals of the three conservation laws
    mass = rho1*u1 - rho2*u2
    mom  = (p1+rho1*u1**2) - (p2+rho2*u2**2)
    cp = g*R/(g-1)
    ener = (cp*T1+0.5*u1**2) - (cp*T2+0.5*u2**2)
    resid = max(abs(mass)/(rho1*u1), abs(mom)/(p1+rho1*u1**2), abs(ener)/(cp*T1))
    worst=max(worst,resid)
    # entropy rise across the shock
    ds_R = (g/(g-1))*np.log(T2/T1) - np.log(p2/p1)   # ds/R for perfect gas
    print(f"    {M1:5.1f} {M2:8.4f} {p_ratio(M1):8.4f} {resid:12.2e}"
          f" {p02_p01(M1):9.5f} {ds_R:8.4f}")
    assert M2 < 1.0, "downstream must be subsonic"
    assert ds_R > 0, "entropy must increase across a shock"
assert worst < 1e-12, "Rankine-Hugoniot inconsistent with conservation laws"
print(f"\n  Worst conservation-law residual: {worst:.2e}")
print("  PASS: shock relations satisfy mass, momentum, energy; ds>0; M2<1.\n")

# --- shock in a converging-diverging nozzle ---------------------------------
# Nozzle Ae/At given; find shock area-location A_s/At so exit pressure = back pressure
Ae_At = 3.0
p0 = 1.0e5
Me_design = brentq(lambda M: area_ratio(M)-Ae_At, 1.0, 10.0)  # supersonic design exit
print(f"  C-D nozzle Ae/At = {Ae_At}: supersonic design exit M = {Me_design:.3f}")

# Parametrise the shock by the Mach number M1s just upstream of it (M1s >= 1),
# so that its area location is A_s/At = area_ratio(M1s).  After the shock the sonic
# reference area grows to A*2 = A_s/area_ratio(M2), and the subsonic exit Mach
# follows from Ae/A*2.  The exit static pressure then fixes the back pressure.
def exit_p_over_p0(M1s):
    As_At = area_ratio(M1s)
    if As_At > Ae_At: return None
    M2s = M2_of(M1s)
    Astar2_over_At = As_At/area_ratio(M2s)
    Ae_Astar2 = Ae_At/Astar2_over_At
    Me = brentq(lambda M: area_ratio(M)-Ae_Astar2, 1e-6, 1.0)  # subsonic exit
    p02 = p0*p02_p01(M1s)
    pe = p02/p0_p(Me)
    return pe/p0, Me, As_At

# choose a back pressure that puts a shock in the diverging section
pb_over_p0 = 0.60
sol = brentq(lambda M1s: exit_p_over_p0(M1s)[0]-pb_over_p0, 1.0001, Me_design-1e-4)
pe_ratio, Me_sub, As_At = exit_p_over_p0(sol)
print(f"  Back pressure pb/p0 = {pb_over_p0}: shock at A_s/At = {As_At:.3f} "
      f"(M just before shock = {sol:.3f})")
print(f"    subsonic exit Mach = {Me_sub:.3f}, matched exit pressure pe/p0 = {pe_ratio:.4f}")
assert abs(pe_ratio - pb_over_p0) < 1e-6
print("  PASS: shock position found; nozzle exit pressure matches back pressure.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
M1r=np.linspace(1,5,200)
ax1.plot(M1r, [p_ratio(m) for m in M1r], "C0-", lw=2, label="$p_2/p_1$")
ax1.plot(M1r, [rho_ratio(m) for m in M1r], "C3--", lw=2, label=r"$\rho_2/\rho_1$")
ax1.plot(M1r, [M2_of(m) for m in M1r], "C2-.", lw=2, label="$M_2$")
ax1.set_xlabel("upstream Mach $M_1$"); ax1.set_title("Normal-shock jumps")
ax1.legend(frameon=False); ax1.grid(alpha=0.3)
ax2.plot(M1r, [p02_p01(m) for m in M1r], "C0-", lw=2)
ax2.set_xlabel("upstream Mach $M_1$"); ax2.set_ylabel("$p_{02}/p_{01}$")
ax2.set_title("Stagnation-pressure loss across a shock"); ax2.grid(alpha=0.3)
fig.suptitle("Normal shock waves", y=1.04, fontsize=13)
fig.savefig("fig9_2_shock.png", dpi=150, bbox_inches="tight")
print("  Wrote fig9_2_shock.png")
