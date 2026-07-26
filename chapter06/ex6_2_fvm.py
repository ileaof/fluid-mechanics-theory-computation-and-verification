#!/usr/bin/env python3
"""
Example 6.2 -- Dynamic similarity: the friction factor f(Re, eps/D).

Dimensional analysis reduces the pressure drop in a rough pipe to a single
dimensionless law, the Darcy friction factor as a function of just two groups,

        f = phi( Re , eps/D ) ,

so that pipes of every size and fluid collapse onto one chart (the Moody chart).
In the laminar regime f = 64/Re exactly (from the Hagen-Poiseuille solution of
Chapter 1).  In the turbulent regime f is given implicitly by the Colebrook
equation

        1/sqrt(f) = -2 log10( (eps/D)/3.7 + 2.51/(Re sqrt(f)) ),

which must be solved numerically.  This program solves Colebrook with a bracketed
root finder (Brent's method), verifies the residual is driven to zero, compares
the result with Haaland's explicit approximation, and shows the family of curves
that constitute the Moody chart.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def colebrook(Re, eps_D):
    """Solve the Colebrook equation for the Darcy friction factor."""
    def g(f):
        return 1.0/np.sqrt(f) + 2.0*np.log10(eps_D/3.7 + 2.51/(Re*np.sqrt(f)))
    return brentq(g, 1e-4, 1.0, xtol=1e-12)

def haaland(Re, eps_D):
    """Haaland's explicit approximation to Colebrook."""
    return (-1.8*np.log10((eps_D/3.7)**1.11 + 6.9/Re))**-2

print("Example 6.2  Friction factor and dynamic similarity\n")

# --- verify Colebrook residual and compare to Haaland ------------------------
print("  Colebrook solved by Brent's method; residual and Haaland comparison:")
print(f"    {'Re':>10} {'eps/D':>8} {'f_Colebrook':>13} {'residual':>11}"
      f" {'f_Haaland':>11} {'diff %':>8}")
worst_res = 0.0; worst_diff = 0.0
for Re in (1e4, 1e5, 1e6):
    for eps_D in (0.0, 1e-4, 1e-2):
        f = colebrook(Re, eps_D)
        res = abs(1/np.sqrt(f) + 2*np.log10(eps_D/3.7 + 2.51/(Re*np.sqrt(f))))
        fh = haaland(Re, eps_D)
        diff = abs(fh-f)/f*100
        worst_res = max(worst_res, res); worst_diff = max(worst_diff, diff)
        print(f"    {Re:10.0e} {eps_D:8.4f} {f:13.6f} {res:11.2e} {fh:11.6f} {diff:8.2f}")
assert worst_res < 1e-8, "Colebrook root not converged"
assert worst_diff < 3.0, "Haaland differs from Colebrook by more than 3%"
print(f"\n  Worst Colebrook residual {worst_res:.1e}; worst Haaland deviation {worst_diff:.2f} %")

# --- laminar branch verification --------------------------------------------
Re_lam = np.array([500, 1000, 2000])
f_lam = 64.0/Re_lam
print(f"\n  Laminar branch f = 64/Re (exact from Hagen-Poiseuille): "
      f"Re=1000 -> f={64/1000:.4f}")
print("  PASS: Colebrook converged, matches Haaland, laminar branch exact.\n")

# --- dynamic similarity: different pipes/fluids collapse onto one curve ------
Re_target, epsD_common = 1.0e5, 1.0e-3
print(f"  Dynamic similarity: three different pipes/fluids, all set to Re={Re_target:.0e},")
print(f"  eps/D={epsD_common} by choosing the velocity V = Re mu/(rho D):")
print(f"    {'case':16s} {'rho':>6} {'mu':>8} {'D[m]':>6} {'V[m/s]':>8} {'Re':>9} {'f':>9}")
# (name, rho, mu, D): velocity computed to hit Re_target exactly
cases = [("water, 50 mm", 1000., 1.0e-3, 0.05),
         ("air,   100 mm", 1.2,   1.8e-5, 0.10),
         ("light oil,100mm",850., 5.0e-3, 0.10)]
fs=[]
for name, rho, mu, D in cases:
    V = Re_target*mu/(rho*D)
    Re = rho*V*D/mu
    f = colebrook(Re, epsD_common)
    fs.append(f)
    print(f"    {name:16s} {rho:6.0f} {mu:8.1e} {D:6.3f} {V:8.3f} {Re:9.0f} {f:9.5f}")
assert max(fs)-min(fs) < 1e-9, "matched-Re cases should give identical f"
print(f"  Spread in f across the three cases: {max(fs)-min(fs):.2e}")
print("  PASS: identical (Re, eps/D) give identical f -> dynamic similarity.\n")

# --- figure: the Moody chart ------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
Re_t = np.logspace(np.log10(4000), 8, 300)
for eps_D in (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 2e-2, 5e-2):
    f = np.array([colebrook(Re, eps_D) for Re in Re_t])
    ax.loglog(Re_t, f, lw=1.6, label=f"{eps_D:g}")
Re_l = np.logspace(np.log10(500), np.log10(2300), 50)
ax.loglog(Re_l, 64/Re_l, "k-", lw=2.5, label="laminar 64/Re")
ax.set_xlabel(r"Reynolds number $\mathrm{Re}=\rho V D/\mu$")
ax.set_ylabel(r"friction factor $f$")
ax.set_title("Moody chart: $f=\\phi(\\mathrm{Re},\\ \\varepsilon/D)$ from similarity")
ax.legend(title=r"$\varepsilon/D$", frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.3, which="both")
fig.savefig("fig6_2_moody.png", dpi=150, bbox_inches="tight")
print("  Wrote fig6_2_moody.png")
