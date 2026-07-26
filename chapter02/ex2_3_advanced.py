#!/usr/bin/env python3
"""
Example 2.3 -- Advanced verification: hydrostatic force on a vertical gate in a
linearly stratified fluid.

Oceans, reservoirs and settling tanks are often density-stratified: the fluid is
heavier at depth.  Take a linear stratification

        rho(z) = rho0 (1 + beta z),        z = depth below the free surface,

so the gauge pressure, from the hydrostatic equation dp/dz = rho(z) g, is

        p(z) = g rho0 ( z + beta z^2 / 2 ) ,

which is QUADRATIC in depth rather than linear.  On a vertical gate of unit width
spanning 0 <= z <= H the resultant force per unit width and the depth of its line
of action (the centre of pressure) have the closed forms

        F    = g rho0 ( H^2/2 + beta H^3/6 ),
        z_cp = [ integral z p dz ] / F
             = g rho0 ( H^3/3 + beta H^4/8 ) / F .

Because there is no analytical shortcut for a general stratification, this is an
ideal verification problem: we compute F and z_cp by the composite midpoint rule
on N panels and confirm second-order convergence to the closed forms, with a
Richardson extrapolation, a Grid Convergence Index, and CPU timing.  A sensitivity
sweep varies the stratification strength beta.

Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g    = 9.81
rho0 = 1000.0        # surface density [kg/m^3]
H    = 4.0           # gate height / depth [m]

def pressure(z, beta):
    return g * rho0 * (z + 0.5 * beta * z**2)

def F_exact(beta):
    return g * rho0 * (H**2 / 2.0 + beta * H**3 / 6.0)

def zcp_exact(beta):
    num = g * rho0 * (H**3 / 3.0 + beta * H**4 / 8.0)
    return num / F_exact(beta)

def midpoint_force(N, beta):
    """Composite midpoint quadrature of F and z_cp over the gate (unit width)."""
    dz = H / N
    zc = (np.arange(N) + 0.5) * dz          # panel midpoints
    p  = pressure(zc, beta)
    F  = np.sum(p) * dz
    zcp = np.sum(zc * p) * dz / F
    return F, zcp

print("Example 2.3  Hydrostatic force on a gate in a linearly stratified fluid\n")
beta = 0.05          # 1/m -> density rises 20% over the 4 m gate
print(f"  rho(z) = {rho0:.0f}(1 + {beta} z);  rho(H)/rho0 = {1+beta*H:.2f}")
print(f"  F_exact    = {F_exact(beta)/1e3:.6f} kN/m")
print(f"  z_cp_exact = {zcp_exact(beta):.6f} m  (homogeneous limit would be {2*H/3:.4f} m)\n")

print(f"  {'N':>6} {'dz [m]':>9} {'F error [N/m]':>14} {'p(F)':>7}"
      f" {'zcp error [m]':>14} {'p(zcp)':>8} {'CPU[ms]':>9}")
Ns = [8, 16, 32, 64, 128, 256, 512]
rows = []
for N in Ns:
    t0 = time.perf_counter()
    F, zcp = midpoint_force(N, beta)
    cpu = (time.perf_counter() - t0) * 1e3
    eF   = abs(F - F_exact(beta))
    ezcp = abs(zcp - zcp_exact(beta))
    rows.append((N, H/N, F, zcp, eF, ezcp, cpu))

for k, r in enumerate(rows):
    if k == 0:
        pF = pz = float("nan")
    else:
        pF = np.log(rows[k-1][4] / r[4]) / np.log(2.0)
        pz = np.log(rows[k-1][5] / r[5]) / np.log(2.0)
    print(f"  {r[0]:6d} {r[1]:9.4f} {r[4]:14.4e} {pF:7.3f}"
          f" {r[5]:14.4e} {pz:8.3f} {r[6]:9.3f}")

pF_final = np.log(rows[-2][4] / rows[-1][4]) / np.log(2.0)
print(f"\n  Observed order of accuracy (force, finest pair): p = {pF_final:.4f}")
assert 1.9 < pF_final < 2.1, "quadrature did not achieve second order"

# ---- Richardson extrapolation + GCI on the resultant force F ----------------
F3, F2, F1 = rows[-3][2], rows[-2][2], rows[-1][2]   # coarse, medium, fine
r = 2.0
p_obs = np.log(abs((F3 - F2) / (F2 - F1))) / np.log(r)
F_ext = F1 + (F1 - F2) / (r**p_obs - 1.0)
Fs = 1.25
GCI_fine = Fs * abs((F1 - F2) / F1) / (r**p_obs - 1.0)
print("\n  Richardson extrapolation of the resultant force F:")
print(f"    coarse (N={rows[-3][0]})  F = {F3:.6f} N/m")
print(f"    medium (N={rows[-2][0]})  F = {F2:.6f} N/m")
print(f"    fine   (N={rows[-1][0]}) F = {F1:.6f} N/m")
print(f"    observed order p        = {p_obs:.4f}")
print(f"    extrapolated F(h->0)    = {F_ext:.6f} N/m")
print(f"    exact F                 = {F_exact(beta):.6f} N/m")
print(f"    |F_ext - F_exact|       = {abs(F_ext - F_exact(beta)):.3e} N/m")
print(f"    GCI_fine (Fs={Fs})       = {GCI_fine*100:.5f} %")
band = GCI_fine * abs(F1)
print(f"    exact within GCI band?  = {abs(F1 - F_exact(beta)) <= band}")
assert abs(F_ext - F_exact(beta)) < 1e-3, "Richardson estimate inconsistent"
print("  PASS: order ~2, Richardson matches exact, exact within GCI band.\n")

# ---- sensitivity to stratification strength beta ---------------------------
print("  Sensitivity of the resultant force to stratification beta:")
print(f"    {'beta[1/m]':>10} {'rho(H)/rho0':>12} {'F [kN/m]':>11} {'shift vs homog.':>16}")
F_homog = g * rho0 * H**2 / 2.0
for b in (0.0, 0.02, 0.05, 0.10, 0.20):
    Fb = F_exact(b)
    print(f"    {b:10.2f} {1+b*H:12.2f} {Fb/1e3:11.4f} {(Fb/F_homog-1)*100:14.1f} %")
print()

# ---- figures ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
zz = np.linspace(0, H, 200)
ax1.plot(rho0*(1+beta*zz), zz, "C0-", lw=2, label=r"$\rho(z)$")
ax1.set_xlabel(r"density $\rho(z)$ [kg/m$^3$]", color="C0")
ax1.tick_params(axis="x", colors="C0")
ax1b = ax1.twiny()
ax1b.plot(pressure(zz, beta)/1e3, zz, "C3--", lw=2)
ax1b.plot(g*rho0*zz/1e3, zz, "0.6", ls=":", lw=1.5)
ax1b.set_xlabel("pressure $p(z)$ [kPa]  (dotted: homogeneous)", color="C3")
ax1b.tick_params(axis="x", colors="C3")
ax1.set_ylabel("depth $z$ [m]"); ax1.invert_yaxis()
ax1.set_title("Stratified density and pressure")
ax1.grid(alpha=0.3)

hs = np.array([r[1] for r in rows])
eFs = np.array([r[4] for r in rows]); ezs = np.array([r[5] for r in rows])
ax2.loglog(hs, eFs, "o-", lw=1.8, label="force error")
ax2.loglog(hs, ezs, "s--", lw=1.8, label=r"$z_{cp}$ error")
ax2.loglog(hs, eFs[-1]*(hs/hs[-1])**2, "k:", lw=1.4, label="slope 2")
ax2.set_xlabel("panel size $\\Delta z$ [m]"); ax2.set_ylabel("absolute error")
ax2.set_title("Grid convergence (midpoint quadrature)")
ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Hydrostatic force on a gate in a stratified fluid", y=1.05, fontsize=13)
fig.savefig("fig2_3_stratified.png", dpi=150, bbox_inches="tight")
print("  Wrote fig2_3_stratified.png")
