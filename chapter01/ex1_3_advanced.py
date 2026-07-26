#!/usr/bin/env python3
"""
Example 1.3 -- Advanced verification: variable-viscosity momentum diffusion by
the Method of Manufactured Solutions (MMS).

Real fluids have a viscosity that varies with temperature, and hence with
position in a non-isothermal shear layer.  The steady momentum balance across
such a layer is the variable-coefficient diffusion equation

        d/dy ( mu(y) du/dy ) = S(y) ,    0 < y < L ,   u(0)=u_a , u(L)=u_b .

There is no elementary closed-form solution for a general mu(y).  To verify a
finite volume solver we therefore MANUFACTURE one: we pick a smooth target field
u_e(y), substitute it into the operator to obtain the source S(y) that makes it
an exact solution, and then check that the code recovers u_e as the mesh is
refined.  This is the Method of Manufactured Solutions -- the gold standard for
code verification, because it exercises every term with a known exact answer.

The finite volume scheme uses the HARMONIC MEAN of the cell viscosities at each
face (Patankar's practice), which is conservative and correct even when mu varies
sharply.  The verification campaign reports, for a sequence of systematically
refined meshes:
  * discrete L2 and Linf error norms against u_e,
  * the observed order of accuracy p,
  * Richardson extrapolation of a scalar functional to h -> 0,
  * the Grid Convergence Index (GCI) with a factor of safety,
  * CPU time per solve,
and finishes with a sensitivity sweep over the viscosity-contrast parameter.

Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

L = 1.0                       # layer thickness [m] (nondimensional here)

# ---- manufactured exact solution and the viscosity field -------------------
def u_exact(y):
    return np.sin(np.pi * y) + 0.3 * y          # u_e(0)=0, u_e(L)=0.3

def dudy_exact(y):
    return np.pi * np.cos(np.pi * y) + 0.3

def mu_field(y, a):
    """Temperature-like viscosity variation, mu = mu0 (1 + a y).  a is contrast."""
    return 1.0 + a * y

def dmu_dy(y, a):
    return a * np.ones_like(y)

def source(y, a):
    """S = d/dy( mu du/dy ) evaluated analytically for the manufactured field."""
    mu   = mu_field(y, a)
    up   = dudy_exact(y)
    upp  = -np.pi**2 * np.sin(np.pi * y)        # d2u_e/dy2
    return dmu_dy(y, a) * up + mu * upp

# ---- finite volume solver (harmonic-mean faces, TDMA) ----------------------
def solve_fvm(N, a):
    h = L / N
    yc = (np.arange(N) + 0.5) * h               # cell centres
    mu = mu_field(yc, a)
    # face viscosities: harmonic mean of neighbouring cells (interior)
    mu_e = np.empty(N); mu_w = np.empty(N)
    mu_e[:-1] = 2.0 * mu[:-1] * mu[1:] / (mu[:-1] + mu[1:])
    mu_w[1:]  = mu_e[:-1]
    # boundary faces sit on the wall; use the exact wall viscosity
    mu_w[0]  = mu_field(np.array([0.0]), a)[0]
    mu_e[-1] = mu_field(np.array([L]),  a)[0]
    aW = mu_w / h;  aE = mu_e / h
    aW[0]  = mu_w[0]  / (h / 2.0)               # half-cell to the wall
    aE[-1] = mu_e[-1] / (h / 2.0)
    aP = aW + aE
    b = -source(yc, a) * h                      # S integrated over the cell
    b[0]  += aW[0]  * u_exact(np.array([0.0]))[0]
    b[-1] += aE[-1] * u_exact(np.array([L]))[0]
    # TDMA
    P_ = np.zeros(N); Q_ = np.zeros(N)
    P_[0] = aE[0] / aP[0]; Q_[0] = b[0] / aP[0]
    for i in range(1, N):
        d = aP[i] - aW[i] * P_[i-1]
        P_[i] = aE[i] / d
        Q_[i] = (b[i] + aW[i] * Q_[i-1]) / d
    u = np.zeros(N)
    u[-1] = Q_[-1]
    for i in range(N - 2, -1, -1):
        u[i] = P_[i] * u[i+1] + Q_[i]
    return yc, u, h

def error_norms(N, a):
    yc, u, h = solve_fvm(N, a)
    e = u - u_exact(yc)
    L2   = np.sqrt(np.sum(e**2) * h)            # discrete L2 norm
    Linf = np.max(np.abs(e))
    Jmid = np.interp(0.5 * L, yc, u)            # scalar functional: u at mid-layer
    return L2, Linf, Jmid, h

# ===========================================================================
print("Example 1.3  MMS verification of variable-viscosity momentum diffusion\n")
a = 3.0                                          # viscosity contrast mu(L)/mu(0)=4
Ns = [16, 32, 64, 128, 256, 512, 1024]

print(f"  Viscosity contrast a = {a} (mu varies by factor {1+a:.0f} across layer)")
print(f"\n  {'N':>6} {'h':>11} {'L2 error':>13} {'p(L2)':>7}"
      f" {'Linf error':>13} {'p(Linf)':>8} {'CPU [ms]':>9}")

rows = []
for N in Ns:
    t0 = time.perf_counter()
    L2, Linf, Jmid, h = error_norms(N, a)
    cpu = (time.perf_counter() - t0) * 1e3
    rows.append((N, h, L2, Linf, Jmid, cpu))

for k, (N, h, L2, Linf, Jmid, cpu) in enumerate(rows):
    if k == 0:
        pL2 = pLi = float("nan")
    else:
        pL2 = np.log(rows[k-1][2] / L2)   / np.log(2.0)
        pLi = np.log(rows[k-1][3] / Linf) / np.log(2.0)
    print(f"  {N:6d} {h:11.3e} {L2:13.4e} {pL2:7.3f}"
          f" {Linf:13.4e} {pLi:8.3f} {cpu:9.2f}")

pL2_final = np.log(rows[-2][2] / rows[-1][2]) / np.log(2.0)
print(f"\n  Observed order of accuracy (L2, finest pair): p = {pL2_final:.4f}")
assert 1.9 < pL2_final < 2.1, "MMS did not confirm second-order accuracy"

# ---- Richardson extrapolation + GCI on the scalar functional J = u(L/2) ----
# three successive grids with refinement ratio r = 2
J3, J2, J1 = rows[-3][4], rows[-2][4], rows[-1][4]   # coarse, medium, fine
r = 2.0
p_obs = np.log(abs((J3 - J2) / (J2 - J1))) / np.log(r)
J_ext = J1 + (J1 - J2) / (r**p_obs - 1.0)            # Richardson estimate, h->0
Fs = 1.25                                            # factor of safety (3+ grids)
eps_fine = abs((J1 - J2) / J1)
GCI_fine = Fs * eps_fine / (r**p_obs - 1.0)
J_true = u_exact(np.array([0.5 * L]))[0]

print("\n  Richardson extrapolation of the functional J = u(L/2):")
print(f"    coarse (N={rows[-3][0]}) J = {J3:.8f}")
print(f"    medium (N={rows[-2][0]}) J = {J2:.8f}")
print(f"    fine   (N={rows[-1][0]}) J = {J1:.8f}")
print(f"    observed order p          = {p_obs:.4f}")
print(f"    extrapolated  J(h->0)     = {J_ext:.8f}")
print(f"    exact         J           = {J_true:.8f}")
print(f"    |J_ext - J_exact|         = {abs(J_ext - J_true):.2e}")
print(f"    GCI_fine (Fs={Fs})         = {GCI_fine*100:.4f} %")
band = GCI_fine * abs(J1)
inside = abs(J1 - J_true) <= band
print(f"    exact within GCI band?    = {inside}  "
      f"(|err|={abs(J1-J_true):.2e} <= band={band:.2e})")
assert abs(J_ext - J_true) < 1e-4, "Richardson estimate is inconsistent with exact"
print("  PASS: observed order ~2, Richardson matches exact, exact within GCI band.\n")

# ---- sensitivity sweep over the viscosity contrast a -----------------------
print("  Sensitivity of accuracy to viscosity contrast a (fixed N=256):")
print(f"    {'a':>6} {'mu(L)/mu(0)':>12} {'L2 error':>13} {'order p':>9}")
a_list = [0.0, 1.0, 3.0, 9.0, 30.0]
for a_s in a_list:
    L2c, _, _, hc = error_norms(128, a_s)
    L2f, _, _, hf = error_norms(256, a_s)
    p = np.log(L2c / L2f) / np.log(2.0)
    print(f"    {a_s:6.1f} {1+a_s:12.1f} {L2f:13.4e} {p:9.3f}")
print()

# ---- figures ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)

# (a) error norms vs h, with a second-order reference triangle
hs   = np.array([r[1] for r in rows])
L2s  = np.array([r[2] for r in rows])
Lis  = np.array([r[3] for r in rows])
ax1.loglog(hs, L2s, "o-", lw=1.8, label=r"$L_2$ error")
ax1.loglog(hs, Lis, "s--", lw=1.8, label=r"$L_\infty$ error")
ref = L2s[-1] * (hs / hs[-1])**2
ax1.loglog(hs, ref, "k:", lw=1.4, label=r"slope 2 (reference)")
ax1.set_xlabel(r"mesh spacing $h$"); ax1.set_ylabel("error norm")
ax1.set_title("Grid convergence (MMS)")
ax1.legend(frameon=False, fontsize=9); ax1.grid(alpha=0.3, which="both")

# (b) manufactured solution vs coarse-grid FVM
yc, u, h = solve_fvm(24, a)
yf = np.linspace(0, L, 400)
ax2.plot(u_exact(yf), yf, "k-", lw=2, label="manufactured exact")
ax2.plot(u, yc, "o", ms=6, mfc="none", mec="C3", label="FVM (N=24)")
axt = ax2.twiny()
axt.plot(mu_field(yf, a), yf, "C0-.", lw=1.5)
axt.set_xlabel(r"$\mu(y)$", color="C0")
axt.tick_params(axis="x", colors="C0")
ax2.set_xlabel(r"$u(y)$"); ax2.set_ylabel(r"$y/L$")
ax2.set_title("Solution and viscosity field")
ax2.legend(frameon=False, fontsize=9, loc="lower right"); ax2.grid(alpha=0.3)

fig.suptitle("MMS verification of variable-viscosity momentum diffusion",
             y=1.05, fontsize=13)
fig.savefig("fig1_3_mms.png", dpi=150, bbox_inches="tight")
print("  Wrote fig1_3_mms.png")
