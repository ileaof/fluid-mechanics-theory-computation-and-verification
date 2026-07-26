#!/usr/bin/env python3
"""
Example 7.2 -- Finite-volume solution of Hagen-Poiseuille flow (cylindrical).

The same fully developed pipe flow solved analytically in Example 7.1,

        (1/r) d/dr( r mu du/dr ) = dp/dx ,   u(R)=0 , du/dr(0)=0 ,

is now solved with a cell-centred finite-volume method in CYLINDRICAL coordinates.
Integrating over an annular control volume of a cell centred at r_i turns the
operator into a balance of face fluxes weighted by the face radius:

        mu r_e (u_{i+1}-u_i)/dr - mu r_w (u_i-u_{i-1})/dr = (dp/dx) r_i dr .

The west face of the first cell lies on the axis (r = 0), where r*du/dr = 0 by
symmetry, so that face carries no flux; the east face of the last cell lies on the
wall (u = 0).  The tridiagonal system is solved by the Thomas algorithm, and a
mesh-refinement study confirms second-order convergence to the analytical parabola.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mu   = 1.0e-3
R    = 0.01
dpdx = -4.0

def u_exact(r):
    return (-dpdx) * (R**2 - r**2) / (4*mu)

def solve_fvm(N):
    dr = R / N
    rc = (np.arange(N) + 0.5) * dr            # cell centres
    rf = np.arange(N + 1) * dr                # faces (r=0 .. R)
    aW = np.zeros(N); aE = np.zeros(N)
    for i in range(N):
        rw, re = rf[i], rf[i+1]
        aW[i] = mu * rw / dr                  # west face weight (0 at axis)
        aE[i] = mu * re / dr
    aE[-1] = mu * R / (dr/2)                  # wall: half-cell to u(R)=0
    aP = aW + aE
    b = (-dpdx) * rc * dr                      # source S=-dp/dx times r dr (S=+ (-dpdx))
    # axis: aW[0]=0 (rw=0) -> no west neighbour; wall: u(R)=0 contributes 0 to b
    # Thomas
    P = np.zeros(N); Q = np.zeros(N)
    P[0] = aE[0]/aP[0]; Q[0] = b[0]/aP[0]
    for i in range(1, N):
        d = aP[i] - aW[i]*P[i-1]
        P[i] = aE[i]/d
        Q[i] = (b[i] + aW[i]*Q[i-1])/d
    u = np.zeros(N); u[-1] = Q[-1]
    for i in range(N-2, -1, -1):
        u[i] = P[i]*u[i+1] + Q[i]
    return rc, u

print("Example 7.2  Finite-volume Hagen-Poiseuille (cylindrical coordinates)\n")
print(f"  {'N':>5} {'dr [m]':>11} {'max|err| [m/s]':>16} {'order p':>9}")
prev = None; errs = []
for N in [10, 20, 40, 80, 160, 320]:
    rc, u = solve_fvm(N)
    err = np.max(np.abs(u - u_exact(rc)))
    p = np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"  {N:5d} {R/N:11.3e} {err:16.4e} {p:9.3f}")
    errs.append((R/N, err)); prev = err

p_final = np.log(errs[-2][1]/errs[-1][1])/np.log(errs[-2][0]/errs[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 1.8 < p_final < 2.2, "axisymmetric FVM is not second order"
# verify discrete flow rate against Hagen-Poiseuille
rc, u = solve_fvm(160)
dr = R/160
Q_num = np.sum(u * 2*np.pi*rc * dr)
Q_ex  = np.pi * R**4 * (-dpdx)/(8*mu)
print(f"  Discrete Q = {Q_num:.6e},  exact = {Q_ex:.6e},  rel err = {abs(Q_num-Q_ex)/Q_ex:.2e}")
assert abs(Q_num-Q_ex)/Q_ex < 1e-3
print("  PASS: cylindrical FVM converges at 2nd order; flow rate verified.\n")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
rc, u = solve_fvm(20)
rf = np.linspace(0, R, 300)
ax1.plot(u_exact(rf), rf/R, "k-", lw=2, label="analytical")
ax1.plot(u, rc/R, "o", ms=6, mfc="none", mec="C3", label="FVM (N=20)")
ax1.set_xlabel("$u(r)$ [m/s]"); ax1.set_ylabel("$r/R$")
ax1.set_title("FVM vs analytical parabola"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
hs = np.array([e[0] for e in errs]); es = np.array([e[1] for e in errs])
ax2.loglog(hs, es, "o-", lw=1.8, label="max error")
ax2.loglog(hs, es[-1]*(hs/hs[-1])**2, "k:", lw=1.4, label="slope 2")
ax2.set_xlabel("mesh spacing $\\Delta r$ [m]"); ax2.set_ylabel("max error [m/s]")
ax2.set_title("Grid convergence"); ax2.legend(frameon=False); ax2.grid(alpha=0.3, which="both")
fig.suptitle("Finite-volume Hagen-Poiseuille flow", y=1.04, fontsize=13)
fig.savefig("fig7_2_fvm.png", dpi=150, bbox_inches="tight")
print("  Wrote fig7_2_fvm.png")
