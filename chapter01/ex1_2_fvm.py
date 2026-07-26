#!/usr/bin/env python3
"""
Example 1.2 -- Finite Volume Method for plane Couette-Poiseuille flow.

The same boundary-value problem solved analytically in Example 1.1,

        d/dy ( mu du/dy ) = dp/dx ,    u(0)=0 ,  u(H)=U ,

is now solved with a cell-centred finite volume method:

  * MESH: the gap [0,H] is split into N control volumes of width h = H/N; the
    unknown u_i lives at each cell centre y_i = (i-1/2) h.
  * DISCRETISATION: integrating the momentum equation over a control volume and
    approximating the face fluxes  mu du/dy  by central differences gives, for an
    interior cell,  mu(u_{i-1} - 2 u_i + u_{i+1})/h = (dp/dx) h.
  * BOUNDARY CONDITIONS: the walls sit on cell faces half a cell from the nearest
    centre, so the wall flux uses a half-width difference (Dirichlet).
  * LINEAR SYSTEM: the assembled coefficient matrix is tridiagonal (a_W, a_P, a_E).
  * SOLVER: point Gauss-Seidel with under-relaxation, so a residual history can be
    monitored to convergence (the exact TDMA answer is also computed as a check).
  * VERIFICATION: the converged field is compared with the Example 1.1 analytical
    solution.  Because the exact solution is a quadratic, the second-order scheme
    reproduces it to round-off at the cell centres.

Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H  = 2.0e-3
U  = 0.50
mu = 1.0e-3
P  = 1.5                                   # dimensionless pressure gradient
dpdx = -P * 2.0 * mu * U / H**2            # matching physical gradient [Pa/m]

def u_exact(y):
    return U * (y / H) + (1.0 / (2.0 * mu)) * dpdx * (y**2 - H * y)

def assemble(N):
    """Return tridiagonal coefficients (aW, aP, aE) and source b for N cells."""
    h = H / N
    y = (np.arange(N) + 0.5) * h
    aW = np.full(N, mu / h)
    aE = np.full(N, mu / h)
    # Dirichlet walls: face is h/2 from the boundary cell centre
    aW[0]    = mu / (h / 2.0)
    aE[-1]   = mu / (h / 2.0)
    b = np.full(N, -dpdx * h)              # S = -dp/dx, integrated over the cell
    b[0]  += aW[0]  * 0.0                   # u(0)=0 contributes nothing
    b[-1] += aE[-1] * U                     # u(H)=U enters the source
    aP = aW + aE
    # interior coupling: cells not touching a wall drop the wall term correctly
    return h, y, aW, aP, aE, b

def gauss_seidel(aW, aP, aE, b, tol=1e-12, itmax=200000, omega=1.6):
    """Solve the tridiagonal system by SOR, monitoring the L2 residual."""
    N = len(b)
    u = np.zeros(N)
    res_hist = []
    r0 = np.linalg.norm(b)
    r0 = r0 if r0 > 0 else 1.0
    for it in range(1, itmax + 1):
        for i in range(N):
            w = u[i-1] if i > 0 else 0.0
            e = u[i+1] if i < N - 1 else 0.0
            u_new = (b[i] + aW[i] * w + aE[i] * e) / aP[i]
            u[i] = (1 - omega) * u[i] + omega * u_new
        # residual r_i = aW w + aE e - aP u + b
        r = np.zeros(N)
        r[1:]  += aW[1:]  * u[:-1]
        r[:-1] += aE[:-1] * u[1:]
        r += b - aP * u
        rn = np.linalg.norm(r) / r0
        res_hist.append(rn)
        if rn < tol:
            break
    return u, np.array(res_hist), it

def tdma(aW, aP, aE, b):
    """Direct Thomas solve of the same system, as an independent check."""
    N = len(b)
    P_ = np.zeros(N); Q_ = np.zeros(N)
    P_[0] = aE[0] / aP[0]; Q_[0] = b[0] / aP[0]
    for i in range(1, N):
        denom = aP[i] - aW[i] * P_[i-1]
        P_[i] = aE[i] / denom
        Q_[i] = (b[i] + aW[i] * Q_[i-1]) / denom
    u = np.zeros(N)
    u[-1] = Q_[-1]
    for i in range(N - 2, -1, -1):
        u[i] = P_[i] * u[i+1] + Q_[i]
    return u

print("Example 1.2  Finite Volume Method -- Couette-Poiseuille flow")
print(f"  P = {P},  dp/dx = {dpdx:.3f} Pa/m\n")

N = 40
h, y, aW, aP, aE, b = assemble(N)
u_gs, res, iters = gauss_seidel(aW, aP, aE, b)
u_td = tdma(aW, aP, aE, b)
ue = u_exact(y)

err_gs = np.max(np.abs(u_gs - ue))
err_td = np.max(np.abs(u_td - ue))
gs_vs_td = np.max(np.abs(u_gs - u_td))

print(f"  N = {N} control volumes, SOR converged in {iters} sweeps "
      f"(final residual {res[-1]:.2e})")
print(f"  max|u_SOR  - u_exact| = {err_gs:.3e} m/s")
print(f"  max|u_TDMA - u_exact| = {err_td:.3e} m/s")
print(f"  max|u_SOR  - u_TDMA | = {gs_vs_td:.3e} m/s")
assert gs_vs_td < 1e-9, "iterative and direct solvers disagree"
print("  Iterative (SOR) and direct (TDMA) solvers agree to round-off.\n")

# Grid-convergence verification against the analytical solution.
print("  Grid convergence against the analytical solution:")
print(f"    {'N':>5} {'h [m]':>12} {'max|err| [m/s]':>16} {'order p':>9}")
prev_e = prev_h = None
errs = []
for Ng in (10, 20, 40, 80, 160, 320):
    hg, yg, aWg, aPg, aEg, bg = assemble(Ng)
    ug = tdma(aWg, aPg, aEg, bg)
    e = np.max(np.abs(ug - u_exact(yg)))
    p = (np.log(prev_e / e) / np.log(2.0)) if prev_e else float("nan")
    print(f"    {Ng:5d} {hg:12.3e} {e:16.4e} {p:9.3f}")
    errs.append((hg, e)); prev_e, prev_h = e, hg
p_final = np.log(errs[-2][1] / errs[-1][1]) / np.log(errs[-2][0] / errs[-1][0])
print(f"\n  Observed order of accuracy on the finest pair: p = {p_final:.3f}")
assert 1.8 < p_final < 2.2, "scheme is not second-order as designed"
print("  PASS: the finite volume scheme converges at the design rate (2).\n")

# figure: solution overlay + residual history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
yf = np.linspace(0, H, 300)
ax1.plot(u_exact(yf) / U, yf / H, "k-", lw=2, label="analytical")
ax1.plot(u_gs / U, y / H, "o", ms=6, mfc="none", mec="C3", label=f"FVM (N={N})")
ax1.set_xlabel(r"$u/U$"); ax1.set_ylabel(r"$y/H$")
ax1.set_title("FVM vs analytical"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
ax2.semilogy(np.arange(1, len(res) + 1), res, "C0-", lw=1.8)
ax2.set_xlabel("SOR sweep"); ax2.set_ylabel(r"normalized $L_2$ residual")
ax2.set_title("Residual convergence"); ax2.grid(alpha=0.3, which="both")
fig.suptitle("Finite volume solution of Couette-Poiseuille flow", y=1.05, fontsize=13)
fig.savefig("fig1_2_fvm.png", dpi=150, bbox_inches="tight")
print("  Wrote fig1_2_fvm.png")
