#!/usr/bin/env python3
"""
Example 5.2 -- Finite-volume solution of Stokes' first problem.

The same unsteady diffusion equation solved analytically in Example 5.1,

        du/dt = nu d^2 u / dy^2 ,   u(0,t)=U0 , u(L,t)=0 , u(y,0)=0 ,

is now solved on a finite domain [0, L] with a cell-centred finite-volume method
in space and the CRANK-NICOLSON scheme in time (second order in both).  Each step
requires the solution of a tridiagonal system, done here with the Thomas
algorithm.  The domain is taken long enough that the diffusing layer never reaches
the far boundary during the simulated interval, so the finite-domain solution
matches the semi-infinite erfc solution of Example 5.1.  A mesh-refinement study
(refining space and time together) confirms second-order convergence.
Uses numpy, matplotlib, and scipy.special.erfc (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

U0 = 1.0
nu = 1.0e-6
L  = 0.12          # domain height [m]; erfc(L/2sqrt(nu t_end)) < 1e-16 so u(L)=0 is exact
t_end = 100.0

def u_exact(y, t):
    return U0 * erfc(y / (2.0*np.sqrt(nu*t)))

def solve_cn(N, nt):
    """Crank-Nicolson finite-volume solve; N cells, nt time steps."""
    dy = L/N
    yc = (np.arange(N)+0.5)*dy
    dt = t_end/nt
    r = nu*dt/dy**2
    u = np.zeros(N)                      # initial condition u=0
    # Dirichlet: u(0)=U0 via half-cell at bottom, u(L)=0 at top
    aWb = nu/(dy/2)                      # bottom wall conductance (to U0)
    aWi = nu/dy                          # interior face conductance
    for _ in range(nt):
        # Crank-Nicolson: (I - dt/2 A) u^{n+1} = (I + dt/2 A) u^n + dt*bc
        lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N); rhs = np.zeros(N)
        for i in range(N):
            aW = aWb if i == 0 else aWi
            aE = aWi if i < N-1 else nu/(dy/2)   # top wall (u=0)
            # diffusion operator L u = (aW u_{i-1} - (aW+aE) u_i + aE u_{i+1})/dy
            cW = aW/dy; cE = aE/dy; cP = (aW+aE)/dy
            di[i] = 1 + 0.5*dt*cP
            if i > 0:  lo[i] = -0.5*dt*cW
            if i < N-1: up[i] = -0.5*dt*cE
            # explicit half
            uW = u[i-1] if i > 0 else 0.0
            uE = u[i+1] if i < N-1 else 0.0
            expl = u[i] + 0.5*dt*(cW*uW - cP*u[i] + cE*uE)
            rhs[i] = expl
            # boundary contributions (u(0)=U0 both halves)
            if i == 0:
                rhs[i] += 0.5*dt*cW*U0     # explicit half
                rhs[i] += 0.5*dt*cW*U0     # implicit half moved to RHS
            # top boundary u=0 contributes nothing
        # Thomas solve
        for i in range(1, N):
            m = lo[i]/di[i-1]; di[i] -= m*up[i-1]; rhs[i] -= m*rhs[i-1]
        u[-1] = rhs[-1]/di[-1]
        for i in range(N-2, -1, -1):
            u[i] = (rhs[i] - up[i]*u[i+1])/di[i]
    return yc, u

print("Example 5.2  Finite-volume (Crank-Nicolson) solution of Stokes' problem\n")
print(f"  Domain L={L} m, t_end={t_end} s, nu={nu:.1e} m^2/s")
print(f"  Boundary-layer thickness at t_end ~ {3.64*np.sqrt(nu*t_end)*1e3:.1f} mm "
      f"(<< L={L*1e3:.0f} mm)\n")

print(f"  {'N':>5} {'nt':>6} {'max|err| [m/s]':>16} {'order p':>9}")
prev = None; errs = []
for N, nt in [(20,50),(40,100),(80,200),(160,400),(320,800)]:
    yc, u = solve_cn(N, nt)
    err = np.max(np.abs(u - u_exact(yc, t_end)))
    p = np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"  {N:5d} {nt:6d} {err:16.4e} {p:9.3f}")
    errs.append((L/N, err)); prev = err

p_final = np.log(errs[-2][1]/errs[-1][1])/np.log(errs[-2][0]/errs[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 1.8 < p_final < 2.2, "Crank-Nicolson FV solve is not second order"
print("  PASS: finite-volume solution converges at 2nd order to the erfc solution.\n")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
yc, u = solve_cn(40, 100)
yf = np.linspace(0, L, 300)
ax1.plot(u_exact(yf, t_end)/U0, yf*1e3, "k-", lw=2, label="analytical erfc")
ax1.plot(u/U0, yc*1e3, "o", ms=5, mfc="none", mec="C3", label="FVM (N=40)")
ax1.set_ylim(0, 45); ax1.set_xlabel("$u/U_0$"); ax1.set_ylabel("$y$ [mm]")
ax1.set_title(f"FVM vs analytical at t={t_end:.0f} s")
ax1.legend(frameon=False); ax1.grid(alpha=0.3)
hs = np.array([e[0] for e in errs]); es = np.array([e[1] for e in errs])
ax2.loglog(hs, es, "o-", lw=1.8, label="max error")
ax2.loglog(hs, es[-1]*(hs/hs[-1])**2, "k:", lw=1.4, label="slope 2 (reference)")
ax2.set_xlabel("mesh spacing $\\Delta y$ [m]"); ax2.set_ylabel("max error [m/s]")
ax2.set_title("Space-time convergence"); ax2.legend(frameon=False)
ax2.grid(alpha=0.3, which="both")
fig.suptitle("Crank-Nicolson finite-volume solution of Stokes' first problem",
             y=1.04, fontsize=13)
fig.savefig("fig5_2_fvm.png", dpi=150, bbox_inches="tight")
print("  Wrote fig5_2_fvm.png")
