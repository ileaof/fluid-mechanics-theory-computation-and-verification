#!/usr/bin/env python3
"""
Example 11.3 -- The lid-driven cavity by the vorticity-streamfunction method.

The lid-driven cavity -- a square box whose top wall slides at unit speed -- is the
standard benchmark of incompressible computational fluid dynamics.  This program
solves it in the vorticity-streamfunction formulation, which eliminates the pressure
(and with it the pressure-velocity coupling difficulty that the SIMPLE family of the
text is built to handle) by working with the scalar vorticity omega and stream
function psi:

        Laplacian(psi) = -omega ,      u = d psi/dy ,   v = -d psi/dx ,
        u domega/dx + v domega/dy = nu Laplacian(omega)   (steady vorticity transport).

The stream-function Poisson equation is solved by successive over-relaxation, the wall
vorticity is set by Thom's formula, and the vorticity transport equation is marched to
steady state.  The converged centreline velocity profile is VERIFIED against the
benchmark data of Ghia, Ghia & Shin (1982) at Reynolds number 100.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GHIA_Y = np.array([0.0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,
                   0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0])
GHIA_U = np.array([0.0,-0.03717,-0.04192,-0.04775,-0.06434,-0.10150,-0.15662,
                   -0.21090,-0.20581,-0.13641,0.00332,0.23151,0.68717,0.73722,
                   0.78871,0.84123,1.0])

def solve_cavity(N, Re, maxit=40000, tol=1e-6):
    h = 1.0/N; nu = 1.0/Re
    psi = np.zeros((N+1, N+1)); w = np.zeros((N+1, N+1))
    dt = 0.9*min(h*h/(4*nu), 0.5*h)      # stable pseudo-time step
    res_hist = []
    for it in range(maxit):
        # --- stream-function Poisson: Laplacian(psi) = -w  (SOR) ---
        for _ in range(15):
            psi[1:N,1:N] = ((psi[2:,1:N]+psi[:-2,1:N]+psi[1:N,2:]+psi[1:N,:-2]
                             + h*h*w[1:N,1:N])*0.25)
        # --- velocities from psi ---
        u = np.zeros((N+1,N+1)); v = np.zeros((N+1,N+1))
        u[1:N,1:N] = (psi[1:N,2:]-psi[1:N,:-2])/(2*h)
        v[1:N,1:N] = -(psi[2:,1:N]-psi[:-2,1:N])/(2*h)
        u[:,N] = 1.0                                     # lid velocity (BC)
        # --- wall vorticity (Thom's formula); lid at top moves at U=1 ---
        w[:,N] = -2*psi[:,N-1]/h**2 - 2*1.0/h          # top lid (U=1)
        w[:,0] = -2*psi[:,1]/h**2                        # bottom
        w[0,:] = -2*psi[1,:]/h**2                        # left
        w[N,:] = -2*psi[N-1,:]/h**2                      # right
        # --- vorticity transport: march one step (central differences) ---
        wxx = (w[2:,1:N]-2*w[1:N,1:N]+w[:-2,1:N])/h**2
        wyy = (w[1:N,2:]-2*w[1:N,1:N]+w[1:N,:-2])/h**2
        wx  = (w[2:,1:N]-w[:-2,1:N])/(2*h)
        wy  = (w[1:N,2:]-w[1:N,:-2])/(2*h)
        rhs = nu*(wxx+wyy) - u[1:N,1:N]*wx - v[1:N,1:N]*wy
        w_new = w[1:N,1:N] + dt*rhs
        change = np.max(np.abs(w_new - w[1:N,1:N]))
        w[1:N,1:N] = w_new
        res_hist.append(change)
        if change < tol and it > 100: break
    return psi, w, u, v, np.array(res_hist), it

print("Example 11.3  Lid-driven cavity (vorticity-streamfunction) at Re=100\n")
N, Re = 64, 100.0
t0=time.perf_counter()
psi,w,u,v,res,it = solve_cavity(N, Re)
cpu=time.perf_counter()-t0
print(f"  Grid {N}x{N}, steady state in {it} iterations "
      f"(max |dw| {res[-1]:.2e}), CPU {cpu:.1f} s")

# u along vertical centreline x=0.5
yc = np.linspace(0,1,N+1)
uc = u[N//2,:]
u_interp = np.interp(GHIA_Y, yc, uc)
err = np.max(np.abs(u_interp - GHIA_U))
print(f"\n  Centreline u vs Ghia et al. (1982), Re=100:")
print(f"    {'y':>7} {'u (present)':>12} {'u (Ghia)':>10} {'diff':>9}")
for yq, ug in zip([0.9531,0.8516,0.5,0.2813,0.1016,0.0547],
                  [0.68717,0.23151,-0.20581,-0.15662,-0.06434,-0.03717]):
    us = np.interp(yq, yc, uc)
    print(f"    {yq:7.4f} {us:12.4f} {ug:10.5f} {abs(us-ug):9.4f}")
print(f"\n  Max deviation from Ghia benchmark: {err:.4f}")
assert err < 0.03, "cavity solution does not match Ghia benchmark"
print("  PASS: solution matches the Ghia lid-driven-cavity benchmark.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,4.6),constrained_layout=True)
ax1.plot(uc, yc, "C0-", lw=2, label="present (100x100)")
ax1.plot(GHIA_U, GHIA_Y, "ks", ms=6, mfc="none", label="Ghia et al. 1982")
ax1.set_xlabel("$u$ at $x=0.5$"); ax1.set_ylabel("$y$")
ax1.set_title("Centreline velocity vs benchmark"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
X,Y=np.meshgrid(np.linspace(0,1,N+1),np.linspace(0,1,N+1),indexing="ij")
ax2.contour(X,Y,psi,levels=20,colors="C0",linewidths=0.7)
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_title("Streamlines (Re=100)")
ax2.set_aspect("equal")
fig.suptitle("Lid-driven cavity: verification against the Ghia benchmark", y=1.04, fontsize=13)
fig.savefig("fig11_3_cavity.png", dpi=150, bbox_inches="tight")
print("  Wrote fig11_3_cavity.png")
