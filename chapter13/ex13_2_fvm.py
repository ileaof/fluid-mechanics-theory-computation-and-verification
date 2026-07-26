#!/usr/bin/env python3
"""
Example 13.2 -- Natural convection in a differentially heated square cavity
(the de Vahl Davis benchmark).

A square cavity has its left wall hot (T=1) and its right wall cold (T=0), with
insulated top and bottom.  Buoyancy drives a circulating flow that carries heat from
the hot to the cold wall.  In the Boussinesq approximation the non-dimensional
governing equations, in the vorticity-streamfunction form, are

  d omega/dt + u d omega/dx + v d omega/dy = Pr (Laplacian omega) + Ra Pr dT/dx ,
  d T/dt     + u dT/dx      + v dT/dy      = Laplacian T ,
  Laplacian psi = -omega ,   u = d psi/dy , v = -d psi/dx ,

with Rayleigh number Ra, Prandtl number Pr, lengths scaled by the cavity side and
time by L^2/kappa.  The program marches these to steady state and VERIFIES the
average Nusselt number on the hot wall against the benchmark solution of de Vahl
Davis (1983): Nu = 2.238 at Ra = 1e4, Pr = 0.71.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def solve(N, Ra, Pr=0.71, maxit=200000, tol=2e-6):
    h = 1.0/N
    psi = np.zeros((N+1,N+1)); w = np.zeros((N+1,N+1))
    T = np.zeros((N+1,N+1))
    x = np.linspace(0,1,N+1)
    T[:, :] = (1.0 - x)[:,None]                 # initial linear temperature (x hot->cold)
    T[0,:] = 1.0; T[N,:] = 0.0                    # hot left, cold right
    dt = 0.2*min(h*h/(4*max(Pr,1.0)), h*h/4)
    for it in range(maxit):
        # stream-function Poisson (SOR)
        for _ in range(20):
            psi[1:N,1:N] = 0.25*(psi[2:,1:N]+psi[:-2,1:N]+psi[1:N,2:]+psi[1:N,:-2]
                                 + h*h*w[1:N,1:N])
        u = np.zeros((N+1,N+1)); v = np.zeros((N+1,N+1))
        u[1:N,1:N] = (psi[1:N,2:]-psi[1:N,:-2])/(2*h)
        v[1:N,1:N] = -(psi[2:,1:N]-psi[:-2,1:N])/(2*h)
        # temperature: adiabatic top/bottom (zero gradient), Dirichlet left/right
        T[:,0] = T[:,1]; T[:,N] = T[:,N-1]
        Told = T.copy()
        Txx=(Told[2:,1:N]-2*Told[1:N,1:N]+Told[:-2,1:N])/h**2
        Tyy=(Told[1:N,2:]-2*Told[1:N,1:N]+Told[1:N,:-2])/h**2
        Tx=(Told[2:,1:N]-Told[:-2,1:N])/(2*h); Ty=(Told[1:N,2:]-Told[1:N,:-2])/(2*h)
        T[1:N,1:N] = Told[1:N,1:N] + dt*((Txx+Tyy) - u[1:N,1:N]*Tx - v[1:N,1:N]*Ty)
        T[0,:]=1.0; T[N,:]=0.0
        # wall vorticity (Thom)
        w[0,:]=-2*psi[1,:]/h**2; w[N,:]=-2*psi[N-1,:]/h**2
        w[:,0]=-2*psi[:,1]/h**2; w[:,N]=-2*psi[:,N-1]/h**2
        # vorticity transport with buoyancy source Ra*Pr*dT/dx
        wxx=(w[2:,1:N]-2*w[1:N,1:N]+w[:-2,1:N])/h**2
        wyy=(w[1:N,2:]-2*w[1:N,1:N]+w[1:N,:-2])/h**2
        wx=(w[2:,1:N]-w[:-2,1:N])/(2*h); wy=(w[1:N,2:]-w[1:N,:-2])/(2*h)
        dTdx=(T[2:,1:N]-T[:-2,1:N])/(2*h)
        w_new = w[1:N,1:N] + dt*(Pr*(wxx+wyy) - u[1:N,1:N]*wx - v[1:N,1:N]*wy
                                 + Ra*Pr*dTdx)
        change = np.max(np.abs(w_new-w[1:N,1:N]))
        w[1:N,1:N]=w_new
        if change < tol and it>200: break
    # average Nusselt number on the hot wall (x=0): Nu = integral -dT/dx dy
    dTdx_wall = (-3*T[0,:]+4*T[1,:]-T[2,:])/(2*h)   # 2nd-order one-sided
    trapz = np.trapezoid if hasattr(np,"trapezoid") else np.trapz
    Nu = trapz(-dTdx_wall, np.linspace(0,1,N+1))
    return psi,w,T,u,v,Nu,it

print("Example 13.2  Natural convection cavity (de Vahl Davis benchmark)\n")
N, Ra = 60, 1.0e4
t0=time.perf_counter()
psi,w,T,u,v,Nu,it=solve(N,Ra)
cpu=time.perf_counter()-t0
Nu_bench = 2.238
print(f"  Grid {N}x{N}, Ra={Ra:.0e}, Pr=0.71: steady in {it} iterations, CPU {cpu:.1f} s")
print(f"\n  Average Nusselt number on the hot wall:")
print(f"    present      Nu = {Nu:.4f}")
print(f"    de Vahl Davis   = {Nu_bench:.4f}  (benchmark, Ra=1e4)")
print(f"    difference      = {abs(Nu-Nu_bench)/Nu_bench*100:.2f} %")
assert abs(Nu-Nu_bench)/Nu_bench < 0.04, "Nu does not match the benchmark"
print("  PASS: cavity Nusselt number matches the de Vahl Davis benchmark.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.6),constrained_layout=True)
X,Y=np.meshgrid(np.linspace(0,1,N+1),np.linspace(0,1,N+1),indexing="ij")
cf=ax1.contourf(X,Y,T,levels=20,cmap="RdBu_r")
ax1.contour(X,Y,psi,levels=12,colors="k",linewidths=0.5)
ax1.set_title(f"Temperature + streamlines (Ra={Ra:.0e})")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_aspect("equal")
fig.colorbar(cf,ax=ax1,shrink=0.85,label="T")
# vertical velocity profile at mid-height (classic benchmark plot)
yc=np.linspace(0,1,N+1)
ax2.plot(v[:,N//2], np.linspace(0,1,N+1), "C0-", lw=2)
ax2.set_xlabel("$v$ at $y=0.5$"); ax2.set_ylabel("x")
ax2.set_title("Vertical velocity at mid-height"); ax2.grid(alpha=0.3)
fig.suptitle("Buoyancy-driven cavity: the de Vahl Davis benchmark", y=1.04, fontsize=13)
fig.savefig("fig13_2_cavity.png", dpi=150, bbox_inches="tight")
print("  Wrote fig13_2_cavity.png")
