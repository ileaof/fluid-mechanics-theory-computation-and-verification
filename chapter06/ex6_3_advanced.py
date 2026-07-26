#!/usr/bin/env python3
"""
Example 6.3 -- Advanced verification: dynamic similarity of laminar channel flow,
computed by the finite volume method.

Fully developed laminar flow between parallel plates (plane Poiseuille flow) has a
famous dimensionless invariant: the product of the Darcy friction factor and the
Reynolds number is exactly

        f * Re = 96          (Re based on hydraulic diameter D_h = 2H).

This number depends on NOTHING dimensional -- not the gap, the fluid, the pressure
gradient, nor the speed -- only on the geometry.  It is therefore a perfect test of
two ideas at once: dynamic SIMILARITY (all physical scales give the same f*Re) and
numerical VERIFICATION (the computed value converges to 96).

The program solves the momentum equation mu d^2u/dy^2 = dp/dx by a cell-centred
finite volume method, computes f and Re from the discrete solution, and (i) shows
that three wildly different physical scales at the same Reynolds number give an
identical f*Re, and (ii) refines the mesh to confirm second-order convergence to
96, with Richardson extrapolation, a Grid Convergence Index, and CPU timing.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def solve_channel(N, H, mu, dpdx, rho):
    """Cell-centred FVM for mu u'' = dp/dx, u(0)=u(H)=0. Returns f, Re, (y,u)."""
    dy = H/N
    yc = (np.arange(N)+0.5)*dy
    aW = np.full(N, mu/dy); aE = np.full(N, mu/dy)
    aW[0]  = mu/(dy/2); aE[-1] = mu/(dy/2)      # walls at y=0,H (u=0)
    aP = aW+aE
    b = np.full(N, -dpdx*dy)                    # source S=-dp/dx per cell
    # Thomas solve
    P=np.zeros(N); Q=np.zeros(N)
    P[0]=aE[0]/aP[0]; Q[0]=b[0]/aP[0]
    for i in range(1,N):
        d=aP[i]-aW[i]*P[i-1]; P[i]=aE[i]/d; Q[i]=(b[i]+aW[i]*Q[i-1])/d
    u=np.zeros(N); u[-1]=Q[-1]
    for i in range(N-2,-1,-1): u[i]=P[i]*u[i+1]+Q[i]
    # bulk velocity by midpoint average (2nd order); friction factor from the
    # known pressure gradient (exact force balance for fully developed flow)
    V = np.sum(u)*dy/H
    Dh = 2*H
    f  = (-dpdx)*Dh/(0.5*rho*V**2)      # Darcy-Weisbach definition
    Re = rho*V*Dh/mu
    return f, Re, f*Re, (yc,u)

print("Example 6.3  Dynamic similarity of laminar channel flow (FVM)\n")
print("  Exact invariant: f*Re = 96 (plane Poiseuille, D_h = 2H)\n")

# (i) dynamic similarity across physical scales, same Re, fixed mesh
print("  Similarity check (N=200 cells) across very different scales:")
print(f"    {'case':16s} {'H[m]':>8} {'mu':>8} {'V[m/s]':>8} {'Re':>9} {'f*Re':>9}")
scales = [("micro-channel", 1e-4, 1.0e-3, None),
          ("lab pipe",      1e-2, 1.0e-3, None),
          ("large duct",    2e-1, 5.0e-2, None)]
Re_t = 500.0
frRe=[]
for name,H,mu,_ in scales:
    rho=1000.
    # choose dp/dx to reach Re_t: V=Re_t*mu/(rho*2H); and V=(-dpdx)H^2/(12mu) -> solve dpdx
    V_target=Re_t*mu/(rho*2*H)
    dpdx=-V_target*12*mu/H**2
    f,Re,fR,_=solve_channel(200,H,mu,dpdx,rho)
    frRe.append(fR)
    print(f"    {name:16s} {H:8.1e} {mu:8.1e} {V_target:8.4f} {Re:9.1f} {fR:9.5f}")
print(f"  Spread in f*Re across scales: {max(frRe)-min(frRe):.2e}  (dynamic similarity)\n")

# (ii) grid convergence to 96
print("  Grid convergence of f*Re to the exact value 96:")
print(f"    {'N':>6} {'f*Re':>12} {'error':>12} {'order p':>9} {'CPU[ms]':>9}")
H,mu,rho=1e-2,1e-3,1000.
V_t=Re_t*mu/(rho*2*H); dpdx=-V_t*12*mu/H**2
prev=None; rows=[]
for N in [10,20,40,80,160,320]:
    t0=time.perf_counter()
    f,Re,fR,_=solve_channel(N,H,mu,dpdx,rho)
    cpu=(time.perf_counter()-t0)*1e3
    err=abs(fR-96.0)
    p=np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"    {N:6d} {fR:12.6f} {err:12.4e} {p:9.3f} {cpu:9.3f}")
    rows.append((H/N,fR,err)); prev=err
p_final=np.log(rows[-2][2]/rows[-1][2])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy: p = {p_final:.3f}")
assert 1.8<p_final<2.2, "channel f*Re not converging at 2nd order"

# Richardson + GCI on f*Re
v3,v2,v1=rows[-3][1],rows[-2][1],rows[-1][1]
r=2.0; p_obs=np.log(abs((v3-v2)/(v2-v1)))/np.log(r)
v_ext=v1+(v1-v2)/(r**p_obs-1); Fs=1.25
GCI=Fs*abs((v1-v2)/v1)/(r**p_obs-1)
print("  Richardson extrapolation of f*Re:")
print(f"    extrapolated (h->0) = {v_ext:.6f};  exact = 96")
print(f"    |extrap - 96| = {abs(v_ext-96):.2e};  GCI_fine (Fs={Fs}) = {GCI*100:.4f} %")
assert abs(v_ext-96)<1e-2, "Richardson estimate inconsistent with 96"
print("  PASS: similarity holds across scales; f*Re -> 96 at 2nd order.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
f,Re,fR,(yc,u)=solve_channel(24,H,mu,dpdx,rho)
yf=np.linspace(0,H,200)
u_ex=(-dpdx)/(2*mu)*(H*yf-yf**2)
ax1.plot(u_ex/np.max(u_ex),yf/H,"k-",lw=2,label="analytical")
ax1.plot(u/np.max(u_ex),yc/H,"o",ms=5,mfc="none",mec="C3",label="FVM (N=24)")
ax1.set_xlabel("$u/u_{max}$"); ax1.set_ylabel("$y/H$")
ax1.set_title("Plane Poiseuille profile"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
hs=np.array([r_[0] for r_ in rows]); es=np.array([r_[2] for r_ in rows])
ax2.loglog(hs,es,"o-",lw=1.8,label=r"$|f\,\mathrm{Re}-96|$")
ax2.loglog(hs,es[-1]*(hs/hs[-1])**2,"k:",lw=1.4,label="slope 2")
ax2.set_xlabel("mesh spacing $\\Delta y$ [m]"); ax2.set_ylabel("error in $f\\,$Re")
ax2.set_title("Convergence to $f\\,$Re $=96$"); ax2.legend(frameon=False)
ax2.grid(alpha=0.3,which="both")
fig.suptitle("Dynamic similarity and verification of laminar channel flow",y=1.04,fontsize=13)
fig.savefig("fig6_3_similarity.png",dpi=150,bbox_inches="tight")
print("  Wrote fig6_3_similarity.png")
