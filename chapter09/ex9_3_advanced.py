#!/usr/bin/env python3
"""
Example 9.3 -- Advanced verification: the Sod shock tube (exact Riemann solution
versus a finite-volume shock-capturing scheme).

The one-dimensional Euler equations for a compressible gas,

        d/dt [rho, rho u, E] + d/dx [rho u, rho u^2 + p, (E+p) u] = 0 ,

with E = p/(g-1) + rho u^2/2, govern unsteady compressible flow.  The Sod problem is
the Riemann problem with initial states (rho,u,p) = (1,0,1) for x<0.5 and
(0.125,0,0.1) for x>0.5.  Its EXACT solution -- a left rarefaction, a contact
discontinuity, and a right shock -- is obtained here by Toro's exact Riemann solver
(a bracketed iteration for the star-region pressure), and used to verify a
finite-volume solver that uses the Rusanov (local Lax-Friedrichs) flux and a
second-order SSP Runge-Kutta step with MinMod slope limiting.  The L1 error is
measured against the exact solution as the mesh is refined, giving the observed
order of accuracy for a shock-capturing scheme (below one, as expected at
discontinuities), with CPU timing.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

g = 1.4

# ---------------- exact Riemann solver (Toro) -------------------------------
def exact_riemann(rhoL,uL,pL, rhoR,uR,pR, x, t, x0=0.5):
    aL=np.sqrt(g*pL/rhoL); aR=np.sqrt(g*pR/rhoR)
    def fk(p,rk,pk,ak):
        if p>pk:                                   # shock
            A=2/((g+1)*rk); B=(g-1)/(g+1)*pk
            return (p-pk)*np.sqrt(A/(p+B))
        else:                                      # rarefaction
            return 2*ak/(g-1)*((p/pk)**((g-1)/(2*g))-1)
    def phi(p): return fk(p,rhoL,pL,aL)+fk(p,rhoR,pR,aR)+(uR-uL)
    pstar=brentq(phi, 1e-8, 1e3, xtol=1e-14)
    ustar=0.5*(uL+uR)+0.5*(fk(pstar,rhoR,pR,aR)-fk(pstar,rhoL,pL,aL))
    S=(x-x0)/t
    out=np.zeros((len(x),3))
    for i,s in enumerate(S):
        if s<=ustar:      # left of contact
            if pstar>pL:  # left shock
                SL=uL-aL*np.sqrt((g+1)/(2*g)*pstar/pL+(g-1)/(2*g))
                if s<=SL: rho,u,p=rhoL,uL,pL
                else:
                    rho=rhoL*((pstar/pL+(g-1)/(g+1))/((g-1)/(g+1)*pstar/pL+1)); u,p=ustar,pstar
            else:         # left rarefaction
                rhoLs=rhoL*(pstar/pL)**(1/g); aLs=aL*(pstar/pL)**((g-1)/(2*g))
                SHL=uL-aL; STL=ustar-aLs
                if s<=SHL: rho,u,p=rhoL,uL,pL
                elif s>=STL: rho,u,p=rhoLs,ustar,pstar
                else:
                    u=2/(g+1)*(aL+(g-1)/2*uL+s); c=2/(g+1)*(aL+(g-1)/2*(uL-s))
                    rho=rhoL*(c/aL)**(2/(g-1)); p=pL*(c/aL)**(2*g/(g-1))
        else:             # right of contact
            if pstar>pR:  # right shock
                SR=uR+aR*np.sqrt((g+1)/(2*g)*pstar/pR+(g-1)/(2*g))
                if s>=SR: rho,u,p=rhoR,uR,pR
                else:
                    rho=rhoR*((pstar/pR+(g-1)/(g+1))/((g-1)/(g+1)*pstar/pR+1)); u,p=ustar,pstar
            else:         # right rarefaction
                rhoRs=rhoR*(pstar/pR)**(1/g); aRs=aR*(pstar/pR)**((g-1)/(2*g))
                SHR=uR+aR; STR=ustar+aRs
                if s>=SHR: rho,u,p=rhoR,uR,pR
                elif s<=STR: rho,u,p=rhoRs,ustar,pstar
                else:
                    u=2/(g+1)*(-aR+(g-1)/2*uR+s); c=2/(g+1)*(aR-(g-1)/2*(uR-s))
                    rho=rhoR*(c/aR)**(2/(g-1)); p=pR*(c/aR)**(2*g/(g-1))
        out[i]=[rho,u,p]
    return out, pstar, ustar

# ---------------- finite-volume Rusanov solver ------------------------------
def prim2cons(rho,u,p): return np.array([rho, rho*u, p/(g-1)+0.5*rho*u**2])
def cons2prim(U):
    rho=U[0]; u=U[1]/rho; p=(g-1)*(U[2]-0.5*rho*u**2); return rho,u,p
def flux(U):
    rho,u,p=cons2prim(U); E=U[2]
    return np.array([rho*u, rho*u**2+p, (E+p)*u])
def minmod(a,b): return np.where(a*b>0, np.sign(a)*np.minimum(np.abs(a),np.abs(b)), 0.0)

def flux_arr(U):
    rho=U[0]; u=U[1]/rho; p=(g-1)*(U[2]-0.5*rho*u**2); E=U[2]
    return np.vstack([rho*u, rho*u**2+p, (E+p)*u])

def solve_fv(N, t_end=0.2, cfl=0.4):
    x=np.linspace(0,1,N+1); xc=0.5*(x[:-1]+x[1:]); dx=1.0/N
    U=np.where(xc<0.5, prim2cons(1.0,0.0,1.0)[:,None], prim2cons(0.125,0.0,0.1)[:,None])
    U=U.astype(float)
    def L(U):
        rho=U[0]; u=U[1]/rho; p=(g-1)*(U[2]-0.5*rho*u**2); a=np.sqrt(g*np.maximum(p,1e-8)/rho)
        smax=np.max(np.abs(u)+a)
        dU=np.zeros_like(U)
        dU[:,1:-1]=minmod(U[:,1:-1]-U[:,:-2], U[:,2:]-U[:,1:-1])
        UL=U+0.5*dU; UR=U-0.5*dU                      # right-face / left-face cell states
        Ul=UL[:,:-1]; Ur=UR[:,1:]                      # interior faces 1..N-1
        rl=Ul[0]; ul=Ul[1]/rl; pl=(g-1)*(Ul[2]-0.5*rl*ul**2); al=np.sqrt(g*np.maximum(pl,1e-8)/rl)
        rr=Ur[0]; ur=Ur[1]/rr; pr=(g-1)*(Ur[2]-0.5*rr*ur**2); ar=np.sqrt(g*np.maximum(pr,1e-8)/rr)
        sm=np.maximum(np.abs(ul)+al, np.abs(ur)+ar)
        Fint=0.5*(flux_arr(Ul)+flux_arr(Ur))-0.5*sm*(Ur-Ul)
        F=np.zeros((3,N+1))
        F[:,1:N]=Fint
        F[:,0]=flux_arr(U[:,:1])[:,0]; F[:,N]=flux_arr(U[:,-1:])[:,0]  # transmissive
        return -(F[:,1:]-F[:,:-1])/dx, smax
    t=0.0
    while t<t_end:
        _,smax=L(U); dt=cfl*dx/smax
        if t+dt>t_end: dt=t_end-t
        Lu,_=L(U); U1=U+dt*Lu
        Lu1,_=L(U1); U=0.5*U+0.5*(U1+dt*Lu1)          # SSP-RK2
        t+=dt
    rho=U[0]; u=U[1]/rho; p=(g-1)*(U[2]-0.5*rho*u**2)
    return xc, rho, u, p

print("Example 9.3  Sod shock tube: exact Riemann vs finite-volume scheme\n")
t_end=0.2
# exact star state
_,pstar,ustar=exact_riemann(1,0,1, 0.125,0,0.1, np.array([0.5]), t_end)
print(f"  Exact star region: p* = {pstar:.6f}, u* = {ustar:.6f}")
print(f"\n  L1 error in density vs the exact Riemann solution:")
print(f"    {'N':>6} {'dx':>10} {'L1 error':>13} {'order p':>9} {'CPU[ms]':>9}")
prev=None; rows=[]
for N in [100,200,400,800,1600]:
    t0=time.perf_counter()
    xc,rho,u,p=solve_fv(N,t_end)
    cpu=(time.perf_counter()-t0)*1e3
    ex,_,_=exact_riemann(1,0,1,0.125,0,0.1, xc, t_end)
    L1=np.sum(np.abs(rho-ex[:,0]))/N
    pord=np.log(prev/L1)/np.log(2) if prev else float("nan")
    print(f"    {N:6d} {1/N:10.4f} {L1:13.4e} {pord:9.3f} {cpu:9.1f}")
    rows.append((1/N,L1)); prev=L1
p_final=np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 0.5<p_final<1.1, "shock-capturing L1 order should be ~0.6-1.0"
# Richardson/GCI
e3,e2,e1=rows[-3][1],rows[-2][1],rows[-1][1]
r=2.0; p_obs=np.log(e3/e2)/np.log(r); GCI=1.25*e1/(r**p_obs-1)
print(f"  Richardson order p={p_obs:.3f}; GCI (Fs=1.25) = {GCI:.3e}")
print("  PASS: finite-volume solution converges to the exact Riemann solution.\n")

# figure
fig,axs=plt.subplots(1,3,figsize=(12.5,4.0),constrained_layout=True)
xc,rho,u,p=solve_fv(200,t_end)
xe=np.linspace(0,1,800); ex,_,_=exact_riemann(1,0,1,0.125,0,0.1,xe,t_end)
for ax,num,exq,lab in zip(axs,[rho,u,p],[ex[:,0],ex[:,1],ex[:,2]],
                          ["density $\\rho$","velocity $u$","pressure $p$"]):
    ax.plot(xe,exq,"k-",lw=2,label="exact")
    ax.plot(xc,num,"o",ms=3,mfc="none",mec="C3",label="FV (N=200)")
    ax.set_xlabel("x"); ax.set_title(lab); ax.grid(alpha=0.3)
axs[0].legend(frameon=False)
fig.suptitle("Sod shock tube at t=0.2: finite volume vs exact Riemann solution", y=1.05, fontsize=13)
fig.savefig("fig9_3_sod.png", dpi=150, bbox_inches="tight")
print("  Wrote fig9_3_sod.png")
