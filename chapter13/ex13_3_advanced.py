#!/usr/bin/env python3
"""
Example 13.3 -- The dam-break problem: shallow-water flow with an exact solution.

When a dam separating deep water (depth h_L) from shallow water (depth h_R) is
suddenly removed, a flood wave races downstream and a rarefaction spreads upstream.
The flow is governed by the shallow-water equations, a hyperbolic system analogous
to the compressible Euler equations of Chapter 9,

        dh/dt + d(h u)/dx = 0 ,
        d(h u)/dt + d(h u^2 + g h^2/2)/dx = 0 .

The dam break is their Riemann problem, and it has an EXACT solution (Ritter/Stoker):
a left rarefaction fan, a constant intermediate state, and a right shock (the flood
bore).  The intermediate depth h* is found from the matching condition

        2( sqrt(g h_L) - sqrt(g h*) ) = (h* - h_R) sqrt( g(h*+h_R)/(2 h* h_R) ) ,

a single nonlinear equation solved here by a bracketed root finder.  This exact
solution verifies a finite-volume shock-capturing scheme (Rusanov flux, second-order
limited reconstruction), whose L1 error is measured as the mesh is refined.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

g = 9.81
hL, hR = 1.0, 0.1        # left (deep) and right (shallow) depths; u=0 initially
x0 = 0.0                  # dam location

def exact(x, t):
    """Exact Ritter/Stoker dam-break solution at time t; returns h(x), u(x)."""
    cL = np.sqrt(g*hL)
    # solve for intermediate depth h*
    def match(hs):
        return 2*(cL - np.sqrt(g*hs)) - (hs-hR)*np.sqrt(g*(hs+hR)/(2*hs*hR))
    hstar = brentq(match, hR, hL, xtol=1e-13)
    us = 2*(cL - np.sqrt(g*hstar))
    cs = np.sqrt(g*hstar)
    # shock speed (right-going bore)
    S = us*hstar/(hstar-hR)
    h = np.empty_like(x); u = np.empty_like(x)
    xi = (x-x0)/t
    for i,s in enumerate(xi):
        if s <= -cL:                       # undisturbed left
            h[i], u[i] = hL, 0.0
        elif s <= us-cs:                   # rarefaction fan
            u[i] = (2.0/3.0)*(cL + s)
            c = (cL - s)/3.0 + s*0 + ( (2*cL - s)/3.0 - (2.0/3.0)*(cL+s) )*0
            c = (2*cL - s)/3.0
            h[i] = c*c/g
        elif s <= S:                       # constant intermediate state
            h[i], u[i] = hstar, us
        else:                              # undisturbed right
            h[i], u[i] = hR, 0.0
    return h, u

def flux(U):
    h=U[0]; hu=U[1]; u=hu/h
    return np.array([hu, hu*u + 0.5*g*h*h])

def minmod(a,b): return np.where(a*b>0, np.sign(a)*np.minimum(np.abs(a),np.abs(b)),0.0)

def solve_fv(N, t_end, cfl=0.4, Lx=(-5.0,5.0)):
    x=np.linspace(Lx[0],Lx[1],N+1); xc=0.5*(x[:-1]+x[1:]); dx=(Lx[1]-Lx[0])/N
    U=np.zeros((2,N))
    U[0]=np.where(xc<x0, hL, hR); U[1]=0.0
    def L(U):
        h=U[0]; u=U[1]/h; c=np.sqrt(g*h); smax=np.max(np.abs(u)+c)
        dU=np.zeros_like(U); dU[:,1:-1]=minmod(U[:,1:-1]-U[:,:-2],U[:,2:]-U[:,1:-1])
        UL=U+0.5*dU; UR=U-0.5*dU
        Ul=UL[:,:-1]; Ur=UR[:,1:]
        hl=Ul[0]; ul=Ul[1]/hl; cl=np.sqrt(g*hl)
        hr=Ur[0]; ur=Ur[1]/hr; cr=np.sqrt(g*hr)
        sm=np.maximum(np.abs(ul)+cl,np.abs(ur)+cr)
        Fl=np.vstack([Ul[1], Ul[1]**2/hl+0.5*g*hl*hl])
        Fr=np.vstack([Ur[1], Ur[1]**2/hr+0.5*g*hr*hr])
        Fint=0.5*(Fl+Fr)-0.5*sm*(Ur-Ul)
        F=np.zeros((2,N+1)); F[:,1:N]=Fint
        F[:,0]=np.array([U[1,0], U[1,0]**2/U[0,0]+0.5*g*U[0,0]**2])
        F[:,N]=np.array([U[1,-1], U[1,-1]**2/U[0,-1]+0.5*g*U[0,-1]**2])
        return -(F[:,1:]-F[:,:-1])/dx, smax
    t=0.0
    while t<t_end:
        _,smax=L(U); dt=cfl*dx/smax
        if t+dt>t_end: dt=t_end-t
        Lu,_=L(U); U1=U+dt*Lu
        Lu1,_=L(U1); U=0.5*U+0.5*(U1+dt*Lu1)
        t+=dt
    return xc, U[0], U[1]/U[0]

print("Example 13.3  Dam-break shallow-water flow vs exact Riemann solution\n")
t_end=0.5
he,ue=exact(np.array([x0]),t_end)
print(f"  Depths: h_L={hL}, h_R={hR}; solved at t={t_end} s\n")
print(f"  L1 error in depth h vs the exact dam-break solution:")
print(f"    {'N':>6} {'dx':>10} {'L1 error':>13} {'order p':>9} {'CPU[ms]':>9}")
prev=None; rows=[]
for N in [100,200,400,800,1600]:
    t0=time.perf_counter(); xc,h,u=solve_fv(N,t_end); cpu=(time.perf_counter()-t0)*1e3
    hex,_=exact(xc,t_end)
    L1=np.sum(np.abs(h-hex))/N
    p=np.log(prev/L1)/np.log(2) if prev else float("nan")
    print(f"    {N:6d} {10.0/N:10.4f} {L1:13.4e} {p:9.3f} {cpu:9.1f}")
    rows.append((10.0/N,L1)); prev=L1
p_final=np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 0.5<p_final<1.1, "shock-capturing L1 order should be ~0.7-1.0"
e3,e2,e1=rows[-3][1],rows[-2][1],rows[-1][1]
r=2.0; p_obs=np.log(e3/e2)/np.log(r); GCI=1.25*e1/(r**p_obs-1)
print(f"  Richardson order p={p_obs:.3f}; GCI (Fs=1.25) = {GCI:.3e}")
print("  PASS: finite-volume solution converges to the exact dam-break solution.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,4.4),constrained_layout=True)
xc,h,u=solve_fv(200,t_end)
xe=np.linspace(-5,5,800); hex,uex=exact(xe,t_end)
ax1.plot(xe,hex,"k-",lw=2,label="exact"); ax1.plot(xc,h,"o",ms=3,mfc="none",mec="C0",label="FV (N=200)")
ax1.fill_between(xe,0,hex,color="C0",alpha=0.12)
ax1.set_xlabel("x"); ax1.set_ylabel("depth $h$"); ax1.set_title("Water depth at t=0.5 s")
ax1.legend(frameon=False); ax1.grid(alpha=0.3)
ax2.plot(xe,uex,"k-",lw=2,label="exact"); ax2.plot(xc,u,"o",ms=3,mfc="none",mec="C3",label="FV")
ax2.set_xlabel("x"); ax2.set_ylabel("velocity $u$"); ax2.set_title("Velocity (rarefaction + bore)")
ax2.legend(frameon=False); ax2.grid(alpha=0.3)
fig.suptitle("Dam-break shallow-water flow: verification against the exact solution", y=1.04, fontsize=13)
fig.savefig("fig13_3_dambreak.png", dpi=150, bbox_inches="tight")
print("  Wrote fig13_3_dambreak.png")
