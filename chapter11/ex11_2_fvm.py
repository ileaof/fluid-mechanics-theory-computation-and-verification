#!/usr/bin/env python3
"""
Example 11.2 -- Finite-volume convection-diffusion: central vs upwind schemes.

The convection-diffusion equation of Example 11.1 is discretised by the finite
volume method on N cells.  Integrating over a control volume gives, for constant
convective flux F = rho u and diffusive conductance D = Gamma/dx, a tridiagonal
system whose coefficients depend on the CONVECTION SCHEME:

  * CENTRAL differencing:  a_E = D - F/2 ,  a_W = D + F/2
        second-order accurate, but a_E < 0 when the cell Peclet number
        Pe_cell = F/D = rho u dx/Gamma exceeds 2, producing unphysical
        oscillations (loss of boundedness).
  * UPWIND differencing:   a_E = D + max(-F,0) ,  a_W = D + max(F,0)
        unconditionally bounded, but only first-order accurate, introducing
        artificial "false diffusion".

The program demonstrates the boundedness failure of central differencing at high
cell Peclet number, and measures the observed order of accuracy of both schemes by
grid refinement against the exact solution -- upwind first order, central second.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, rho, u = 1.0, 1.0, 1.0

def phi_exact(x, Pe):
    return (np.exp(Pe*x/L)-1.0)/(np.exp(Pe)-1.0)

def solve(N, Pe, scheme):
    Gamma = rho*u*L/Pe
    dx = L/N
    xc = (np.arange(N)+0.5)*dx
    F = rho*u
    D = Gamma/dx
    aW = np.zeros(N); aE = np.zeros(N)
    for i in range(N):
        if scheme == "central":
            aW[i] = D + F/2; aE[i] = D - F/2
        else:  # upwind
            aW[i] = D + max(F, 0.0); aE[i] = D + max(-F, 0.0)
    # boundary faces (half cell): Dirichlet phi(0)=0, phi(L)=1
    Db = Gamma/(dx/2)
    aWb = Db + max(F,0.0) if scheme=="upwind" else Db + F/2
    aEb = Db + max(-F,0.0) if scheme=="upwind" else Db - F/2
    aP = np.zeros(N); b = np.zeros(N)
    for i in range(N):
        awi = aW[i]; aei = aE[i]
        if i==0:   awi = aWb; b[i] += awi*0.0     # phi(0)=0
        if i==N-1: aei = aEb; b[i] += aei*1.0     # phi(L)=1
        aP[i] = (aWb if i==0 else aW[i]) + (aEb if i==N-1 else aE[i])
        aW[i]=awi; aE[i]=aei
    # Thomas
    P=np.zeros(N); Q=np.zeros(N)
    P[0]=aE[0]/aP[0]; Q[0]=b[0]/aP[0]
    for i in range(1,N):
        d=aP[i]-aW[i]*P[i-1]; P[i]=aE[i]/d; Q[i]=(b[i]+aW[i]*Q[i-1])/d
    phi=np.zeros(N); phi[-1]=Q[-1]
    for i in range(N-2,-1,-1): phi[i]=P[i]*phi[i+1]+Q[i]
    return xc, phi

print("Example 11.2  FVM convection-diffusion: central vs upwind\n")

# --- boundedness demonstration at high cell Peclet number -------------------
Pe, N = 50.0, 10
xc_c, phi_c = solve(N, Pe, "central")
xc_u, phi_u = solve(N, Pe, "upwind")
print(f"  Boundedness at Pe={Pe}, N={N} (Pe_cell={Pe/N:.1f} > 2):")
print(f"    central: min(phi) = {phi_c.min():+.4f}  (negative -> oscillatory, UNBOUNDED)")
print(f"    upwind : min(phi) = {phi_u.min():+.4f}  (bounded)")
assert phi_c.min() < -1e-3, "central should oscillate here"
assert phi_u.min() > -1e-9, "upwind should stay bounded"
print("  -> central differencing loses boundedness for Pe_cell > 2.\n")

# --- grid convergence at moderate Pe (central stable) -----------------------
Pe = 10.0
print(f"  Grid convergence at Pe={Pe} (central stable):")
print(f"    {'N':>5} {'central err':>13} {'p_c':>6} {'upwind err':>13} {'p_u':>6}")
prevc=prevu=None; rc=[]; ru=[]
for N in [20,40,80,160,320]:
    xc,pc = solve(N,Pe,"central"); _,pu = solve(N,Pe,"upwind")
    ec = np.max(np.abs(pc-phi_exact(xc,Pe)))
    eu = np.max(np.abs(pu-phi_exact(xc,Pe)))
    pcord = np.log(prevc/ec)/np.log(2) if prevc else float("nan")
    puord = np.log(prevu/eu)/np.log(2) if prevu else float("nan")
    print(f"    {N:5d} {ec:13.4e} {pcord:6.2f} {eu:13.4e} {puord:6.2f}")
    rc.append((L/N,ec)); ru.append((L/N,eu)); prevc,prevu=ec,eu
pc_final=np.log(rc[-2][1]/rc[-1][1])/np.log(2)
pu_final=np.log(ru[-2][1]/ru[-1][1])/np.log(2)
print(f"\n  Observed orders: central p = {pc_final:.2f} (design 2), upwind p = {pu_final:.2f} (design 1)")
assert 1.8<pc_final<2.2 and 0.8<pu_final<1.2, "scheme orders wrong"
print("  PASS: central 2nd order, upwind 1st order; central bounded only for Pe_cell<2.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
xx=np.linspace(0,L,400)
ax1.plot(xx, phi_exact(xx,50), "k-", lw=2, label="exact")
ax1.plot(xc_c, phi_c, "C3o-", ms=5, label="central (oscillates)")
ax1.plot(xc_u, phi_u, "C0s-", ms=5, mfc="none", label="upwind (bounded)")
ax1.axhline(0,color="0.6",lw=0.8)
ax1.set_xlabel("x/L"); ax1.set_ylabel(r"$\phi$")
ax1.set_title(f"Pe=50, N=10 (Pe_cell=5)"); ax1.legend(frameon=False,fontsize=8); ax1.grid(alpha=0.3)
hc=np.array([r[0] for r in rc]); ecs=np.array([r[1] for r in rc])
hu=np.array([r[0] for r in ru]); eus=np.array([r[1] for r in ru])
ax2.loglog(hc,ecs,"C3o-",lw=1.7,label="central")
ax2.loglog(hu,eus,"C0s-",lw=1.7,label="upwind")
ax2.loglog(hc,ecs[-1]*(hc/hc[-1])**2,"k:",lw=1.2,label="slope 2")
ax2.loglog(hu,eus[-1]*(hu/hu[-1])**1,"k--",lw=1.2,label="slope 1")
ax2.set_xlabel("mesh spacing $\\Delta x$"); ax2.set_ylabel("max error")
ax2.set_title("Grid convergence (Pe=10)"); ax2.legend(frameon=False,fontsize=8); ax2.grid(alpha=0.3,which="both")
fig.suptitle("Finite-volume convection-diffusion schemes", y=1.04, fontsize=13)
fig.savefig("fig11_2_schemes.png", dpi=150, bbox_inches="tight")
print("  Wrote fig11_2_schemes.png")
