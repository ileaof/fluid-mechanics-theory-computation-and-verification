#!/usr/bin/env python3
"""
Example 10.2 -- Mixing-length RANS solution of turbulent channel flow.

The Reynolds-averaged momentum equation for fully developed turbulent flow between
parallel walls is

        d/dy [ (nu + nu_t) dU/dy ] = (1/rho) dp/dx = -u_tau^2 / h ,

where h is the channel half-height and the total shear stress varies linearly,
tau/rho = u_tau^2 (1 - y/h).  The eddy viscosity is closed with Prandtl's mixing
length modified by van Driest wall damping,

        nu_t = l^2 |dU/dy| ,   l = kappa y [ 1 - exp(-y+/A+) ] ,   A+ = 26 ,

which is NONLINEAR because nu_t depends on dU/dy.  The program solves it by a
cell-centred finite-volume method with Picard iteration, verifies that the computed
mean profile reproduces the law of the wall (sublayer and log law), and confirms
second-order grid convergence of the centreline velocity.  Only numpy and matplotlib
are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kappa, Aplus = 0.41, 26.0
Re_tau = 395.0            # friction Reynolds number (DNS benchmark value)
u_tau, h = 1.0, 1.0
nu = u_tau*h/Re_tau       # so that y+ = y/nu, h+ = Re_tau

def solve_channel(N, tol=1e-11, itmax=3000):
    """Cell-centred FVM for the mixing-length momentum equation; wall y=0, centre y=h."""
    dy = h/N
    yc = (np.arange(N)+0.5)*dy
    yf = np.arange(N+1)*dy
    U = np.zeros(N)
    nut = np.zeros(N+1)
    for _ in range(itmax):
        Uold = U.copy()
        # face gradients dU/dy and van Driest mixing length -> nu_t at faces
        dUdy = np.zeros(N+1)
        dUdy[1:N] = (U[1:]-U[:-1])/dy
        dUdy[0]  = (U[0]-0.0)/(dy/2)          # wall U=0
        dUdy[N]  = 0.0                          # symmetry at centreline
        yplus = yf*u_tau/nu
        l = kappa*yf*(1-np.exp(-yplus/Aplus))
        nut = l**2*np.abs(dUdy)
        Gamma = nu + nut                        # effective diffusivity at faces
        # assemble tridiagonal: d/dy(Gamma dU/dy) = -u_tau^2/h
        aW = np.zeros(N); aE = np.zeros(N)
        aW[1:] = Gamma[1:N]/dy; aE[:-1] = Gamma[1:N]/dy
        aW[0]  = Gamma[0]/(dy/2)               # wall
        aE[-1] = 0.0                            # symmetry (zero flux)
        aP = aW + aE
        b = np.full(N, (u_tau**2/h)*dy)        # source -dp/dx*dy = u_tau^2/h * dy
        b[0] += aW[0]*0.0                       # wall U=0
        # Thomas
        P=np.zeros(N); Q=np.zeros(N)
        P[0]=aE[0]/aP[0]; Q[0]=b[0]/aP[0]
        for i in range(1,N):
            d=aP[i]-aW[i]*P[i-1]; P[i]=aE[i]/d; Q[i]=(b[i]+aW[i]*Q[i-1])/d
        U=np.zeros(N); U[-1]=Q[-1]
        for i in range(N-2,-1,-1): U[i]=P[i]*U[i+1]+Q[i]
        U = 0.5*U + 0.5*Uold                    # under-relax for stability
        if np.max(np.abs(U-Uold)) < tol: break
    return yc, U

print("Example 10.2  Mixing-length RANS channel flow (Re_tau = 395)\n")
# grid convergence of the centreline velocity U+_c
print(f"  {'N':>5} {'U+_centre':>12} {'error vs fine':>15} {'order p':>9}")
ref = None; rows=[]
Us=[]
for N in [40, 80, 160, 320, 640]:
    yc, U = solve_channel(N)
    Uc = U[-1]/u_tau
    Us.append((N,yc,U,Uc))
Uc_fine = Us[-1][3]
prev=None
for (N,yc,U,Uc) in Us[:-1]:
    err = abs(Uc - Uc_fine)
    p = np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"  {N:5d} {Uc:12.5f} {err:15.4e} {p:9.3f}")
    rows.append((h/N, err)); prev=err
print(f"  {Us[-1][0]:5d} {Uc_fine:12.5f} {'(reference)':>15}")
p_final = np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy (centreline U+): p = {p_final:.3f}")
assert 1.6 < p_final < 2.4, "mixing-length channel not ~2nd order"

# verify law of the wall: profile matches sublayer and log law
yc, U = solve_channel(320)
yplus = yc*u_tau/nu; uplus = U/u_tau
# near-wall point in sublayer
i_sub = np.argmin(np.abs(yplus-2.0))
# log-region point
i_log = np.argmin(np.abs(yplus-100.0))
loglaw = (1/kappa)*np.log(yplus[i_log]) + 5.0
print(f"\n  Law-of-the-wall check:")
print(f"    y+={yplus[i_sub]:.2f} (sublayer): u+={uplus[i_sub]:.3f} vs y+={yplus[i_sub]:.3f}")
print(f"    y+={yplus[i_log]:.1f} (log region): u+={uplus[i_log]:.3f} vs log law {loglaw:.3f}")
assert abs(uplus[i_sub]-yplus[i_sub]) < 0.2
assert abs(uplus[i_log]-loglaw) < 1.0
print("  PASS: 2nd-order convergence; profile reproduces the law of the wall.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
ax1.semilogx(yplus, uplus, "C3-", lw=2, label="mixing-length RANS")
yp=np.logspace(0,np.log10(Re_tau),100)
ax1.semilogx(yp[yp<8], yp[yp<8], "k--", lw=1.3, label="$u^+=y^+$")
ax1.semilogx(yp[yp>20], (1/kappa)*np.log(yp[yp>20])+5.0, "k-.", lw=1.3, label="log law")
ax1.set_xlabel("$y^+$"); ax1.set_ylabel("$u^+$"); ax1.set_title("Mean profile in wall units")
ax1.legend(frameon=False); ax1.grid(alpha=0.3, which="both")
# eddy viscosity profile
dUdy=np.gradient(U, yc)
yf=np.arange(321)*(h/320); yplusf=yf*u_tau/nu
l=kappa*yf*(1-np.exp(-yplusf/Aplus)); 
ax2.plot(yc/h, (kappa*yc*(1-np.exp(-yc*u_tau/nu/Aplus)))**2*np.abs(dUdy)/nu, "C0-", lw=2)
ax2.set_xlabel("$y/h$"); ax2.set_ylabel(r"$\nu_t/\nu$")
ax2.set_title("Eddy-viscosity profile"); ax2.grid(alpha=0.3)
fig.suptitle("Mixing-length model of turbulent channel flow", y=1.04, fontsize=13)
fig.savefig("fig10_2_channel.png", dpi=150, bbox_inches="tight")
print("  Wrote fig10_2_channel.png")
