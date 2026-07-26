#!/usr/bin/env python3
"""
Example 10.3 -- Code verification of a coupled k-epsilon RANS discretisation by the
Method of Manufactured Solutions.

Two-equation turbulence models solve, with the mean momentum equation, two nonlinear
transport equations for the turbulent kinetic energy k and its dissipation rate
epsilon, coupled through the eddy viscosity nu_t = C_mu k^2/epsilon:

  momentum:  d/dy[(nu+nu_t) dU/dy] + S_U = 0
  k:         d/dy[(nu+nu_t/sk) dk/dy] + P - epsilon + S_k = 0,   P = nu_t (dU/dy)^2
  epsilon:   d/dy[(nu+nu_t/se) de/dy] + Ce1 (e/k) P - Ce2 e^2/k + S_e = 0

There is no analytical solution to verify such a solver against.  The Method of
Manufactured Solutions supplies one: choose smooth fields U_e, k_e, epsilon_e,
substitute them into the operators to obtain the source terms that make them exact,
and then confirm that the DISCRETE operator reproduces the equations to the design
order of accuracy.  This program measures the truncation error -- the discrete
residual obtained when the exact fields are substituted into the second-order
finite-difference operators -- and shows it vanishes at second order for all three
coupled equations, with Richardson extrapolation, a Grid Convergence Index, and CPU
timing.  This "order-of-accuracy" test is the gold standard of CFD code verification.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nu = 0.01
Cmu, Ce1, Ce2, sk, se = 0.09, 1.44, 1.92, 1.0, 1.3
PI = np.pi

# ---- manufactured fields and their analytic derivatives --------------------
def U_e(y):  return np.sin(PI*y/2)
def Up(y):   return (PI/2)*np.cos(PI*y/2)
def Upp(y):  return -(PI/2)**2*np.sin(PI*y/2)
def k_e(y):  return 1.5 - 0.5*np.cos(PI*y)
def kp(y):   return 0.5*PI*np.sin(PI*y)
def kpp(y):  return 0.5*PI**2*np.cos(PI*y)
def e_e(y):  return 2.0 + np.sin(PI*y)
def ep(y):   return PI*np.cos(PI*y)
def epp(y):  return -PI**2*np.sin(PI*y)
def nut(y):  return Cmu*k_e(y)**2/e_e(y)
def nutp(y): return Cmu*(2*k_e(y)*kp(y)*e_e(y) - k_e(y)**2*ep(y))/e_e(y)**2

def sources(y):
    """Exact source terms that make (U_e,k_e,e_e) an exact solution."""
    P = nut(y)*Up(y)**2
    S_U = -( nutp(y)*Up(y) + (nu+nut(y))*Upp(y) )
    S_k = -( (nutp(y)/sk)*kp(y) + (nu+nut(y)/sk)*kpp(y) + P - e_e(y) )
    S_e = -( (nutp(y)/se)*ep(y) + (nu+nut(y)/se)*epp(y)
             + Ce1*(e_e(y)/k_e(y))*P - Ce2*e_e(y)**2/k_e(y) )
    return S_U, S_k, S_e

def ddy_flux(Gam, phi, h):
    """Second-order discrete form of d/dy(Gam dphi/dy) at interior nodes."""
    N = len(phi)-1
    r = np.zeros(N+1)
    Ge = 0.5*(Gam[1:-1]+Gam[2:]); Gw = 0.5*(Gam[1:-1]+Gam[:-2])
    r[1:-1] = (Ge*(phi[2:]-phi[1:-1]) - Gw*(phi[1:-1]-phi[:-2]))/h**2
    return r

def residuals(N):
    """Discrete residual of each equation when the exact fields are substituted."""
    y = np.linspace(0,1,N+1); h = y[1]-y[0]
    U, k, e = U_e(y), k_e(y), e_e(y); nt = Cmu*k**2/e
    S_U, S_k, S_e = sources(y)
    Up_num = np.gradient(U, h); P = nt*Up_num**2
    rU = ddy_flux(nu+nt,    U, h) + S_U
    rk = ddy_flux(nu+nt/sk, k, h) + P - e + S_k
    re = ddy_flux(nu+nt/se, e, h) + Ce1*(e/k)*P - Ce2*e**2/k + S_e
    L2 = lambda r: np.sqrt(np.mean(r[1:-1]**2))
    return L2(rU), L2(rk), L2(re)

print("Example 10.3  MMS order-of-accuracy verification of a k-epsilon solver\n")
print(f"  {'N':>5} {'res(U)':>12} {'res(k)':>12} {'res(eps)':>12} {'p(k)':>7} {'CPU[ms]':>9}")
prev=None; rows=[]
for N in [20,40,80,160,320,640]:
    t0=time.perf_counter(); rU,rk,re=residuals(N); cpu=(time.perf_counter()-t0)*1e3
    p=np.log(prev/rk)/np.log(2) if prev else float("nan")
    print(f"  {N:5d} {rU:12.4e} {rk:12.4e} {re:12.4e} {p:7.3f} {cpu:9.2f}")
    rows.append((1.0/N,rU,rk,re)); prev=rk

pU=np.log(rows[-2][1]/rows[-1][1])/np.log(2)
pk=np.log(rows[-2][2]/rows[-1][2])/np.log(2)
pe=np.log(rows[-2][3]/rows[-1][3])/np.log(2)
print(f"\n  Observed order of accuracy (finest pair): U {pU:.3f}, k {pk:.3f}, eps {pe:.3f}")
assert 1.8<pU<2.2 and 1.8<pk<2.2 and 1.8<pe<2.2, "coupled operator not 2nd order"

# Richardson + GCI on the k-equation residual
e3,e2,e1=rows[-3][2],rows[-2][2],rows[-1][2]
r=2.0; p_obs=np.log(e3/e2)/np.log(r); GCI=1.25*e1/(r**p_obs-1)
print("  Richardson / GCI on the k-equation truncation error:")
print(f"    observed order p = {p_obs:.3f}")
print(f"    finest residual  = {e1:.3e}")
print(f"    GCI (Fs=1.25)    = {GCI:.3e}")
print("  PASS: momentum, k and epsilon operators are all 2nd-order (MMS verified).\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,4.5),constrained_layout=True)
y=np.linspace(0,1,200)
ax1.plot(y,U_e(y),"C0-",lw=2,label="$U_e$")
ax1.plot(y,k_e(y),"C3--",lw=2,label="$k_e$")
ax1.plot(y,e_e(y),"C2-.",lw=2,label=r"$\epsilon_e$")
ax1.plot(y,nut(y),"C1:",lw=2,label=r"$\nu_t=C_\mu k^2/\epsilon$")
ax1.set_xlabel("y"); ax1.set_title("Manufactured solution")
ax1.legend(frameon=False,fontsize=9); ax1.grid(alpha=0.3)
hs=np.array([r_[0] for r_ in rows])
rUs=np.array([r_[1] for r_ in rows]); rks=np.array([r_[2] for r_ in rows]); res=np.array([r_[3] for r_ in rows])
ax2.loglog(hs,rUs,"o-",lw=1.7,label="momentum"); ax2.loglog(hs,rks,"s-",lw=1.7,label="$k$")
ax2.loglog(hs,res,"^-",lw=1.7,label=r"$\epsilon$")
ax2.loglog(hs,rks[-1]*(hs/hs[-1])**2,"k:",lw=1.4,label="slope 2")
ax2.set_xlabel("mesh spacing $h$"); ax2.set_ylabel("truncation error (L2 residual)")
ax2.set_title("Order-of-accuracy verification"); ax2.legend(frameon=False,fontsize=9)
ax2.grid(alpha=0.3,which="both")
fig.suptitle("MMS code verification of a coupled k-epsilon discretisation", y=1.04, fontsize=13)
fig.savefig("fig10_3_kepsilon.png", dpi=150, bbox_inches="tight")
print("  Wrote fig10_3_kepsilon.png")
