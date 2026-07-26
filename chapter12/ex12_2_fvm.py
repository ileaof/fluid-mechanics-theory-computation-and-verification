#!/usr/bin/env python3
"""
Example 12.2 -- The Grid Convergence Index (GCI) for solution verification.

When a numerical solution has no exact answer to compare against, its numerical
uncertainty is estimated by the Grid Convergence Index of Roache, computed from
three systematically refined grids.  With a functional f measured on coarse (f3),
medium (f2), and fine (f1) grids at refinement ratio r, the OBSERVED order is

        p = ln( (f3 - f2)/(f2 - f1) ) / ln(r) ,

the Richardson estimate of the grid-converged value is

        f_ext = f1 + (f1 - f2)/(r^p - 1) ,

and the GCI on the fine grid, a conservative error bar, is

        GCI = Fs |(f2 - f1)/f1| / (r^p - 1) ,

with a factor of safety Fs = 1.25 for three or more grids.  Here the model
convection-diffusion problem of Chapter 11 IS solved with a known exact solution, so
the GCI can be checked: it should (i) bracket the true error, and (ii) show the
solution is in the asymptotic range, GCI_23 / (r^p GCI_12) ~ 1.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, rho, u, Pe = 1.0, 1.0, 1.0, 10.0
Gamma = rho*u*L/Pe

def phi_exact(x): return (np.exp(Pe*x/L)-1.0)/(np.exp(Pe)-1.0)

def solve(N):
    """Central-difference finite-volume solution; return phi at the midpoint x=0.5."""
    dx=L/N; xc=(np.arange(N)+0.5)*dx; F=rho*u; D=Gamma/dx
    aW=np.full(N, D+F/2); aE=np.full(N, D-F/2)
    Db=Gamma/(dx/2); aWb=Db+F/2; aEb=Db-F/2
    aP=np.zeros(N); b=np.zeros(N)
    aWv=aW.copy(); aEv=aE.copy()
    aWv[0]=aWb; aEv[-1]=aEb
    for i in range(N):
        awi=aWb if i==0 else aW[i]; aei=aEb if i==N-1 else aE[i]
        aP[i]=awi+aei
        if i==N-1: b[i]+=aei*1.0
    P=np.zeros(N); Q=np.zeros(N)
    P[0]=aEv[0]/aP[0]; Q[0]=b[0]/aP[0]
    for i in range(1,N):
        d=aP[i]-aWv[i]*P[i-1]; P[i]=aEv[i]/d; Q[i]=(b[i]+aWv[i]*Q[i-1])/d
    phi=np.zeros(N); phi[-1]=Q[-1]
    for i in range(N-2,-1,-1): phi[i]=P[i]*phi[i+1]+Q[i]
    return np.interp(0.5, xc, phi)

print("Example 12.2  Grid Convergence Index for solution verification\n")
f_true = phi_exact(0.5)
print(f"  Functional: phi at x=0.5;  exact value = {f_true:.10f}\n")

r = 2.0; Fs = 1.25
N3, N2, N1 = 20, 40, 80        # coarse, medium, fine
f3, f2, f1 = solve(N3), solve(N2), solve(N1)
print(f"  Three-grid solutions (refinement ratio r = {r:g}):")
print(f"    coarse (N={N3}):  f3 = {f3:.8f}   true error = {abs(f3-f_true):.3e}")
print(f"    medium (N={N2}):  f2 = {f2:.8f}   true error = {abs(f2-f_true):.3e}")
print(f"    fine   (N={N1}):  f1 = {f1:.8f}   true error = {abs(f1-f_true):.3e}\n")

p_obs = np.log((f3-f2)/(f2-f1))/np.log(r)
f_ext = f1 + (f1-f2)/(r**p_obs-1)
GCI_12 = Fs*abs((f2-f1)/f1)/(r**p_obs-1)          # fine-grid GCI
GCI_23 = Fs*abs((f3-f2)/f2)/(r**p_obs-1)          # medium-grid GCI
print(f"  Observed order of accuracy   p     = {p_obs:.4f}  (central scheme, design 2)")
print(f"  Richardson extrapolate       f_ext = {f_ext:.8f}")
print(f"  Exact value                        = {f_true:.8f}")
print(f"  |f_ext - exact|                    = {abs(f_ext-f_true):.2e}")
print(f"  Fine-grid GCI (Fs={Fs})             = {GCI_12*100:.4f} %")
band = GCI_12*abs(f1)
print(f"  GCI error band on fine grid        = {band:.3e}")
print(f"  Actual fine-grid error             = {abs(f1-f_true):.3e}")
print(f"  GCI brackets the true error?         {band >= abs(f1-f_true)}")

# asymptotic-range check: GCI_23 should ~ r^p * GCI_12
ratio = GCI_23/(r**p_obs * GCI_12)
print(f"\n  Asymptotic-range check  GCI_23/(r^p GCI_12) = {ratio:.4f}  (should be ~1)")
assert abs(p_obs-2.0) < 0.1, "observed order should be ~2"
assert band >= abs(f1-f_true), "GCI must bracket the true error"
assert abs(ratio-1.0) < 0.1, "solution not in the asymptotic range"
print("  PASS: observed order ~2; GCI conservatively brackets the true error;")
print("        solution confirmed in the asymptotic range.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,4.4),constrained_layout=True)
Ns=[10,20,40,80,160,320]; errs=[abs(solve(N)-f_true) for N in Ns]
hs=np.array([L/N for N in Ns])
ax1.loglog(hs, errs, "C0o-", lw=1.8, label="actual error")
ax1.loglog(hs, np.array(errs)[-1]*(hs/hs[-1])**2, "k:", lw=1.2, label="slope 2")
ax1.set_xlabel("mesh spacing $h$"); ax1.set_ylabel("error in $\\phi(0.5)$")
ax1.set_title("Grid convergence"); ax1.legend(frameon=False); ax1.grid(alpha=0.3,which="both")
# GCI error bar illustration
ax2.axhline(f_true, color="k", lw=1.5, label="exact")
xs=[N3,N2,N1]; ys=[f3,f2,f1]
ax2.plot(xs, ys, "C0o-", ms=8, label="grid solutions")
ax2.errorbar([N1],[f1], yerr=[band], fmt="C3s", ms=9, capsize=6, label="fine GCI band")
ax2.plot([N1*1.3],[f_ext],"C2^",ms=9,label="Richardson estimate")
ax2.set_xscale("log"); ax2.set_xlabel("grid size N"); ax2.set_ylabel(r"$\phi(0.5)$")
ax2.set_title("GCI uncertainty band"); ax2.legend(frameon=False,fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle("The Grid Convergence Index", y=1.04, fontsize=13)
fig.savefig("fig12_2_gci.png", dpi=150, bbox_inches="tight")
print("  Wrote fig12_2_gci.png")
