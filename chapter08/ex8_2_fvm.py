#!/usr/bin/env python3
"""
Example 8.2 -- The Blasius boundary layer by an iterative integral method.

Example 8.1 solved the Blasius equation f''' + (1/2) f f'' = 0 by shooting.  Here
the SAME problem is solved by a completely different, non-iterative-in-boundary
route that exposes its structure and provides an independent verification.  Writing
p = f'' , the equation is the first-order linear relation p' = -(1/2) f p, whose
solution is

        f''(eta) = C exp( -(1/2) integral_0^eta f d eta' ),

so that, given the current f, the whole profile is rebuilt by quadrature:

        f'(eta) = C integral_0^eta exp(-(1/2) I) d eta' ,   I(eta)=integral_0^eta f ,

with the constant C fixed by the outer condition f'(inf)=1.  Iterating this map
(a Picard iteration) converges to the Blasius solution, and the wall curvature is
recovered as f''(0) = C.  Because every integral is evaluated with the trapezoidal
rule, the method is SECOND-ORDER in the mesh spacing, and refining the grid drives
f''(0) to 0.33206 at the design rate -- an independent check on Example 8.1.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def cumtrap(y, h):
    """Cumulative trapezoidal integral with c[0]=0 (2nd-order)."""
    c = np.zeros_like(y)
    c[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*h)
    return c

def blasius_integral(N, eta_max=10.0, tol=1e-13, itmax=500):
    """Solve Blasius by Picard iteration on the integral form; return eta, f', f''(0)."""
    eta = np.linspace(0.0, eta_max, N+1); h = eta[1]-eta[0]
    f = 0.5*eta**2 * 0.166                       # mild initial guess
    C_old = 0.0
    for _ in range(itmax):
        I = cumtrap(f, h)                        # I(eta) = integral of f
        w = np.exp(-0.5*I)                       # shape of f''
        C = 1.0/trapezoid(w, eta)                # enforce f'(inf)=1
        fp = C*cumtrap(w, h)                      # f' = u/U
        f  = cumtrap(fp, h)                       # f
        if abs(C - C_old) < tol: break
        C_old = C
    return eta, fp, C                             # C = f''(0)

print("Example 8.2  Blasius boundary layer by an iterative integral method\n")
print("  Independent cross-check of Example 8.1 (shooting), which gave f''(0)=0.332057\n")
hdr = "N".rjust(6)+"d_eta".rjust(11)+"f''(0)".rjust(12)+"error".rjust(12)+"order p".rjust(9)
print("  "+hdr)
exact = 0.33205734                                # accepted Blasius constant
prev=None; rows=[]
for N in [40, 80, 160, 320, 640, 1280]:
    eta, fp, C = blasius_integral(N)
    err = abs(C - exact)
    p = np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"  {N:6d} {eta[1]-eta[0]:10.4f} {C:12.7f} {err:12.3e} {p:9.3f}")
    rows.append((eta[1]-eta[0], err, (eta, fp))); prev=err

p_final = np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 1.9 < p_final < 2.1, "integral method not second order"
assert abs(rows[-1][1]) < 1e-4, "did not converge to the Blasius constant"

# Richardson extrapolation of f''(0)
e2,e1 = rows[-2][1], rows[-1][1]
C_fine = exact + (e1 if rows[-1] else 0)          # placeholder; use values
C3,C2,C1 = (exact+rows[-3][1]), (exact+rows[-2][1]), (exact+rows[-1][1])
# use signed values instead:
vals=[]
for N in [320,640,1280]:
    _,_,C=blasius_integral(N); vals.append(C)
r=2.0; p_obs=np.log(abs((vals[0]-vals[1])/(vals[1]-vals[2])))/np.log(r)
C_ext=vals[2]+(vals[2]-vals[1])/(r**p_obs-1)
print(f"  Richardson extrapolate f''(0) = {C_ext:.8f}  (accepted 0.33205734)")
print(f"    |extrap - accepted| = {abs(C_ext-exact):.2e}")
assert abs(C_ext-exact) < 1e-5
print("  PASS: second order; f''(0) -> 0.33206, cross-checking Example 8.1.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,4.5),constrained_layout=True)
eta,fp = rows[-1][2]
ax1.plot(fp, eta, "C0-", lw=2, label="integral method")
ax1.set_ylim(0,8); ax1.set_xlabel("$u/U=f'(\\eta)$"); ax1.set_ylabel(r"$\eta$")
ax1.set_title("Blasius profile (integral method)"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
hs=np.array([r[0] for r in rows]); es=np.array([r[1] for r in rows])
ax2.loglog(hs,es,"o-",lw=1.8,label="error in $f''(0)$")
ax2.loglog(hs,es[-1]*(hs/hs[-1])**2,"k:",lw=1.4,label="slope 2")
ax2.set_xlabel(r"mesh spacing $\Delta\eta$"); ax2.set_ylabel("error in $f''(0)$")
ax2.set_title("Second-order convergence"); ax2.legend(frameon=False); ax2.grid(alpha=0.3,which="both")
fig.suptitle("Blasius solution by iterative quadrature", y=1.04, fontsize=13)
fig.savefig("fig8_2_marching.png", dpi=150, bbox_inches="tight")
print("  Wrote fig8_2_marching.png")
