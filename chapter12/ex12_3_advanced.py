#!/usr/bin/env python3
"""
Example 12.3 -- Code verification of a nonlinear solver by the Method of Manufactured
Solutions, and the detection of a coding bug.

The steady viscous Burgers equation,

        u du/dx - nu d^2u/dx^2 = S(x) ,   u(0)=u_e(0) , u(1)=u_e(1) ,

is the one-dimensional nonlinear model of the Navier-Stokes equations: it has the
same convective nonlinearity u du/dx balanced by diffusion.  The Method of
Manufactured Solutions verifies a solver for it: choose a smooth field u_e(x),
substitute it into the operator to obtain the source S(x) that makes it exact, and
confirm the solver reproduces u_e at its design order of accuracy.

Two solvers are compared.  The CORRECT solver uses second-order central differences
for both the convection and the diffusion term, and should converge at second order.
The BUGGY solver differs by a single, plausible-looking line -- it discretises the
convective derivative with a first-order one-sided difference -- and, although it still
runs and produces reasonable-looking answers, its order of accuracy collapses to one.
The Method of Manufactured Solutions detects the bug that ordinary testing would miss.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nu = 0.05
PI = np.pi

# manufactured solution and the source term that makes it exact
def u_e(x):  return 1.0 + 0.5*np.sin(PI*x)
def up(x):   return 0.5*PI*np.cos(PI*x)
def upp(x):  return -0.5*PI**2*np.sin(PI*x)
def source(x): return u_e(x)*up(x) - nu*upp(x)

def solve(N, buggy=False, tol=1e-12, itmax=500):
    x=np.linspace(0,1,N+1); h=x[1]-x[0]
    S=source(x)
    u=np.linspace(u_e(0), u_e(1), N+1)          # initial guess (linear)
    for _ in range(itmax):
        uo=u.copy()
        a=np.zeros(N+1); b=np.zeros(N+1); c=np.zeros(N+1); d=np.zeros(N+1)
        b[0]=1; d[0]=u_e(0); b[N]=1; d[N]=u_e(1)
        for i in range(1,N):
            uc=u[i]
            if not buggy:
                # CORRECT: central convection (2nd order) + central diffusion
                aW = -uc/(2*h) - nu/h**2
                aP =  2*nu/h**2
                aE =  uc/(2*h) - nu/h**2
            else:
                # BUG: first-order backward difference for the convective term
                aW = -uc/h - nu/h**2
                aP =  2*nu/h**2 + uc/h
                aE = -nu/h**2
            a[i]=aW; b[i]=aP; c[i]=aE; d[i]=S[i]
        # Thomas
        for i in range(1,N+1):
            m=a[i]/b[i-1]; b[i]-=m*c[i-1]; d[i]-=m*d[i-1]
        u=np.zeros(N+1); u[N]=d[N]/b[N]
        for i in range(N-1,-1,-1): u[i]=(d[i]-c[i]*u[i+1])/b[i]
        if np.max(np.abs(u-uo))<tol: break
    return x,u

def order_study(buggy):
    print(f"    {'N':>6} {'L2 error':>13} {'observed order':>15}")
    prev=None; rows=[]
    for N in [20,40,80,160,320]:
        x,u=solve(N,buggy)
        e=np.sqrt(np.mean((u-u_e(x))**2))
        p=np.log(prev/e)/np.log(2) if prev else float("nan")
        print(f"    {N:6d} {e:13.4e} {p:15.3f}")
        rows.append((1.0/N,e)); prev=e
    pf=np.log(rows[-2][1]/rows[-1][1])/np.log(2)
    return pf, rows

print("Example 12.3  MMS code verification of a nonlinear solver, with bug detection\n")
print("  Manufactured solution u_e(x) = 1 + 0.5 sin(pi x), viscous Burgers equation\n")

print("  CORRECT solver (central convection + central diffusion):")
p_ok, rows_ok = order_study(False)
print(f"    -> observed order of accuracy = {p_ok:.3f}  (design 2)")
assert 1.8 < p_ok < 2.2, "correct solver should be 2nd order"
print("    VERIFICATION PASSED: solver achieves its design order.\n")

print("  BUGGY solver (first-order one-sided convective difference):")
p_bug, rows_bug = order_study(True)
print(f"    -> observed order of accuracy = {p_bug:.3f}  (design 2 -- but only 1 achieved!)")
assert 0.8 < p_bug < 1.3, "buggy solver should degrade to 1st order"
print("    VERIFICATION FAILED: order below design -> the Method of Manufactured")
print("    Solutions has detected the coding bug.\n")

# Richardson/GCI on the correct solver
e3,e2,e1=rows_ok[-3][1],rows_ok[-2][1],rows_ok[-1][1]
r=2.0; p_obs=np.log(e3/e2)/np.log(r); GCI=1.25*e1/(r**p_obs-1)
print(f"  Correct solver: Richardson order p={p_obs:.3f}, fine-grid GCI (Fs=1.25) = {GCI:.3e}")
print("  PASS: MMS verifies the correct solver (order 2) and flags the buggy one (order 1).\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,4.5),constrained_layout=True)
x,u=solve(40,False); xb,ub=solve(40,True)
xx=np.linspace(0,1,200)
ax1.plot(xx,u_e(xx),"k-",lw=2,label="manufactured $u_e$")
ax1.plot(x,u,"C0o",ms=4,mfc="none",label="correct solver")
ax1.plot(xb,ub,"C3s",ms=4,mfc="none",label="buggy solver")
ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.set_title("Solutions (N=40)")
ax1.legend(frameon=False,fontsize=9); ax1.grid(alpha=0.3)
ho=np.array([r_[0] for r_ in rows_ok]); eo=np.array([r_[1] for r_ in rows_ok])
hb=np.array([r_[0] for r_ in rows_bug]); eb=np.array([r_[1] for r_ in rows_bug])
ax2.loglog(ho,eo,"C0o-",lw=1.8,label="correct (order 2)")
ax2.loglog(hb,eb,"C3s-",lw=1.8,label="buggy (order 1)")
ax2.loglog(ho,eo[-1]*(ho/ho[-1])**2,"k:",lw=1.2,label="slope 2")
ax2.loglog(hb,eb[-1]*(hb/hb[-1])**1,"k--",lw=1.2,label="slope 1")
ax2.set_xlabel("mesh spacing $h$"); ax2.set_ylabel("L2 error")
ax2.set_title("MMS detects the order degradation"); ax2.legend(frameon=False,fontsize=8)
ax2.grid(alpha=0.3,which="both")
fig.suptitle("Manufactured-solution verification and bug detection", y=1.04, fontsize=13)
fig.savefig("fig12_3_mms.png", dpi=150, bbox_inches="tight")
print("  Wrote fig12_3_mms.png")
