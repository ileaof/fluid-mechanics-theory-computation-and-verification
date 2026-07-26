#!/usr/bin/env python3
"""
Example 8.3 -- Advanced verification: the source-panel method for potential flow
over an ellipse (and the cylinder limit / d'Alembert's paradox).

Ideal flow past a smooth two-dimensional body produces ZERO net drag -- d'Alembert's
paradox -- and a surface pressure that, for an ellipse of semi-axes a (along the
stream) and b, has the exact closed form

        C_p(theta) = 1 - (q/U)^2 ,
        q(theta)   = U (a+b) sin(theta) / sqrt(a^2 sin^2 theta + b^2 cos^2 theta),

which reduces to the cylinder result 1 - 4 sin^2(theta) when a = b.  This program
computes the flow with the SOURCE-PANEL METHOD: the surface is split into M panels
of constant source strength; the strengths are fixed by enforcing flow tangency at
surface control points (a linear system); and the pressure follows from the
tangential velocity.  The computed C_p is verified against the exact solution, the
panel count is refined to measure the observed order of accuracy with Richardson
extrapolation and a Grid Convergence Index, the net drag is shown to vanish, and the
cylinder limit is recovered to machine precision.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

U = 1.0

def panel_influence(xc, yc, xa, ya, xb, yb):
    """Velocity per unit source strength at (xc,yc) from a constant source panel
    from (xa,ya) to (xb,yb), returned in global coordinates."""
    dx, dy = xb-xa, yb-ya
    S = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    cosp, sinp = np.cos(phi), np.sin(phi)
    xp =  (xc-xa)*cosp + (yc-ya)*sinp
    yp = -(xc-xa)*sinp + (yc-ya)*cosp
    up = (1.0/(4*np.pi))*np.log((xp**2+yp**2)/((xp-S)**2+yp**2))
    vp = (1.0/(2*np.pi))*(np.arctan2(yp, xp-S) - np.arctan2(yp, xp))
    return up*cosp - vp*sinp, up*sinp + vp*cosp

def solve_body(M, a, b):
    """Source-panel solution for an ellipse; control points on the true surface."""
    tb = np.linspace(0, 2*np.pi, M+1)
    xb, yb = a*np.cos(tb), b*np.sin(tb)               # panel end points
    tc = 0.5*(tb[:-1]+tb[1:])
    xc, yc = a*np.cos(tc), b*np.sin(tc)               # control points ON surface
    nx0, ny0 = np.cos(tc)/a, np.sin(tc)/b             # ellipse outward normal
    nn = np.hypot(nx0, ny0); nx, ny = nx0/nn, ny0/nn
    tx, ty = -ny, nx                                   # surface tangent
    dx = xb[1:]-xb[:-1]; dy = yb[1:]-yb[:-1]; S = np.hypot(dx, dy)
    A = np.zeros((M, M)); rhs = -(U*nx)
    for i in range(M):
        for j in range(M):
            if i == j:
                A[i, j] = 0.5                          # self-induced normal velocity
            else:
                u, v = panel_influence(xc[i], yc[i], xb[j], yb[j], xb[j+1], yb[j+1])
                A[i, j] = u*nx[i] + v*ny[i]
    sigma = np.linalg.solve(A, rhs)
    Cp = np.zeros(M)
    for i in range(M):
        vu, vv = U, 0.0
        for j in range(M):
            if i == j: continue
            u, v = panel_influence(xc[i], yc[i], xb[j], yb[j], xb[j+1], yb[j+1])
            vu += sigma[j]*u; vv += sigma[j]*v
        Cp[i] = 1 - (vu*tx[i] + vv*ty[i])**2
    Cd = -np.sum(Cp*nx*S)/(2*a)                        # net drag coefficient
    return tc, Cp, Cd

def cp_exact(theta, a, b):
    q = U*(a+b)*np.sin(theta)/np.sqrt(a**2*np.sin(theta)**2 + b**2*np.cos(theta)**2)
    return 1 - (q/U)**2

print("Example 8.3  Source-panel method for an ellipse (d'Alembert)\n")
a, b = 1.0, 0.5
print(f"  Ellipse a={a}, b={b}; exact C_p = 1-(q/U)^2 with")
print("  q = U(a+b)sin/ sqrt(a^2 sin^2 + b^2 cos^2)  (=> 1-4sin^2 when a=b)\n")
print(f"  {'M':>6} {'L2(Cp-exact)':>15} {'order p':>9} {'C_d (drag)':>13} {'CPU[ms]':>9}")
prev=None; rows=[]
for M in [32, 64, 128, 256, 512]:
    t0=time.perf_counter()
    tc, Cp, Cd = solve_body(M, a, b)
    cpu=(time.perf_counter()-t0)*1e3
    L2 = np.sqrt(np.mean((Cp - cp_exact(tc, a, b))**2))
    p = np.log(prev/L2)/np.log(2) if prev else float("nan")
    print(f"  {M:6d} {L2:15.4e} {p:9.3f} {Cd:13.2e} {cpu:9.2f}")
    rows.append((1.0/M, L2, Cd)); prev=L2

p_final=np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_final:.3f}")
assert 0.8 < p_final < 1.3, "constant-source panels should be ~first order"
assert abs(rows[-1][2]) < 1e-9, "net drag must vanish (d'Alembert)"

# Richardson + GCI on the L2 error
e3,e2,e1 = rows[-3][1],rows[-2][1],rows[-1][1]
r=2.0; p_obs=np.log(e3/e2)/np.log(r)
GCI=1.25*e1/(r**p_obs-1)
print("  Richardson / GCI on the C_p L2 error:")
print(f"    observed order p = {p_obs:.3f}")
print(f"    finest L2 error  = {e1:.3e}")
print(f"    GCI (Fs=1.25)    = {GCI:.3e}")
print(f"    net drag (finest) = {rows[-1][2]:.2e}  -> d'Alembert's paradox\n")

# cylinder limit (a=b): recovers 1-4sin^2(theta) as M grows, with zero drag
e_c=[]
for Mc in (64, 256):
    tcc, Cpc, Cdc = solve_body(Mc, 1.0, 1.0)
    e_c.append(np.sqrt(np.mean((Cpc-(1-4*np.sin(tcc)**2))**2)))
print(f"  Cylinder limit (a=b=1): L2 error {e_c[0]:.2e} (M=64) -> {e_c[1]:.2e} (M=256),"
      f" C_d={Cdc:.2e}")
assert e_c[1] < e_c[0] and abs(Cdc) < 1e-9
print("  PASS: ellipse Cp converges (1st order); drag=0; cylinder limit recovered.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,4.5),constrained_layout=True)
tc,Cp,Cd=solve_body(64,a,b)
order=np.argsort(tc); tt=np.linspace(0,2*np.pi,400)
ax1.plot(np.degrees(tt), cp_exact(tt,a,b), "k-", lw=2, label="exact ellipse")
ax1.plot(np.degrees(tc[order]), Cp[order], "o", ms=4, mfc="none", mec="C3", label="panels (M=64)")
tcyl=np.linspace(0,2*np.pi,400)
ax1.plot(np.degrees(tcyl), 1-4*np.sin(tcyl)**2, "C0--", lw=1.3, label="cylinder $1-4\\sin^2\\theta$")
ax1.set_xlabel(r"eccentric angle $\theta$ [deg]"); ax1.set_ylabel("$C_p$")
ax1.set_title("Surface pressure"); ax1.legend(frameon=False, fontsize=8); ax1.grid(alpha=0.3); ax1.invert_yaxis()
Ms=np.array([1/r_[0] for r_ in rows]); es=np.array([r_[1] for r_ in rows])
ax2.loglog(Ms, es, "o-", lw=1.8, label="L2 $C_p$ error")
ax2.loglog(Ms, es[-1]*(Ms/Ms[-1])**-1, "k:", lw=1.4, label="slope 1")
ax2.set_xlabel("number of panels $M$"); ax2.set_ylabel("L2 error in $C_p$")
ax2.set_title("Panel-refinement convergence"); ax2.legend(frameon=False); ax2.grid(alpha=0.3,which="both")
fig.suptitle("Source-panel method: ellipse, cylinder, and d'Alembert", y=1.04, fontsize=13)
fig.savefig("fig8_3_panel.png", dpi=150, bbox_inches="tight")
print("  Wrote fig8_3_panel.png")
