#!/usr/bin/env python3
"""
Example 3.3 -- Advanced verification: vorticity and circulation of the
Taylor-Green vortex.

The Taylor-Green velocity field on the plane,

        u =  cos(x) sin(y) ,     v = -sin(x) cos(y) ,

is a standard analytical benchmark.  Its vorticity is known in closed form,

        omega = dv/dx - du/dy = -2 cos(x) cos(y) ,

and the circulation around any closed loop equals, by Stokes' theorem, both the
line integral of the velocity and the area integral of the vorticity,

        Gamma = closed_integral V . dl = double_integral omega dA .

For a rectangle [x1,x2] x [y1,y2] the exact circulation is

        Gamma = -2 (sin x2 - sin x1)(sin y2 - sin y1).

The program (i) computes the vorticity field by second-order central differences
and measures its L2 / Linf convergence to the exact field; (ii) computes the
circulation two independent ways -- the velocity line integral and the vorticity
area integral -- and verifies both against the exact value, with the observed
order of accuracy, Richardson extrapolation, a Grid Convergence Index and CPU
timing.  Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def u(x, y):  return np.cos(x) * np.sin(y)
def v(x, y):  return -np.sin(x) * np.cos(y)
def omega_exact(x, y): return -2.0 * np.cos(x) * np.cos(y)

# rectangular circulation loop (deliberately not aligned with a period)
x1, x2 = 0.5, 2.3
y1, y2 = 0.8, 2.9
Gamma_exact = -2.0 * (np.sin(x2) - np.sin(x1)) * (np.sin(y2) - np.sin(y1))

def vorticity_field_error(N):
    """Second-order central-difference vorticity on an N x N grid over [0,2pi]^2."""
    xs = np.linspace(0, 2*np.pi, N, endpoint=False)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    U, Vv = u(X, Y), v(X, Y)
    # periodic central differences
    dVdx = (np.roll(Vv, -1, 0) - np.roll(Vv, 1, 0)) / (2*h)
    dUdy = (np.roll(U, -1, 1) - np.roll(U, 1, 1)) / (2*h)
    w = dVdx - dUdy
    we = omega_exact(X, Y)
    L2 = np.sqrt(np.mean((w - we)**2))
    Linf = np.max(np.abs(w - we))
    return h, L2, Linf

def circulation_line(n):
    """Midpoint line integral of V.dl around the rectangle, n segments per edge."""
    G = 0.0
    # bottom (y=y1, x: x1->x2), dl = (dx,0)
    xm = x1 + (np.arange(n)+0.5)*(x2-x1)/n; dx = (x2-x1)/n
    G += np.sum(u(xm, y1)) * dx
    # right (x=x2, y: y1->y2), dl=(0,dy)
    ym = y1 + (np.arange(n)+0.5)*(y2-y1)/n; dy = (y2-y1)/n
    G += np.sum(v(x2, ym)) * dy
    # top (y=y2, x: x2->x1), dl=(-dx,0)
    G += np.sum(u(xm, y2)) * (-dx)
    # left (x=x1, y: y2->y1), dl=(0,-dy)
    G += np.sum(v(x1, ym)) * (-dy)
    return G

def circulation_area(n):
    """Midpoint area integral of vorticity over the rectangle, n x n cells."""
    xm = x1 + (np.arange(n)+0.5)*(x2-x1)/n
    ym = y1 + (np.arange(n)+0.5)*(y2-y1)/n
    X, Y = np.meshgrid(xm, ym, indexing="ij")
    dA = (x2-x1)/n * (y2-y1)/n
    return np.sum(omega_exact(X, Y)) * dA

print("Example 3.3  Taylor-Green vortex: vorticity and circulation\n")
print(f"  Loop [{x1},{x2}] x [{y1},{y2}];  exact circulation Gamma = {Gamma_exact:.8f}\n")

# --- (i) vorticity field convergence ----------------------------------------
print("  Central-difference vorticity field convergence:")
print(f"    {'N':>5} {'h':>10} {'L2 error':>12} {'p(L2)':>7} {'Linf err':>12} {'p(Linf)':>8}")
prev2 = previ = None
for N in [16, 32, 64, 128, 256]:
    h, L2, Linf = vorticity_field_error(N)
    p2 = np.log(prev2/L2)/np.log(2) if prev2 else float("nan")
    pi = np.log(previ/Linf)/np.log(2) if previ else float("nan")
    print(f"    {N:5d} {h:10.4f} {L2:12.4e} {p2:7.3f} {Linf:12.4e} {pi:8.3f}")
    prev2, previ = L2, Linf
assert 1.9 < p2 < 2.1, "vorticity differencing is not second order"

# --- (ii) circulation: line vs area integral, verification campaign ---------
print("\n  Circulation by line integral (V.dl) and area integral (omega dA):")
print(f"    {'n':>5} {'Gamma_line':>13} {'err_line':>11} {'p':>6}"
      f" {'Gamma_area':>13} {'err_area':>11} {'CPU[ms]':>8}")
rows = []
prev = None
for n in [4, 8, 16, 32, 64, 128]:
    t0 = time.perf_counter()
    Gl = circulation_line(n)
    Ga = circulation_area(n)
    cpu = (time.perf_counter()-t0)*1e3
    el = abs(Gl - Gamma_exact); ea = abs(Ga - Gamma_exact)
    p = np.log(prev/el)/np.log(2) if prev else float("nan")
    print(f"    {n:5d} {Gl:13.8f} {el:11.3e} {p:6.3f} {Ga:13.8f} {ea:11.3e} {cpu:8.3f}")
    rows.append((n, Gl, el, Ga, ea)); prev = el

p_line = np.log(rows[-2][2]/rows[-1][2])/np.log(2)
print(f"\n  Observed order of the line-integral circulation: p = {p_line:.4f}")
assert 1.9 < p_line < 2.1, "circulation line integral not second order"

# Richardson + GCI on the line-integral circulation
G3, G2, G1 = rows[-3][1], rows[-2][1], rows[-1][1]
r = 2.0
p_obs = np.log(abs((G3-G2)/(G2-G1)))/np.log(r)
G_ext = G1 + (G1-G2)/(r**p_obs - 1)
Fs = 1.25
GCI = Fs*abs((G1-G2)/G1)/(r**p_obs - 1)
print("\n  Richardson extrapolation of the circulation:")
print(f"    extrapolated Gamma(h->0) = {G_ext:.8f}")
print(f"    exact Gamma              = {Gamma_exact:.8f}")
print(f"    |Gamma_ext - exact|      = {abs(G_ext-Gamma_exact):.2e}")
print(f"    GCI_fine (Fs={Fs})        = {GCI*100:.5f} %")
print(f"    line vs area agree to     {abs(rows[-1][1]-rows[-1][3]):.2e}"
      f"  (Stokes' theorem check)")
assert abs(G_ext - Gamma_exact) < 1e-4, "Richardson estimate inconsistent"
print("  PASS: vorticity 2nd order; line = area = exact circulation (Stokes).\n")

# --- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
xs = np.linspace(0, 2*np.pi, 200); X, Y = np.meshgrid(xs, xs, indexing="ij")
cf = ax1.contourf(X, Y, omega_exact(X, Y), levels=20, cmap="RdBu_r")
sk = 12
ax1.quiver(X[::sk*1, ::sk*1], Y[::sk*1, ::sk*1],
           u(X, Y)[::sk*1, ::sk*1], v(X, Y)[::sk*1, ::sk*1], color="k", alpha=0.5)
ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k-", lw=2.2)
ax1.set_title("Vorticity field and circulation loop")
ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_aspect("equal")
fig.colorbar(cf, ax=ax1, shrink=0.85, label=r"$\omega$")

ns = np.array([r[0] for r in rows]); els = np.array([r[2] for r in rows])
eas = np.array([r[4] for r in rows])
ax2.loglog(ns, els, "o-", lw=1.8, label="line integral error")
ax2.loglog(ns, eas, "s--", lw=1.8, label="area integral error")
ax2.loglog(ns, els[-1]*(ns[-1]/ns)**2, "k:", lw=1.4, label="slope 2 (reference)")
ax2.set_xlabel("segments per edge $n$"); ax2.set_ylabel("circulation error")
ax2.set_title("Convergence of the circulation")
ax2.legend(frameon=False, fontsize=9); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Taylor-Green vortex: vorticity and circulation (Stokes' theorem)",
             y=1.05, fontsize=13)
fig.savefig("fig3_3_taylorgreen.png", dpi=150, bbox_inches="tight")
print("  Wrote fig3_3_taylorgreen.png")
