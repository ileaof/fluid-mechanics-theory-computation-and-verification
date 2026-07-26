#!/usr/bin/env python3
"""
Example 5.3 -- Advanced verification: the Taylor-Green vortex as an exact
Navier-Stokes solution, and its vorticity-transport equation.

The two-dimensional velocity field

        u =  cos(x) sin(y) F(t) ,   v = -sin(x) cos(y) F(t) ,   F(t)=exp(-2 nu t)

is an EXACT solution of the incompressible Navier-Stokes equations: the nonlinear
convective term is exactly balanced by a pressure gradient, and viscosity makes the
whole field decay as exp(-2 nu t).  Its vorticity omega = -2 cos(x) cos(y) F(t)
obeys the 2-D vorticity-transport equation

        d omega/dt + (V . grad) omega = nu laplacian(omega) ,

in which vortex stretching is absent.  For this field the convective term vanishes
identically, so the vorticity simply diffuses and decays.

The program verifies the DISCRETE vorticity-transport operator: it evaluates
R_h = -(V . grad) omega + nu laplacian(omega) by second-order central differences
and compares it with the exact d omega/dt = -2 nu omega.  It measures the L2 / Linf
convergence of the operator, extrapolates by Richardson, forms a Grid Convergence
Index, then marches the vorticity in time (RK4) to confirm the exp(-2 nu t) decay.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nu = 0.05

def grid(N):
    xs = np.linspace(0, 2*np.pi, N, endpoint=False)
    h = xs[1]-xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return X, Y, h

def fields(X, Y, t=0.0):
    F = np.exp(-2*nu*t)
    u = np.cos(X)*np.sin(Y)*F
    v = -np.sin(X)*np.cos(Y)*F
    w = -2*np.cos(X)*np.cos(Y)*F
    return u, v, w

def d_dx(f, h): return (np.roll(f,-1,0) - np.roll(f,1,0))/(2*h)
def d_dy(f, h): return (np.roll(f,-1,1) - np.roll(f,1,1))/(2*h)
def lap(f, h):
    return ((np.roll(f,-1,0)-2*f+np.roll(f,1,0)) +
            (np.roll(f,-1,1)-2*f+np.roll(f,1,1)))/h**2

def operator_error(N):
    """Discrete vorticity-transport RHS vs exact d omega/dt = -2 nu omega."""
    X, Y, h = grid(N)
    u, v, w = fields(X, Y, 0.0)
    conv = u*d_dx(w, h) + v*d_dy(w, h)
    diff = nu*lap(w, h)
    Rh = -conv + diff
    exact = -2*nu*w                        # d omega/dt
    e = Rh - exact
    return h, np.sqrt(np.mean(e**2)), np.max(np.abs(e))

print("Example 5.3  Taylor-Green vortex: exact NS solution & vorticity transport\n")
print(f"  nu = {nu};  omega(t) = -2 cos x cos y exp(-2 nu t)\n")
print("  Discrete vorticity-transport operator convergence:")
print(f"    {'N':>5} {'h':>10} {'L2 error':>13} {'p(L2)':>7} {'Linf err':>13} {'p(Linf)':>8}")
prev2 = previ = None; rows = []
for N in [16, 32, 64, 128, 256]:
    h, L2, Linf = operator_error(N)
    p2 = np.log(prev2/L2)/np.log(2) if prev2 else float("nan")
    pi = np.log(previ/Linf)/np.log(2) if previ else float("nan")
    print(f"    {N:5d} {h:10.4f} {L2:13.4e} {p2:7.3f} {Linf:13.4e} {pi:8.3f}")
    rows.append((h, L2)); prev2, previ = L2, Linf
p_final = np.log(rows[-2][1]/rows[-1][1])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed order of the vorticity-transport operator: p = {p_final:.4f}")
assert 1.9 < p_final < 2.1, "operator is not second order"

# Richardson + GCI on the L2 operator error
e3, e2, e1 = rows[-3][1], rows[-2][1], rows[-1][1]
r = 2.0
p_obs = np.log(e3/e2)/np.log(r)
Fs = 1.25
GCI = Fs*abs(e1/ (r**p_obs - 1))
print("  Richardson / GCI on the operator error:")
print(f"    observed order p = {p_obs:.3f}")
print(f"    L2 error (finest N=256) = {e1:.4e}")
print(f"    Richardson error estimate for the fine grid: {e1/(r**p_obs-1):.4e}")
print(f"    GCI_fine (Fs={Fs}) = {GCI:.4e}\n")

# ---- (ii) time march confirms exp(-2 nu t) decay ---------------------------
def march_rk4(N, t_end, nt):
    X, Y, h = grid(N)
    u0, v0, w = fields(X, Y, 0.0)
    dt = t_end/nt
    def rhs(w, t):
        F = np.exp(-2*nu*t)
        u = np.cos(X)*np.sin(Y)*F; v = -np.sin(X)*np.cos(Y)*F
        return -(u*d_dx(w,h) + v*d_dy(w,h)) + nu*lap(w,h)
    t = 0.0
    for _ in range(nt):
        k1 = rhs(w, t); k2 = rhs(w+0.5*dt*k1, t+0.5*dt)
        k3 = rhs(w+0.5*dt*k2, t+0.5*dt); k4 = rhs(w+dt*k3, t+dt)
        w = w + dt/6*(k1+2*k2+2*k3+k4); t += dt
    return X, Y, w

t_end = 2.0
t0 = time.perf_counter()
X, Y, w_num = march_rk4(128, t_end, 400)
cpu = time.perf_counter()-t0
_, _, w_ex = fields(X, Y, t_end)
err_decay = np.max(np.abs(w_num - w_ex))
peak_num = np.max(np.abs(w_num)); peak_ex = 2*np.exp(-2*nu*t_end)
print(f"  Time march to t={t_end}: peak |omega| numeric {peak_num:.5f}, "
      f"exact {peak_ex:.5f}")
print(f"    max|omega_num - omega_exact| = {err_decay:.3e},  CPU = {cpu*1e3:.1f} ms")
assert err_decay < 1e-3, "time march does not reproduce the decaying solution"
print("  PASS: 2nd-order operator; Taylor-Green decay exp(-2 nu t) reproduced.\n")

# ---- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
cf = ax1.contourf(X, Y, w_num, levels=20, cmap="RdBu_r")
ax1.set_title(f"Vorticity at t={t_end} (decayed)")
ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_aspect("equal")
fig.colorbar(cf, ax=ax1, shrink=0.85, label=r"$\omega$")
hs = np.array([r_[0] for r_ in rows]); es = np.array([r_[1] for r_ in rows])
ax2.loglog(hs, es, "o-", lw=1.8, label="operator $L_2$ error")
ax2.loglog(hs, es[-1]*(hs/hs[-1])**2, "k:", lw=1.4, label="slope 2 (reference)")
ax2.set_xlabel("mesh spacing $h$"); ax2.set_ylabel("operator error")
ax2.set_title("Vorticity-transport operator convergence")
ax2.legend(frameon=False); ax2.grid(alpha=0.3, which="both")
fig.suptitle("Taylor-Green vortex: an exact Navier-Stokes solution", y=1.05, fontsize=13)
fig.savefig("fig5_3_taylorgreen.png", dpi=150, bbox_inches="tight")
print("  Wrote fig5_3_taylorgreen.png")
