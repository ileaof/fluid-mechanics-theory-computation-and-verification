#!/usr/bin/env python3
"""
Example 4.3 -- Advanced verification: unsteady control-volume mass balance by the
Method of Manufactured Solutions, with adaptive time stepping.

The integral mass balance for a tank with time-varying inflow Q_in(t) and a
gravity-driven outflow Q_out(h) = c sqrt(2 g h) is the ordinary differential
equation

        A dh/dt = Q_in(t) - c sqrt(2 g h) .

For a general Q_in(t) there is no closed-form solution, so to verify a solver we
MANUFACTURE one: we pick a smooth target level h_e(t) and define the inflow

        Q_in(t) = A h_e'(t) + c sqrt(2 g h_e(t))

that makes h_e(t) an exact solution by construction.  We take
h_e(t) = H0 + a sin(omega t), which stays safely positive.

The program then (i) integrates the ODE with the fixed-step classical RK4 method
and confirms fourth-order temporal convergence, with Richardson extrapolation and a
Grid Convergence Index on the final level; and (ii) integrates it with an ADAPTIVE
embedded Runge-Kutta method (Bogacki-Shampine RK23) under several tolerances,
reporting the achieved error, the number of steps, and the CPU time -- showing how
error control trades work for accuracy.  Only numpy and matplotlib are used; no
random numbers are involved.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = 9.81
A = 1.0            # tank area [m^2]
c = 0.02           # outflow coefficient [m^2]
H0, aa, om = 2.0, 0.5, 0.8
Tend = 6.0

def h_exact(t):   return H0 + aa*np.sin(om*t)
def dh_exact(t):  return aa*om*np.cos(om*t)
def Q_in(t):      return A*dh_exact(t) + c*np.sqrt(2*g*h_exact(t))
def f(t, h):      return (Q_in(t) - c*np.sqrt(2*g*max(h, 0.0)))/A

def rk4_fixed(dt):
    n = int(round(Tend/dt)); t, h = 0.0, h_exact(0.0)
    for _ in range(n):
        k1 = f(t, h)
        k2 = f(t+0.5*dt, h+0.5*dt*k1)
        k3 = f(t+0.5*dt, h+0.5*dt*k2)
        k4 = f(t+dt,     h+dt*k3)
        h += dt/6.0*(k1+2*k2+2*k3+k4); t += dt
    return h

print("Example 4.3  Unsteady control volume (MMS) + adaptive time stepping\n")
print(f"  Manufactured level h_e(t) = {H0} + {aa} sin({om} t),  0 <= t <= {Tend}")
print(f"  Exact final level h_e(T) = {h_exact(Tend):.8f} m\n")

# ---- (i) fixed-step RK4 temporal convergence -------------------------------
print("  Fixed-step RK4 temporal convergence:")
print(f"    {'dt':>10} {'final h':>14} {'error':>13} {'order':>8}")
he = h_exact(Tend); prev = None; rows = []
for dt in [0.4, 0.2, 0.1, 0.05, 0.025]:   # stop above the ~1e-13 round-off floor
    hh = rk4_fixed(dt); err = abs(hh - he)
    order = np.log(prev/err)/np.log(2) if prev else float("nan")
    print(f"    {dt:10.5f} {hh:14.8f} {err:13.4e} {order:8.3f}")
    rows.append((dt, hh, err)); prev = err
p_obs = np.log(rows[-2][2]/rows[-1][2])/np.log(rows[-2][0]/rows[-1][0])
print(f"\n  Observed temporal order: p = {p_obs:.3f}")
assert 3.8 < p_obs < 4.2, "RK4 did not achieve fourth-order time accuracy"

# Richardson + GCI on the final level (r=2, three finest grids)
h3, h2, h1 = rows[-3][1], rows[-2][1], rows[-1][1]
r = 2.0
p_r = np.log(abs((h3-h2)/(h2-h1)))/np.log(r)
h_ext = h1 + (h1-h2)/(r**p_r - 1)
Fs = 1.25
GCI = Fs*abs((h1-h2)/h1)/(r**p_r - 1)
print("  Richardson extrapolation of the final level:")
print(f"    observed order p      = {p_r:.3f}")
print(f"    extrapolated h(dt->0) = {h_ext:.10f} m")
print(f"    exact h               = {he:.10f} m")
print(f"    |h_ext - exact|       = {abs(h_ext-he):.2e} m")
print(f"    GCI_fine (Fs={Fs})     = {GCI*100:.3e} %")
assert abs(h_ext - he) < 1e-6, "Richardson estimate inconsistent with exact"
print("  PASS: RK4 fourth order; Richardson recovers the exact final level.\n")

# ---- (ii) adaptive embedded RK23 (Bogacki-Shampine) ------------------------
def rk23_adaptive(tol):
    """Adaptive Bogacki-Shampine RK23 with PI step control; returns (maxerr, nsteps, T, H)."""
    t, h = 0.0, h_exact(0.0)
    dt = 0.05
    ts = [t]; hs = [h]; nsteps = 0; maxerr = 0.0
    while t < Tend - 1e-14:
        dt = min(dt, Tend - t)
        k1 = f(t, h)
        k2 = f(t+0.5*dt,  h+0.5*dt*k1)
        k3 = f(t+0.75*dt, h+0.75*dt*k2)
        h3 = h + dt*(2*k1 + 3*k2 + 4*k3)/9.0         # 3rd-order solution
        k4 = f(t+dt, h3)
        h2 = h + dt*(7*k1 + 6*k2 + 8*k3 + 3*k4)/24.0 # 2nd-order solution
        errest = abs(h3 - h2)
        tolabs = tol*(1 + abs(h))
        if errest <= tolabs or dt < 1e-9:
            t += dt; h = h3
            ts.append(t); hs.append(h); nsteps += 1
            maxerr = max(maxerr, abs(h - h_exact(t)))
        # PI-like step update
        fac = 0.9*(tolabs/max(errest, 1e-16))**(1/3)
        dt *= min(4.0, max(0.2, fac))
    return maxerr, nsteps, np.array(ts), np.array(hs)

print("  Adaptive embedded RK23 (error-controlled):")
print(f"    {'tol':>10} {'steps':>7} {'max error':>13} {'CPU [ms]':>10}")
for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    t0 = time.perf_counter()
    me, ns, T, Hh = rk23_adaptive(tol)
    cpu = (time.perf_counter()-t0)*1e3
    print(f"    {tol:10.0e} {ns:7d} {me:13.4e} {cpu:10.3f}")
me, ns, T_ad, H_ad = rk23_adaptive(1e-6)
assert me < 1e-4, "adaptive integrator did not meet its tolerance"
print("  PASS: adaptive stepping controls the error; cost grows as tol tightens.\n")

# ---- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
tt = np.linspace(0, Tend, 300)
ax1.plot(tt, h_exact(tt), "k-", lw=2, label="manufactured exact $h_e(t)$")
ax1.plot(T_ad, H_ad, "o", ms=5, mfc="none", mec="C3",
         label=f"adaptive RK23 ({len(T_ad)-1} steps)")
ax1.set_xlabel("time $t$ [s]"); ax1.set_ylabel("level $h(t)$ [m]")
ax1.set_title("Unsteady tank level (MMS)")
ax1.legend(frameon=False, fontsize=9); ax1.grid(alpha=0.3)
dts = np.array([r[0] for r in rows]); es = np.array([r[2] for r in rows])
ax2.loglog(dts, es, "o-", lw=1.8, label="RK4 error")
ax2.loglog(dts, es[-1]*(dts/dts[-1])**4, "k:", lw=1.4, label="slope 4 (reference)")
ax2.set_xlabel("time step $\\Delta t$ [s]"); ax2.set_ylabel("final-level error [m]")
ax2.set_title("Fixed-step RK4 convergence")
ax2.legend(frameon=False); ax2.grid(alpha=0.3, which="both")
fig.suptitle("Verification of an unsteady control-volume solver", y=1.04, fontsize=13)
fig.savefig("fig4_3_unsteady.png", dpi=150, bbox_inches="tight")
print("  Wrote fig4_3_unsteady.png")
