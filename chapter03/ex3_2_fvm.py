#!/usr/bin/env python3
"""
Example 3.2 -- Streamlines, pathlines and streaklines of an unsteady flow.

For a STEADY flow the three families of curves coincide.  For an UNSTEADY flow
they differ, and confusing them is a classic source of error in interpreting flow
visualisations.  Take the spatially uniform but time-periodic field

        u = U0 ,     v = V0 cos(omega t) ,

which has simple closed forms for all three curves through a seed point (x0, y0):

  * Streamline at instant t*:  a straight line of slope  V0 cos(omega t*)/U0.
  * Pathline of the particle released from (x0,y0) at t=0:
        x(t) = x0 + U0 t ,   y(t) = y0 + (V0/omega) sin(omega t).
  * Streakline at observation time t_obs (dye injected continuously at the seed):
        x(tau) = x0 + U0 (t_obs - tau),
        y(tau) = y0 + (V0/omega)[ sin(omega t_obs) - sin(omega tau) ].

The program (i) draws the three curves at a chosen instant to show they differ,
and (ii) integrates the pathline with the classical fourth-order Runge-Kutta
method and confirms fourth-order convergence to the exact pathline.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

U0, V0, omega = 1.0, 1.0, 2.0
x0, y0 = 0.0, 0.0

def velocity(x, y, t):
    return U0, V0 * np.cos(omega * t)

def pathline_exact(t):
    return x0 + U0 * t, y0 + (V0 / omega) * np.sin(omega * t)

def rk4_pathline(dt, Tend):
    n = int(round(Tend / dt))
    x, y, t = x0, y0, 0.0
    for _ in range(n):
        u1, v1 = velocity(x, y, t)
        u2, v2 = velocity(x + 0.5*dt*u1, y + 0.5*dt*v1, t + 0.5*dt)
        u3, v3 = velocity(x + 0.5*dt*u2, y + 0.5*dt*v2, t + 0.5*dt)
        u4, v4 = velocity(x + dt*u3,     y + dt*v3,     t + dt)
        x += dt/6.0 * (u1 + 2*u2 + 2*u3 + u4)
        y += dt/6.0 * (v1 + 2*v2 + 2*v3 + v4)
        t += dt
    return x, y

print("Example 3.2  Streamlines, pathlines and streaklines of an unsteady flow\n")
print(f"  Field: u = {U0}, v = {V0} cos({omega} t)\n")

# --- RK4 convergence of the pathline to the exact solution ------------------
Tend = 3.0
xe, ye = pathline_exact(Tend)
print(f"  Pathline endpoint at T={Tend}: exact = ({xe:.8f}, {ye:.8f})")
print(f"\n  {'dt':>10} {'error':>14} {'order':>8}")
prev = None
errs = []
for dt in [0.2, 0.1, 0.05, 0.025, 0.0125]:
    xn, yn = rk4_pathline(dt, Tend)
    err = np.hypot(xn - xe, yn - ye)
    order = np.log(prev / err) / np.log(2.0) if prev else float("nan")
    print(f"  {dt:10.4f} {err:14.4e} {order:8.3f}")
    errs.append((dt, err)); prev = err

p = np.log(errs[-2][1] / errs[-1][1]) / np.log(errs[-2][0] / errs[-1][0])
print(f"\n  Observed order of the RK4 pathline integrator: p = {p:.3f}")
assert 3.8 < p < 4.2, "RK4 did not achieve fourth-order convergence"
print("  PASS: RK4 pathline converges at fourth order to the exact particle path.\n")

# --- confirm the three curves are distinct at the observation instant -------
t_obs = 1.1
# slopes/tangents at the seed for streamline and pathline
m_stream = V0 * np.cos(omega * t_obs) / U0
m_path0  = V0 * np.cos(0.0) / U0            # pathline tangent at release (t=0)
print(f"  At t_obs={t_obs}: streamline slope={m_stream:+.4f}, "
      f"pathline-launch slope={m_path0:+.4f} -> curves are distinct.\n")

# --- figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)

# three curves through the origin at observation time t_obs
s = np.linspace(0, 3, 200)
ax1.plot(x0 + s, y0 + m_stream * s, "C0-", lw=2, label=f"streamline @ t={t_obs}")
tp = np.linspace(0, t_obs, 200)
xp, yp = pathline_exact(tp)
ax1.plot(xp, yp, "C3--", lw=2, label="pathline (0 -> t_obs)")
tau = np.linspace(0, t_obs, 200)
xs = x0 + U0 * (t_obs - tau)
ys = y0 + (V0/omega) * (np.sin(omega*t_obs) - np.sin(omega*tau))
ax1.plot(xs, ys, "C2-.", lw=2, label="streakline @ t_obs")
ax1.plot(0, 0, "ko", ms=5)
ax1.set_xlabel("$x$ [m]"); ax1.set_ylabel("$y$ [m]")
ax1.set_title("Three curves differ in unsteady flow")
ax1.legend(frameon=False, fontsize=9); ax1.grid(alpha=0.3)

ds = np.array([e[0] for e in errs]); es = np.array([e[1] for e in errs])
ax2.loglog(ds, es, "o-", lw=1.8, label="RK4 error")
ax2.loglog(ds, es[-1]*(ds/ds[-1])**4, "k:", lw=1.4, label="slope 4 (reference)")
ax2.set_xlabel("time step $\\Delta t$ [s]"); ax2.set_ylabel("pathline endpoint error [m]")
ax2.set_title("RK4 pathline convergence")
ax2.legend(frameon=False); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Streamlines, pathlines and streaklines", y=1.05, fontsize=13)
fig.savefig("fig3_2_lines.png", dpi=150, bbox_inches="tight")
print("  Wrote fig3_2_lines.png")
