#!/usr/bin/env python3
"""
Example 4.1 -- Analytical: the draining-tank (Torricelli) problem.

A cylindrical tank of cross-sectional area A_t drains through a small orifice of
area A_o at its base.  Two integral conservation laws close the problem.  The
integral MASS balance for the control volume enclosing the liquid gives

        A_t dh/dt = -A_o v_jet ,

and BERNOULLI's equation from the free surface to the jet (both at atmospheric
pressure), with A_o << A_t so the surface speed is negligible, gives Torricelli's
law v_jet = sqrt(2 g h).  Together,

        dh/dt = -(A_o/A_t) sqrt(2 g h) ,

which integrates in closed form to

        h(t) = ( sqrt(h0) - (A_o/A_t) sqrt(g/2) t )^2 ,   t <= t_drain ,
        t_drain = (A_t/A_o) sqrt(2 h0 / g) .

The program evaluates the closed-form level and drain time and verifies them
against a direct fourth-order Runge-Kutta integration of the mass-balance ODE.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g   = 9.81
D_t = 1.0                      # tank diameter [m]
d_o = 0.05                     # orifice diameter [m]
A_t = np.pi * D_t**2 / 4.0
A_o = np.pi * d_o**2 / 4.0
h0  = 2.0                      # initial level [m]
k   = (A_o / A_t) * np.sqrt(2*g)

def h_exact(t):
    root = np.sqrt(h0) - 0.5 * k * t
    return np.where(root > 0, root**2, 0.0)

t_drain = np.sqrt(h0) / (0.5 * k)

def rk4(dt):
    """Integrate dh/dt = -k sqrt(h) until the tank empties; return (t, h) arrays."""
    ts = [0.0]; hs = [h0]
    t, h = 0.0, h0
    def f(h):
        return -k * np.sqrt(max(h, 0.0))
    while h > 1e-12 and t < 2*t_drain:
        k1 = f(h); k2 = f(h + 0.5*dt*k1)
        k3 = f(h + 0.5*dt*k2); k4 = f(h + dt*k3)
        h = h + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        ts.append(t); hs.append(max(h, 0.0))
    return np.array(ts), np.array(hs)

print("Example 4.1  Torricelli draining tank -- analytical vs numerical\n")
print(f"  Tank D={D_t} m, orifice d={d_o} m, A_o/A_t = {A_o/A_t:.4e}")
print(f"  Initial level h0 = {h0} m")
print(f"  Analytical drain time t_drain = {t_drain:.3f} s\n")

# verify the closed-form level against RK4 at several sample times
dt = 0.02
t_rk, h_rk = rk4(dt)
sample = [0.25, 0.5, 0.75, 0.95]
print(f"  {'t/t_drain':>10} {'h_exact [m]':>13} {'h_RK4 [m]':>12} {'|err| [m]':>12}")
worst = 0.0
for frac in sample:
    ts = frac * t_drain
    he = h_exact(ts)
    hr = np.interp(ts, t_rk, h_rk)
    e = abs(he - hr)
    worst = max(worst, e)
    print(f"  {frac:10.2f} {he:13.6f} {hr:12.6f} {e:12.3e}")

# drain time from RK4 (first time level reaches ~0)
idx = np.argmax(h_rk <= 1e-6)
t_drain_rk = t_rk[idx] if idx > 0 else t_rk[-1]
print(f"\n  Drain time: analytical {t_drain:.3f} s, RK4 {t_drain_rk:.3f} s")
print(f"  Worst level discrepancy over the drain: {worst:.3e} m")
assert worst < 1e-3, "closed-form level disagrees with the ODE integration"
assert abs(t_drain_rk - t_drain)/t_drain < 0.01, "drain time disagrees"
print("  PASS: Torricelli level and drain time verified against RK4.\n")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
tt = np.linspace(0, t_drain, 300)
ax1.plot(tt, h_exact(tt), "k-", lw=2, label="analytical $h(t)$")
ax1.plot(t_rk[::25], h_rk[::25], "o", ms=5, mfc="none", mec="C3", label="RK4")
ax1.set_xlabel("time $t$ [s]"); ax1.set_ylabel("level $h(t)$ [m]")
ax1.set_title("Tank level during draining")
ax1.legend(frameon=False); ax1.grid(alpha=0.3)
vj = np.sqrt(2*g*h_exact(tt))
ax2.plot(tt, vj, "C0-", lw=2)
ax2.set_xlabel("time $t$ [s]"); ax2.set_ylabel("jet velocity $v=\\sqrt{2gh}$ [m/s]")
ax2.set_title("Jet velocity (Torricelli)")
ax2.grid(alpha=0.3)
fig.suptitle("Draining tank: integral mass balance + Bernoulli", y=1.04, fontsize=13)
fig.savefig("fig4_1_torricelli.png", dpi=150, bbox_inches="tight")
print("  Wrote fig4_1_torricelli.png")
