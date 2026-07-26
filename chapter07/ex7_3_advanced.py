#!/usr/bin/env python3
"""
Example 7.3 -- Advanced engineering application: the three-reservoir pipe network
and a pump-system operating point.

Three reservoirs at elevations z1, z2, z3 are joined at a common junction J by
pipes 1, 2, 3.  The unknown is the junction head H_J.  For each pipe the head loss
equals the elevation difference to the junction,

        z_k - H_J = sign(Q_k) * ( f_k L_k/D_k + sum K ) * Q_k^2 / (2 g A_k^2),

with the Darcy friction factor f_k from the Colebrook equation (turbulent) or
64/Re (laminar).  Continuity at the junction requires the net flow to vanish,
sum_k Q_k(H_J) = 0, a single nonlinear equation solved here for H_J by a bracketed
root finder, with the friction factors updated self-consistently.

The program (i) solves the network and verifies that continuity closes to machine
precision; (ii) checks a symmetric limiting case with a known answer; and
(iii) finds the operating point of a pump feeding a pipe system by intersecting the
pump curve with the system curve.  Uses numpy, matplotlib, and
scipy.optimize.brentq (via the bundled shim).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

g, rho, nu, mu = 9.81, 1000.0, 1.0e-6, 1.0e-3

def colebrook(Re, eps_D):
    if Re < 2300:
        return 64.0/max(Re, 1e-6)
    return brentq(lambda f: 1/np.sqrt(f) + 2*np.log10(eps_D/3.7 + 2.51/(Re*np.sqrt(f))),
                  1e-4, 1.0, xtol=1e-12)

class Pipe:
    def __init__(self, z, L, D, eps=4.5e-5, Kminor=0.5):
        self.z, self.L, self.D, self.eps, self.K = z, L, D, eps, Kminor
        self.A = np.pi*D**2/4

def flow_to_junction(pipe, HJ):
    """Signed flow Q from reservoir (head z) toward junction head HJ."""
    dz = pipe.z - HJ
    if abs(dz) < 1e-14:
        return 0.0
    # iterate f <-> Q (fixed point) since f depends on Re(Q)
    Q = np.sign(dz)*1e-3
    for _ in range(100):
        V = abs(Q)/pipe.A
        Re = V*pipe.D/nu
        f = colebrook(Re, pipe.eps/pipe.D)
        coeff = (f*pipe.L/pipe.D + pipe.K)/(2*g*pipe.A**2)
        Qnew = np.sign(dz)*np.sqrt(abs(dz)/coeff)
        if abs(Qnew-Q) < 1e-12: Q = Qnew; break
        Q = Qnew
    return Q

def net_flow(pipes, HJ):
    return sum(flow_to_junction(p, HJ) for p in pipes)

print("Example 7.3  Three-reservoir network and pump operating point\n")
pipes = [Pipe(z=100.0, L=1000.0, D=0.30),
         Pipe(z= 80.0, L=1200.0, D=0.25),
         Pipe(z= 60.0, L= 800.0, D=0.20)]
lo, hi = 60.0+1e-6, 100.0-1e-6
t0 = time.perf_counter()
HJ = brentq(lambda h: net_flow(pipes, h), lo, hi, xtol=1e-10)
cpu = (time.perf_counter()-t0)*1e3
Qs = [flow_to_junction(p, HJ) for p in pipes]
print(f"  Junction head H_J = {HJ:.4f} m   (solved in {cpu:.1f} ms)")
for k, (p, Q) in enumerate(zip(pipes, Qs), 1):
    V = abs(Q)/p.A; Re = V*p.D/nu
    dirn = "into J" if Q > 0 else "out of J"
    print(f"    pipe {k}: z={p.z:5.0f} m  Q={Q*1e3:+7.2f} L/s  V={V:.3f} m/s  Re={Re:6.0f}  ({dirn})")
residual = sum(Qs)
print(f"\n  Continuity residual sum(Q) = {residual:.2e} m^3/s")
assert abs(residual) < 1e-8, "junction continuity did not close"
print("  PASS: junction continuity closes to machine precision.\n")

# --- limiting case: three identical reservoirs at equal elevation -> zero flow --
print("  Limiting case: three identical reservoirs at equal elevation")
eq = [Pipe(z=50.0, L=1000.0, D=0.3) for _ in range(3)]
HJ2 = brentq(lambda h: net_flow(eq, h), 50.0-5, 50.0+5, xtol=1e-12)
Qs2 = [flow_to_junction(p, HJ2) for p in eq]
print(f"    H_J = {HJ2:.6f} m (expect 50), max|Q| = {max(abs(q) for q in Qs2):.2e} (expect 0)")
assert abs(HJ2-50.0) < 1e-4 and max(abs(q) for q in Qs2) < 1e-6
print("    PASS: symmetric case gives H_J = 50 m and zero flow.\n")

# --- pump-system operating point --------------------------------------------
# pump head curve H_p(Q) = H0 - a Q^2 ; system curve H_s(Q) = dz_static + b Q^2
H0, a_pump = 50.0, 2.5e5          # pump: shutoff head 50 m
dz_static, b_sys = 20.0, 1.8e5    # static lift 20 m + friction/minor losses
def pump(Q):   return H0 - a_pump*Q**2
def system(Q): return dz_static + b_sys*Q**2
Qop = brentq(lambda Q: pump(Q)-system(Q), 1e-4, 0.02, xtol=1e-12)
Hop = pump(Qop)
print("  Pump-system operating point (pump curve meets system curve):")
print(f"    Q_op = {Qop*1e3:.3f} L/s,  H_op = {Hop:.3f} m")
print(f"    check |H_pump - H_system| at Q_op = {abs(pump(Qop)-system(Qop)):.2e} m")
assert abs(pump(Qop)-system(Qop)) < 1e-8
print("  PASS: operating point balances pump and system heads.\n")

# --- figures ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
# network schematic
ax1.set_title("Three-reservoir network")
coords = [(-1.0, 1.0), (1.0, 0.9), (0.0, -1.0)]
Jx, Jy = 0.0, 0.15
for (x, y), p, Q in zip(coords, pipes, Qs):
    ax1.plot([x, Jx], [y, Jy], "C0-", lw=2)
    ax1.plot(x, y, "s", ms=16, color="C0")
    ax1.text(x, y+0.18, f"z={p.z:.0f} m", ha="center", fontsize=9)
    xm, ym = (x+Jx)/2, (y+Jy)/2
    ax1.annotate("", xy=((x+Jx)/2+0.12*np.sign(Jx-x) if Q>0 else x+0.6*(Jx-x),
                         (y+Jy)/2), xytext=(xm, ym))
    ax1.text(xm, ym+0.06, f"{abs(Q)*1e3:.1f} L/s", ha="center", fontsize=8, color="C3")
ax1.plot(Jx, Jy, "o", ms=12, color="C3"); ax1.text(Jx+0.08, Jy, f"J: {HJ:.1f} m", fontsize=9)
ax1.set_xlim(-1.6, 1.6); ax1.set_ylim(-1.5, 1.5); ax1.axis("off")
# pump/system curves
Qg = np.linspace(0, 0.02, 200)
ax2.plot(Qg*1e3, pump(Qg), "C0-", lw=2, label="pump curve")
ax2.plot(Qg*1e3, system(Qg), "C3-", lw=2, label="system curve")
ax2.plot(Qop*1e3, Hop, "ko", ms=8); ax2.text(Qop*1e3, Hop+2, "operating point", ha="center")
ax2.set_xlabel("flow rate $Q$ [L/s]"); ax2.set_ylabel("head $H$ [m]")
ax2.set_title("Pump-system operating point"); ax2.legend(frameon=False); ax2.grid(alpha=0.3)
fig.suptitle("Pipe network analysis and pump matching", y=1.04, fontsize=13)
fig.savefig("fig7_3_network.png", dpi=150, bbox_inches="tight")
print("  Wrote fig7_3_network.png")
