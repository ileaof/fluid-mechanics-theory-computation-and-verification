#!/usr/bin/env python3
"""
Example 2.2 -- Numerical integration of the hydrostatic balance in a
compressible atmosphere (finite-volume, second order).

The hydrostatic equation dp/dz = -rho g, closed with the ideal-gas law
rho = p/(R T), governs the pressure in a still atmosphere.  In the troposphere
the temperature falls linearly with altitude,

        T(z) = T0 - L z ,          L = 0.0065 K/m  (lapse rate),

for which the balance has the exact solution (the International Standard
Atmosphere)

        p(z) = p0 [ T(z)/T0 ]^{ g / (R L) } .

Because rho depends on p, the discrete balance is integrated cell by cell with
the trapezoidal (Crank-Nicolson) face rule -- a conservative, second-order
finite-volume march:

        p_{i+1} [ 1 + (dz/2) g/(R T_{i+1}) ] = p_i [ 1 - (dz/2) g/(R T_i) ] .

The program marches the discrete balance on a sequence of refined grids and
measures the observed order of accuracy against the exact ISA solution.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g   = 9.80665       # m/s^2
R   = 287.05        # J/(kg K), dry air
T0  = 288.15        # K, sea level
L   = 0.0065        # K/m, tropospheric lapse rate
p0  = 101325.0      # Pa, sea level
Ztop = 11000.0      # m, top of the troposphere

def T_of_z(z):
    return T0 - L * z

def p_exact(z):
    return p0 * (T_of_z(z) / T0) ** (g / (R * L))

def march_fv(N):
    """Second-order trapezoidal finite-volume march of dp/dz = -(g/(R T)) p."""
    z = np.linspace(0.0, Ztop, N + 1)
    dz = z[1] - z[0]
    p = np.empty(N + 1)
    p[0] = p0
    for i in range(N):
        a_i  = 0.5 * dz * g / (R * T_of_z(z[i]))
        a_ip = 0.5 * dz * g / (R * T_of_z(z[i + 1]))
        p[i + 1] = p[i] * (1.0 - a_i) / (1.0 + a_ip)
    return z, p

print("Example 2.2  Hydrostatic balance in the ISA troposphere (finite volume)\n")
print(f"  Exact model: p(z) = p0 (1 - L z/T0)^(g/RL),  g/RL = {g/(R*L):.4f}")
print(f"  p({Ztop:.0f} m) exact = {p_exact(Ztop):.3f} Pa "
      f"({p_exact(Ztop)/p0*100:.2f} % of sea-level)\n")
print(f"  {'N':>6} {'dz [m]':>10} {'max|err| [Pa]':>15} {'order p':>9}")

Ns = [20, 40, 80, 160, 320, 640]
prev = None
errs = []
for N in Ns:
    z, p = march_fv(N)
    err = np.max(np.abs(p - p_exact(z)))
    order = np.log(prev / err) / np.log(2.0) if prev else float("nan")
    print(f"  {N:6d} {Ztop/N:10.2f} {err:15.4e} {order:9.3f}")
    errs.append((Ztop / N, err)); prev = err

p_obs = np.log(errs[-2][1] / errs[-1][1]) / np.log(errs[-2][0] / errs[-1][0])
print(f"\n  Observed order of accuracy (finest pair): p = {p_obs:.3f}")
assert 1.9 < p_obs < 2.1, "finite-volume march is not second order"
print("  PASS: the finite-volume march converges at second order to the ISA.\n")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
z, p = march_fv(40)
zf = np.linspace(0, Ztop, 300)
ax1.plot(p_exact(zf)/1e3, zf/1e3, "k-", lw=2, label="exact ISA")
ax1.plot(p/1e3, z/1e3, "o", ms=5, mfc="none", mec="C3", label="finite volume (N=40)")
ax1b = ax1.twiny()
ax1b.plot(T_of_z(zf), zf/1e3, "C0-.", lw=1.5)
ax1b.set_xlabel("temperature $T(z)$ [K]", color="C0")
ax1b.tick_params(axis="x", colors="C0")
ax1.set_xlabel("pressure $p(z)$ [kPa]"); ax1.set_ylabel("altitude $z$ [km]")
ax1.set_title("ISA pressure and temperature")
ax1.legend(frameon=False, loc="upper right", fontsize=9); ax1.grid(alpha=0.3)

hs = np.array([e[0] for e in errs]); es = np.array([e[1] for e in errs])
ax2.loglog(hs, es, "o-", lw=1.8, label="max error")
ax2.loglog(hs, es[-1]*(hs/hs[-1])**2, "k:", lw=1.4, label="slope 2 (reference)")
ax2.set_xlabel("mesh spacing $\\Delta z$ [m]"); ax2.set_ylabel("max$|p-p_{exact}|$ [Pa]")
ax2.set_title("Grid convergence"); ax2.legend(frameon=False); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Finite-volume solution of the hydrostatic balance", y=1.05, fontsize=13)
fig.savefig("fig2_2_atmos.png", dpi=150, bbox_inches="tight")
print("  Wrote fig2_2_atmos.png")
