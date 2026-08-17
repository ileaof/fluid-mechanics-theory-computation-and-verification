#!/usr/bin/env python3
"""
Example 7.4 -- Analytical solution: stratified two-phase Poiseuille flow of two
immiscible Newtonian fluids between two stationary parallel plates.

This is the two-layer counterpart of the single-phase plane Poiseuille flow of
Chapter 1 and belongs with the internal fully developed viscous flows of Chapter
7.  Two immiscible fluids fill the gap between two infinite parallel plates a
distance H = 2h apart; the lower fluid (fluid 1) occupies 0 <= y <= h1 and the
upper fluid (fluid 2) occupies h1 <= y <= H, with a flat interface at y = h1.
Both plates are STATIONARY (no slip on each wall).  The flow is steady, laminar,
fully developed and incompressible, driven by a constant pressure gradient dp/dx
(a gravity term is included so an inclined channel can also be treated).

Canonical problem solved by the default parameters:
    two fluids of equal thickness h = 2.5 mm between fixed plates (H = 5 mm);
    the upper fluid is twice as viscous as the lower one,
        mu_lower = 0.5 Pa*s ,  mu_upper = 1.0 Pa*s ;
    applied pressure gradient dp/dx = -1000 Pa/m.
    -> interface velocity  u(h1) ~ 4.167 mm/s
    -> maximum velocity    u_max ~ 4.340 mm/s, in the LESS viscous (lower) fluid.

For each fluid i the fully developed x-momentum equation reduces to the
constant-coefficient ODE

        mu_i d^2 u_i/dy^2 = dp/dx - rho_i g sin(theta) ,        (i = 1, 2)

so u_i'' = a_i is constant in each layer, with the driving term and curvature

        D_i = dp/dx - rho_i g sin(theta),        a_i = D_i / mu_i .

The layer solution is the quadratic  u_i(y) = 0.5 a_i y^2 + b_i y + c_i, i.e.
each layer is a PARABOLA.  The four constants (b_1, c_1, b_2, c_2) are fixed by

        u_1(0)   = 0                                 (no slip, lower plate)
        u_2(H)   = 0                                 (no slip, upper plate)
        u_1(h1)  = u_2(h1)                           (velocity continuity)
        mu_1 u_1'(h1) = mu_2 u_2'(h1)                (shear continuity)

which form a 4x4 linear system solved below.  Because the two parabolas have
different curvatures (a_i = D_i/mu_i) they meet at the interface with a SLOPE
KINK set by shear continuity, du_2/dy = (mu_1/mu_2) du_1/dy; the more viscous
fluid shears less, so the velocity maximum is displaced toward the LESS viscous
layer rather than sitting at the mid-gap.

The program derives and evaluates the profiles; reports the interfacial and
maximum velocities and their locations, the shear stress on each plate, the
volumetric flow rate per unit width of each layer and the total, the sign
convention, the flow direction of each fluid and a reverse-flow check; and runs
a verification section (the four boundary/interface conditions, an overall force
balance, numerical quadrature and four limiting cases).  Every quantity is in
coherent SI units.  Only numpy and matplotlib are used; no random numbers.

Setting rho_2 = rho_1 and mu_2 = mu_1 recovers the single-phase plane Poiseuille
flow between fixed plates (a symmetric parabola with its maximum at the mid-gap),
so the solver composes cleanly with the single-phase solver of Example 1.1.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ===========================================================================
# User-specified physical parameters (all coherent SI units)
# ===========================================================================
mu1 = 0.5             # lower-fluid dynamic viscosity     [Pa*s]
mu2 = 1.0             # upper-fluid dynamic viscosity      [Pa*s]  (= 2*mu1)
h1  = 2.5e-3          # lower-layer thickness             [m]
h2  = 2.5e-3          # upper-layer thickness             [m]
dpdx = -1000.0        # applied pressure gradient dp/dx   [Pa/m]

# Optional body force for an INCLINED channel (defaults make it inactive: a
# horizontal channel driven by pressure alone, as in the canonical problem).
rho1 = 1000.0         # lower-fluid density               [kg/m^3]
rho2 = 850.0          # upper-fluid density               [kg/m^3]
theta_deg = 0.0       # channel inclination from horizontal [deg]
g = 9.81              # gravitational acceleration         [m/s^2]

# Sign convention:
#   x is the streamwise coordinate (down-channel); y is measured across the gap
#   from the lower plate (y = 0) to the upper plate (y = H).  u > 0 is flow in
#   the +x direction.  A favourable pressure gradient is dp/dx < 0 and drives
#   +x flow; for an inclined channel, gravity drives +x flow for theta > 0.

H = h1 + h2                          # plate separation (= 2h if equal)   [m]
theta = np.deg2rad(theta_deg)

# Driving terms D_i = dp/dx - rho_i g sin(theta) and curvatures a_i = D_i/mu_i
D1 = dpdx - rho1 * g * np.sin(theta)
D2 = dpdx - rho2 * g * np.sin(theta)
a1 = D1 / mu1
a2 = D2 / mu2


# ===========================================================================
# Solve the 4x4 linear system for the integration constants [b1, c1, b2, c2]
# ===========================================================================
#   row 1 : u1(0)  = 0                 ->            c1            = 0
#   row 2 : u2(H)  = 0                 ->      H b2 + c2 = -0.5 a2 H^2
#   row 3 : u1(h1) = u2(h1)            -> h1 b1 + c1 - h1 b2 - c2 = 0.5(a2-a1) h1^2
#   row 4 : mu1 u1'(h1) = mu2 u2'(h1)  ->   mu1 b1 - mu2 b2       = h1 (D2 - D1)
A_sys = np.array([
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0,  H,   1.0],
    [h1,  1.0, -h1, -1.0],
    [mu1, 0.0, -mu2, 0.0],
])
rhs = np.array([
    0.0,
    -0.5 * a2 * H**2,
    0.5 * (a2 - a1) * h1**2,
    h1 * (D2 - D1),
])
b1, c1, b2, c2 = np.linalg.solve(A_sys, rhs)


# ===========================================================================
# Analytical piecewise velocity and shear-stress profiles
# ===========================================================================
def u_lower(y):
    """Lower-fluid velocity u_1(y) [m/s] for 0 <= y <= h1."""
    return 0.5 * a1 * y**2 + b1 * y + c1


def u_upper(y):
    """Upper-fluid velocity u_2(y) [m/s] for h1 <= y <= H."""
    return 0.5 * a2 * y**2 + b2 * y + c2


def u_profile(y):
    """Piecewise velocity u(y) [m/s] across the gap (y scalar or array)."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= h1, u_lower(y), u_upper(y))


def tau_lower(y):
    """Lower-fluid shear stress tau_1 = mu_1 du_1/dy [Pa]."""
    return mu1 * (a1 * y + b1)


def tau_upper(y):
    """Upper-fluid shear stress tau_2 = mu_2 du_2/dy [Pa]."""
    return mu2 * (a2 * y + b2)


def tau_profile(y):
    """Piecewise shear stress tau(y) [Pa] across the gap."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= h1, tau_lower(y), tau_upper(y))


def flow_rate_lower():
    """Closed-form lower-fluid flow rate per unit width Q_1 [m^2/s]."""
    return a1 * h1**3 / 6.0 + b1 * h1**2 / 2.0 + c1 * h1


def flow_rate_upper():
    """Closed-form upper-fluid flow rate per unit width Q_2 [m^2/s]."""
    def prim(y):
        return a2 * y**3 / 6.0 + b2 * y**2 / 2.0 + c2 * y
    return prim(H) - prim(h1)


# ===========================================================================
# Derived engineering quantities
# ===========================================================================
u_interface = u_lower(h1)              # = u_upper(h1) by construction
tau_lower_wall = tau_lower(0.0)        # shear on the lower plate
tau_upper_wall = tau_upper(H)          # shear on the upper plate
tau_iface_1 = tau_lower(h1)            # interface shear, lower side
tau_iface_2 = tau_upper(h1)            # interface shear, upper side
Q1 = flow_rate_lower()
Q2 = flow_rate_upper()
Q_total = Q1 + Q2

# Maximum velocity and its location.  The extremum is where tau = 0
# (du/dy = 0); for a favourable pressure gradient with two fixed plates it lies
# inside the LESS viscous layer, displaced from the mid-gap.
y_scan = np.linspace(0.0, H, 200001)
u_scan = u_profile(y_scan)
imax = int(np.argmax(np.abs(u_scan)))          # extremum of |u| (handles either sign)
u_max, y_umax = float(u_scan[imax]), float(y_scan[imax])
layer_of_max = "lower fluid" if y_umax < h1 else "upper fluid"


def direction(u):
    """Report the flow direction implied by a signed velocity."""
    if u > 1e-14:
        return "+x"
    if u < -1e-14:
        return "-x"
    return "quiescent"


# ===========================================================================
# Console report
# ===========================================================================
print("Example 7.4  Two-phase Poiseuille flow of two immiscible fluids between "
      "stationary parallel plates")
print("=" * 82)
print("Configuration (coherent SI units):")
print(f"  lower fluid : mu_1 = {mu1:7.4f} Pa*s, h_1 = {h1*1e3:6.3f} mm, "
      f"rho_1 = {rho1:7.1f} kg/m^3")
print(f"  upper fluid : mu_2 = {mu2:7.4f} Pa*s, h_2 = {h2*1e3:6.3f} mm, "
      f"rho_2 = {rho2:7.1f} kg/m^3   (mu_2/mu_1 = {mu2/mu1:.2f})")
print(f"  gap H = {H*1e3:.3f} mm,  dp/dx = {dpdx:.3e} Pa/m,  theta = "
      f"{theta_deg:.1f} deg,  g = {g:.3f} m/s^2")
print(f"  driving terms:  D_1 = {D1:11.4e} Pa/m,  D_2 = {D2:11.4e} Pa/m")
print("\nSign convention: x is streamwise; u > 0 is +x. Favourable dp/dx < 0.\n")

print("Analytical coefficients  u_i(y) = 0.5 a_i y^2 + b_i y + c_i :")
print(f"  lower : a_1 = {a1:12.5e} 1/(m*s), b_1 = {b1:12.5e} 1/s, "
      f"c_1 = {c1:12.5e} m/s")
print(f"  upper : a_2 = {a2:12.5e} 1/(m*s), b_2 = {b2:12.5e} 1/s, "
      f"c_2 = {c2:12.5e} m/s\n")

print("Results:")
print(f"  interface velocity    u(h_1)       = {u_interface:12.5e} m/s "
      f"= {u_interface*1e3:8.4f} mm/s  [{direction(u_interface)}]")
print(f"  maximum velocity      u_max        = {u_max:12.5e} m/s "
      f"= {u_max*1e3:8.4f} mm/s  at y = {y_umax*1e3:6.4f} mm ({layer_of_max})")
print(f"  lower-plate shear     tau(0)       = {tau_lower_wall:12.5e} Pa")
print(f"  upper-plate shear     tau(H)       = {tau_upper_wall:12.5e} Pa")
print(f"  interface shear (lower side)       = {tau_iface_1:12.5e} Pa")
print(f"  interface shear (upper side)       = {tau_iface_2:12.5e} Pa")
print(f"  lower flow rate/width Q_1          = {Q1:12.5e} m^2/s")
print(f"  upper flow rate/width Q_2          = {Q2:12.5e} m^2/s")
print(f"  total flow rate/width Q_total      = {Q_total:12.5e} m^2/s")
print(f"  bulk mean velocity    Q_total/H    = {Q_total/H:12.5e} m/s "
      f"= {Q_total/H*1e3:8.4f} mm/s")

u1_mid = float(u_lower(0.5 * h1))
u2_mid = float(u_upper(0.5 * (h1 + H)))
print(f"\n  lower fluid moves in the {direction(u1_mid)} direction")
print(f"  upper fluid moves in the {direction(u2_mid)} direction")
sgn = np.sign(u_max) if u_max != 0 else 1.0
rev1 = bool(np.any(u_lower(np.linspace(0.0, h1, 2001)) * sgn < -1e-12))
rev2 = bool(np.any(u_upper(np.linspace(h1, H, 2001)) * sgn < -1e-12))
print(f"  reverse flow in lower fluid : {'YES' if rev1 else 'no'}")
print(f"  reverse flow in upper fluid : {'YES' if rev2 else 'no'}")


# ===========================================================================
# Verification section
# ===========================================================================
print("\n" + "=" * 82)
print("VERIFICATION")
print("=" * 82)

scale_u = max(abs(u_max), 1e-30)
scale_tau = max(abs(tau_lower_wall), abs(tau_upper_wall), 1e-30)
checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# 1. Boundary and interface conditions -------------------------------------
check("no slip, lower plate      u_1(0) = 0",
      abs(u_lower(0.0)) < 1e-12 * scale_u,
      f"|u_1(0)| = {abs(u_lower(0.0)):.2e} m/s")
check("no slip, upper plate      u_2(H) = 0",
      abs(u_upper(H)) < 1e-12 * scale_u,
      f"|u_2(H)| = {abs(u_upper(H)):.2e} m/s")
check("velocity continuity       u_1(h1) = u_2(h1)",
      abs(u_lower(h1) - u_upper(h1)) < 1e-12 * scale_u,
      f"jump = {abs(u_lower(h1) - u_upper(h1)):.2e} m/s")
check("shear continuity          mu_1 u_1'(h1) = mu_2 u_2'(h1)",
      abs(tau_iface_1 - tau_iface_2) < 1e-10 * scale_tau,
      f"jump = {abs(tau_iface_1 - tau_iface_2):.2e} Pa")

# 2. Overall force balance --------------------------------------------------
# Steady fully developed flow: the net pressure/body force per unit width is
# balanced by the two wall shears, tau_2(H) - tau_1(0) = D_1 h1 + D_2 h2.
force_lhs = tau_upper_wall - tau_lower_wall
force_rhs = D1 * h1 + D2 * h2
check("global x-force balance     tau(H) - tau(0) = D_1 h1 + D_2 h2",
      abs(force_lhs - force_rhs) < 1e-9 * scale_tau,
      f"residual = {abs(force_lhs - force_rhs):.2e} Pa")

# 3. Flow rates vs numerical quadrature ------------------------------------
yl = np.linspace(0.0, h1, 8001)
yu = np.linspace(h1, H, 8001)
check("lower flow rate  Q_1 vs quadrature",
      abs(Q1 - trapezoid(u_lower(yl), yl)) < 1e-6 * (abs(Q1) + 1e-30),
      f"rel.err = {abs(Q1 - trapezoid(u_lower(yl), yl))/(abs(Q1)+1e-30):.2e}")
check("upper flow rate  Q_2 vs quadrature",
      abs(Q2 - trapezoid(u_upper(yu), yu)) < 1e-6 * (abs(Q2) + 1e-30),
      f"rel.err = {abs(Q2 - trapezoid(u_upper(yu), yu))/(abs(Q2)+1e-30):.2e}")


# 4. Limiting cases ---------------------------------------------------------
def solve_two_plate(m1, m2, hh1, hh2, dp, r1, r2, th):
    """Re-solve the four constants for arbitrary inputs (used by the checks)."""
    Ht = hh1 + hh2
    Dd1 = dp - r1 * g * np.sin(th)
    Dd2 = dp - r2 * g * np.sin(th)
    aa1, aa2 = Dd1 / m1, Dd2 / m2
    M = np.array([[0., 1., 0., 0.],
                  [0., 0., Ht, 1.],
                  [hh1, 1., -hh1, -1.],
                  [m1, 0., -m2, 0.]])
    r = np.array([0.0, -0.5*aa2*Ht**2, 0.5*(aa2-aa1)*hh1**2, hh1*(Dd2-Dd1)])
    bb1, cc1, bb2, cc2 = np.linalg.solve(M, r)
    return aa1, bb1, cc1, aa2, bb2, cc2, Ht


# (a) Equal viscosities and no gravity -> single-fluid plane Poiseuille between
#     fixed plates:  u(y) = (dp/dx)/(2 mu) (y^2 - H y),  max at y = H/2.
aa1, bb1, cc1, aa2, bb2, cc2, Ht = solve_two_plate(mu1, mu1, h1, h2, dpdx,
                                                   rho1, rho1, 0.0)
yv = np.linspace(0.0, Ht, 4001)
u_two = np.where(yv <= h1, 0.5*aa1*yv**2 + bb1*yv + cc1,
                 0.5*aa2*yv**2 + bb2*yv + cc2)
u_one = (dpdx / (2.0 * mu1)) * (yv**2 - Ht * yv)
check("limiting case: equal mu -> single-fluid plane Poiseuille",
      np.max(np.abs(u_two - u_one)) < 1e-12 * (np.max(np.abs(u_one)) + 1e-30),
      f"max|u_2layer - u_1layer| = {np.max(np.abs(u_two - u_one)):.2e} m/s")

# (b) Symmetric case: equal mu AND equal thickness -> maximum at the mid-gap.
yfine = np.linspace(0.0, Ht, 200001)
uu = np.where(yfine <= h1, 0.5*aa1*yfine**2 + bb1*yfine + cc1,
              0.5*aa2*yfine**2 + bb2*yfine + cc2)
y_at_max = yfine[int(np.argmax(np.abs(uu)))]
check("limiting case: equal mu & equal h -> max at mid-gap y = H/2",
      abs(y_at_max - 0.5 * Ht) < 2.0 * (Ht / 200000),
      f"y_max = {y_at_max*1e3:.4f} mm vs H/2 = {0.5*Ht*1e3:.4f} mm")

# (c) Zero pressure gradient, horizontal -> quiescent fluid (u == 0).
aa1, bb1, cc1, aa2, bb2, cc2, Ht = solve_two_plate(mu1, mu2, h1, h2, 0.0,
                                                   rho1, rho2, 0.0)
yv = np.linspace(0.0, Ht, 2001)
u_rest = np.where(yv <= h1, 0.5*aa1*yv**2 + bb1*yv + cc1,
                  0.5*aa2*yv**2 + bb2*yv + cc2)
check("limiting case: dp/dx = 0, horizontal -> fluid at rest (u = 0)",
      np.max(np.abs(u_rest)) < 1e-14,
      f"max|u| = {np.max(np.abs(u_rest)):.2e} m/s")

# (d) Vanishing upper layer (h2 -> 0) -> single lower fluid Poiseuille over H=h1.
aa1, bb1, cc1, aa2, bb2, cc2, Ht = solve_two_plate(mu1, mu2, h1, 1e-12, dpdx,
                                                   rho1, rho2, 0.0)
yv = np.linspace(0.0, h1, 4001)
u_thin = 0.5*aa1*yv**2 + bb1*yv + cc1
u_single = (dpdx / (2.0 * mu1)) * (yv**2 - h1 * yv)
check("limiting case: h2 -> 0 -> single lower-fluid Poiseuille of gap h1",
      np.max(np.abs(u_thin - u_single)) < 1e-6 * (np.max(np.abs(u_single)) + 1e-30),
      f"max|diff| = {np.max(np.abs(u_thin - u_single)):.2e} m/s")

print("\n" + "-" * 82)
if all(checks):
    print(f"  ALL {len(checks)} CHECKS PASSED -- the analytical two-phase "
          "solution is verified.")
else:
    print(f"  {checks.count(False)} of {len(checks)} CHECKS FAILED.")
assert all(checks), "two-phase analytical solution failed its verification"


# ===========================================================================
# Plots:  velocity profile (both layers, coloured) + shear-stress distribution
# ===========================================================================
yl = np.linspace(0.0, h1, 400)
yu = np.linspace(h1, H, 400)
ul, uu = u_lower(yl), u_upper(yu)
tl, tu = tau_lower(yl), tau_upper(yu)

LOW_C, UP_C = "#1f77b4", "#d4a017"     # lower (blue), upper (amber)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.2),
                               constrained_layout=True)

# -- velocity profile --
ax1.plot(ul * 1e3, yl * 1e3, color=LOW_C, lw=2.6, label="lower fluid  $u_1(y)$")
ax1.plot(uu * 1e3, yu * 1e3, color=UP_C, lw=2.6, label="upper fluid  $u_2(y)$")
ax1.axhspan(0.0, h1 * 1e3, color=LOW_C, alpha=0.08)
ax1.axhspan(h1 * 1e3, H * 1e3, color=UP_C, alpha=0.10)
ax1.axhline(h1 * 1e3, color="0.35", lw=1.2, ls="--", label="interface")
# stationary plates (no-slip walls)
ax1.axhline(0.0, color="k", lw=3.0)
ax1.axhline(H * 1e3, color="k", lw=3.0)
ax1.plot(u_interface * 1e3, h1 * 1e3, "o", color="0.2", ms=6, zorder=5)
ax1.plot(u_max * 1e3, y_umax * 1e3, "*", color="crimson", ms=12, zorder=6,
         label="maximum")
ax1.axvline(0.0, color="0.7", lw=0.8)
ax1.set_xlabel("velocity  u (mm/s)")
ax1.set_ylabel("cross-gap coordinate  y (mm)")
ax1.set_title("Velocity profile (two-phase Poiseuille)")
# Anchor the legend a little below the top edge so the thick upper-plate line
# (drawn at y = H) does not cross the legend text.
ax1.legend(frameon=False, fontsize=9, loc="upper right",
           bbox_to_anchor=(1.0, 0.90))
ax1.grid(alpha=0.3)

# -- shear-stress distribution --
ax2.plot(tl, yl * 1e3, color=LOW_C, lw=2.6, label=r"lower  $\tau_1(y)$")
ax2.plot(tu, yu * 1e3, color=UP_C, lw=2.6, label=r"upper  $\tau_2(y)$")
ax2.axhline(h1 * 1e3, color="0.35", lw=1.2, ls="--")
ax2.axhline(0.0, color="k", lw=3.0)
ax2.axhline(H * 1e3, color="k", lw=3.0)
ax2.axvline(0.0, color="0.7", lw=0.8)
ax2.set_xlabel(r"shear stress  $\tau = \mu\,du/dy$  (Pa)")
ax2.set_ylabel("cross-gap coordinate  y (mm)")
ax2.set_title("Shear-stress distribution")
ax2.legend(frameon=False, fontsize=9, loc="upper right",
           bbox_to_anchor=(1.0, 0.90))
ax2.grid(alpha=0.3)

fig.suptitle("Stratified two-phase Poiseuille flow between stationary parallel "
             "plates", fontsize=13)
fig.savefig("fig7_4_two_phase.png", dpi=150, bbox_inches="tight")
print("\n  Wrote fig7_4_two_phase.png")
