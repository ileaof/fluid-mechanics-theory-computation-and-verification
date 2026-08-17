#!/usr/bin/env python3
"""
Example 7.4 -- Analytical solution: stratified two-phase flow of two immiscible
Newtonian fluids (water under oil) over a stationary inclined plate with a
free upper surface.

This viscous-flow example extends the internal fully developed flows of Chapter
7 to two immiscible layers; it is the two-layer counterpart of the single-phase
plane Couette-Poiseuille flow of Example 1.1.  Steady, laminar, fully developed,
incompressible flow of
two immiscible Newtonian fluids is driven along the x-direction by a pressure
gradient dp/dx, by gravity along a plate inclined at angle theta, or by both:

    * a stationary lower plate at y = 0  (no slip);
    * a WATER layer occupying 0 <= y <= h_w;
    * an OIL layer occupying   h_w <= y <= H   (H = h_w + h_o);
    * a flat fluid-fluid interface at y = h_w;
    * a shear-free free surface at y = H.

Assumptions: incompressible fluids, constant properties, no slip at the wall,
no interfacial mass transfer, a flat (non-deforming) interface and free surface,
and no Marangoni stress.  For each fluid i in {w, o} the fully developed
x-momentum equation reduces to the constant-coefficient ODE

        mu_i d^2 u_i/dy^2 = dp/dx - rho_i g sin(theta) ,        (i = w, o)

so u_i'' = a_i is constant in each layer, with the driving term

        D_i = dp/dx - rho_i g sin(theta),        a_i = D_i / mu_i .

The general layer solution is the quadratic

        u_i(y) = 0.5 a_i y^2 + b_i y + c_i .

The four constants (b_w, c_w, b_o, c_o) are fixed by the four conditions

        u_w(0)      = 0                              (no slip, lower wall)
        u_w(h_w)    = u_o(h_w)                       (velocity continuity)
        mu_w u_w'(h_w) = mu_o u_o'(h_w)              (shear continuity)
        u_o'(H)     = 0                              (shear-free free surface)

giving the closed form (see the derivation in the code below)

        c_w = 0
        b_w = -(D_w h_w + D_o h_o) / mu_w        (= tau_wall / mu_w)
        b_o = -a_o H
        c_o = 0.5 (a_w - a_o) h_w^2 + (b_w - b_o) h_w .

The shear stress tau_i(y) = mu_i u_i'(y) is piecewise LINEAR, continuous at the
interface, and vanishes at the free surface:

        tau_w(y) = D_w y + mu_w b_w ,     tau_o(y) = D_o (y - H) .

The program derives and evaluates the profiles, reports the interfacial and
maximum velocities and their locations, the wall shear stress, the flow rate
per unit width of each layer and the total, states the sign convention and the
flow direction of each fluid, checks for reverse flow, and runs a verification
section (boundary/interface conditions, an overall force balance, numerical
quadrature and four limiting cases).  Every quantity is in coherent SI units.
Only numpy and matplotlib are used; no random numbers are involved.

The layer solver is written so it can later be composed with the single-phase
Couette/Poiseuille solver of Example 1.1: setting rho_o = rho_w, mu_o = mu_w
recovers a single homogeneous fluid between the wall and the free surface.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ===========================================================================
# User-specified physical parameters (all coherent SI units)
# ===========================================================================
rho_w = 1000.0        # water density                     [kg/m^3]
rho_o = 850.0         # oil   density                     [kg/m^3]
mu_w  = 1.0e-3        # water dynamic viscosity           [Pa*s]
mu_o  = 8.0e-2        # oil   dynamic viscosity           [Pa*s]
h_w   = 1.0e-3        # water-layer thickness             [m]
h_o   = 1.5e-3        # oil-layer thickness               [m]
dpdx  = 0.0           # imposed pressure gradient dp/dx   [Pa/m]
theta_deg = 5.0       # plate inclination from horizontal [deg]
g     = 9.81          # gravitational acceleration        [m/s^2]

# Sign convention:
#   x points ALONG the plate in the down-slope direction; y is the wall-normal
#   coordinate measured from the plate (y = 0) toward the free surface (y = H).
#   u > 0 means motion in the +x (down-slope) direction.  A favourable pressure
#   gradient is dp/dx < 0; gravity drives +x flow for theta > 0.

H = h_w + h_o                       # total film thickness           [m]
theta = np.deg2rad(theta_deg)       # inclination                    [rad]

# Driving terms D_i = dp/dx - rho_i g sin(theta)  and curvatures a_i = D_i/mu_i
D_w = dpdx - rho_w * g * np.sin(theta)
D_o = dpdx - rho_o * g * np.sin(theta)
a_w = D_w / mu_w
a_o = D_o / mu_o


# ===========================================================================
# Analytical piecewise velocity profile and its derivatives
# ===========================================================================
# Constants of integration from the four boundary/interface conditions.
c_w = 0.0                                   # u_w(0) = 0
b_w = -(D_w * h_w + D_o * h_o) / mu_w       # shear continuity + shear-free top
b_o = -a_o * H                              # u_o'(H) = 0
c_o = 0.5 * (a_w - a_o) * h_w**2 + (b_w - b_o) * h_w   # velocity continuity


def u_water(y):
    """Water-layer velocity u_w(y) [m/s] for 0 <= y <= h_w."""
    return 0.5 * a_w * y**2 + b_w * y + c_w


def u_oil(y):
    """Oil-layer velocity u_o(y) [m/s] for h_w <= y <= H."""
    return 0.5 * a_o * y**2 + b_o * y + c_o


def u_profile(y):
    """Piecewise velocity u(y) [m/s] across both layers (y scalar or array)."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= h_w, u_water(y), u_oil(y))


def tau_water(y):
    """Water-layer shear stress tau_w = mu_w du_w/dy [Pa]."""
    return mu_w * (a_w * y + b_w)


def tau_oil(y):
    """Oil-layer shear stress tau_o = mu_o du_o/dy [Pa]."""
    return mu_o * (a_o * y + b_o)


def tau_profile(y):
    """Piecewise shear stress tau(y) [Pa] across both layers."""
    y = np.asarray(y, dtype=float)
    return np.where(y <= h_w, tau_water(y), tau_oil(y))


def flow_rate_water():
    """Closed-form water flow rate per unit width Q_w [m^2/s]."""
    return a_w * h_w**3 / 6.0 + b_w * h_w**2 / 2.0 + c_w * h_w


def flow_rate_oil():
    """Closed-form oil flow rate per unit width Q_o [m^2/s]."""
    def prim(y):                       # primitive of u_o(y)
        return a_o * y**3 / 6.0 + b_o * y**2 / 2.0 + c_o * y
    return prim(H) - prim(h_w)


# ===========================================================================
# Derived engineering quantities
# ===========================================================================
u_interface = u_water(h_w)             # = u_oil(h_w) by construction
tau_wall    = tau_water(0.0)           # mu_w u_w'(0)
tau_iface_w = tau_water(h_w)           # interface shear seen from water side
tau_iface_o = tau_oil(h_w)            # interface shear seen from oil side
Q_w = flow_rate_water()
Q_o = flow_rate_oil()
Q_total = Q_w + Q_o

# Maximum velocity and its location (scan the whole film; the extremum is where
# tau = 0, i.e. at the free surface for a monotone film, but a mixed
# pressure/gravity drive can place it inside a layer).
y_scan = np.linspace(0.0, H, 20001)
u_scan = u_profile(y_scan)
imax = int(np.argmax(u_scan))
u_max, y_umax = float(u_scan[imax]), float(y_scan[imax])
imin = int(np.argmin(u_scan))
u_min, y_umin = float(u_scan[imin]), float(y_scan[imin])


def direction(u):
    """Report the flow direction implied by a signed velocity."""
    if u > 1e-14:
        return "+x (down-slope)"
    if u < -1e-14:
        return "-x (up-slope)"
    return "quiescent"


# ===========================================================================
# Console report
# ===========================================================================
print("Example 7.4  Stratified two-phase (water/oil) flow over an inclined "
      "plate with a free surface")
print("=" * 78)
print("Configuration (coherent SI units):")
print(f"  water : rho = {rho_w:8.2f} kg/m^3, mu = {mu_w:9.3e} Pa*s, "
      f"h_w = {h_w*1e3:6.3f} mm")
print(f"  oil   : rho = {rho_o:8.2f} kg/m^3, mu = {mu_o:9.3e} Pa*s, "
      f"h_o = {h_o*1e3:6.3f} mm")
print(f"  H = {H*1e3:.3f} mm,  dp/dx = {dpdx:.3e} Pa/m,  theta = "
      f"{theta_deg:.1f} deg,  g = {g:.3f} m/s^2")
print(f"  driving terms:  D_w = {D_w:11.4e} Pa/m,  D_o = {D_o:11.4e} Pa/m")
print("\nSign convention: x is down-slope; u > 0 is +x flow. "
      "Favourable dp/dx < 0.\n")

print("Analytical coefficients  u_i(y) = 0.5 a_i y^2 + b_i y + c_i :")
print(f"  water : a_w = {a_w:12.5e} 1/(m*s), b_w = {b_w:12.5e} 1/s, "
      f"c_w = {c_w:12.5e} m/s")
print(f"  oil   : a_o = {a_o:12.5e} 1/(m*s), b_o = {b_o:12.5e} 1/s, "
      f"c_o = {c_o:12.5e} m/s\n")

print("Results:")
print(f"  interfacial velocity  u(h_w)        = {u_interface:12.5e} m/s "
      f"[{direction(u_interface)}]")
print(f"  maximum velocity      u_max         = {u_max:12.5e} m/s at "
      f"y = {y_umax*1e3:6.3f} mm  [{direction(u_max)}]")
print(f"  minimum velocity      u_min         = {u_min:12.5e} m/s at "
      f"y = {y_umin*1e3:6.3f} mm")
print(f"  wall shear stress     tau_wall      = {tau_wall:12.5e} Pa")
print(f"  interface shear (water side)        = {tau_iface_w:12.5e} Pa")
print(f"  interface shear (oil   side)        = {tau_iface_o:12.5e} Pa")
print(f"  free-surface shear    tau(H)        = {tau_oil(H):12.5e} Pa")
print(f"  water flow rate/width Q_w           = {Q_w:12.5e} m^2/s")
print(f"  oil   flow rate/width Q_o           = {Q_o:12.5e} m^2/s")
print(f"  total flow rate/width Q_total       = {Q_total:12.5e} m^2/s")
u_mean = Q_total / H
print(f"  bulk mean velocity    Q_total/H     = {u_mean:12.5e} m/s "
      f"[{direction(u_mean)}]")

# Flow-direction summary per layer (sample the mid-height of each layer).
u_w_mid = float(u_water(0.5 * h_w))
u_o_mid = float(u_oil(0.5 * (h_w + H)))
print(f"\n  water layer moves in the {direction(u_w_mid)} direction")
print(f"  oil   layer moves in the {direction(u_o_mid)} direction")

# Reverse-flow check (any sign opposite to the bulk mean within a layer).
rev_w = bool(np.any(u_water(np.linspace(0.0, h_w, 2001)) *
                    np.sign(u_mean) < -1e-12))
rev_o = bool(np.any(u_oil(np.linspace(h_w, H, 2001)) *
                    np.sign(u_mean) < -1e-12))
print(f"  reverse flow in water layer : {'YES' if rev_w else 'no'}")
print(f"  reverse flow in oil   layer : {'YES' if rev_o else 'no'}")


# ===========================================================================
# Verification section
# ===========================================================================
print("\n" + "=" * 78)
print("VERIFICATION")
print("=" * 78)

scale_u = max(abs(u_max), abs(u_min), 1e-30)
scale_tau = max(abs(tau_wall), 1e-30)
checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# 1. Boundary and interface conditions -------------------------------------
check("no slip at wall           u_w(0) = 0",
      abs(u_water(0.0)) < 1e-12 * scale_u,
      f"|u_w(0)| = {abs(u_water(0.0)):.2e} m/s")
check("velocity continuity       u_w(h_w) = u_o(h_w)",
      abs(u_water(h_w) - u_oil(h_w)) < 1e-12 * scale_u,
      f"jump = {abs(u_water(h_w) - u_oil(h_w)):.2e} m/s")
check("shear continuity          mu_w u_w'(h_w) = mu_o u_o'(h_w)",
      abs(tau_iface_w - tau_iface_o) < 1e-10 * scale_tau,
      f"jump = {abs(tau_iface_w - tau_iface_o):.2e} Pa")
check("shear-free free surface   tau(H) = 0",
      abs(tau_oil(H)) < 1e-10 * scale_tau,
      f"|tau(H)| = {abs(tau_oil(H)):.2e} Pa")

# 2. Overall force balance --------------------------------------------------
# Steady fully developed flow: the wall shear balances the total streamwise
# body/pressure force per unit width, tau_wall = -(D_w h_w + D_o h_o).
force_rhs = -(D_w * h_w + D_o * h_o)
check("global x-force balance     tau_wall = -(D_w h_w + D_o h_o)",
      abs(tau_wall - force_rhs) < 1e-9 * scale_tau,
      f"residual = {abs(tau_wall - force_rhs):.2e} Pa")

# 3. Flow rates vs numerical quadrature ------------------------------------
yw = np.linspace(0.0, h_w, 8001)
yo = np.linspace(h_w, H, 8001)
Qw_quad = trapezoid(u_water(yw), yw)
Qo_quad = trapezoid(u_oil(yo), yo)
check("water flow rate  Q_w vs quadrature",
      abs(Q_w - Qw_quad) < 1e-6 * (abs(Q_w) + 1e-30),
      f"rel.err = {abs(Q_w - Qw_quad)/(abs(Q_w)+1e-30):.2e}")
check("oil   flow rate  Q_o vs quadrature",
      abs(Q_o - Qo_quad) < 1e-6 * (abs(Q_o) + 1e-30),
      f"rel.err = {abs(Q_o - Qo_quad)/(abs(Q_o)+1e-30):.2e}")


# 4. Limiting cases ---------------------------------------------------------
def solve_generic(rw, ro, mw, mo, hw, ho, dp, th):
    """Re-derive the four coefficients for arbitrary inputs (for the checks)."""
    Ht = hw + ho
    Dw = dp - rw * g * np.sin(th)
    Do = dp - ro * g * np.sin(th)
    aw, ao = Dw / mw, Do / mo
    bw = -(Dw * hw + Do * ho) / mw
    bo = -ao * Ht
    co = 0.5 * (aw - ao) * hw**2 + (bw - bo) * hw
    return aw, bw, 0.0, ao, bo, co, Ht


# (a) Equal properties -> single homogeneous fluid, wall to free surface.
#     Exact single-layer film:  u(y) = a(H y - 0.5 y^2),  a = D/mu.
aw, bw, cw, ao, bo, co, Ht = solve_generic(rho_w, rho_w, mu_w, mu_w,
                                           h_w, h_o, dpdx, theta)
D = dpdx - rho_w * g * np.sin(theta)
a_single = D / mu_w
yv = np.linspace(0.0, Ht, 4001)
u_two = np.where(yv <= h_w, 0.5*aw*yv**2 + bw*yv + cw,
                 0.5*ao*yv**2 + bo*yv + co)
u_one = a_single * (0.5 * yv**2 - Ht * yv)      # single-layer film, no-slip+free
check("limiting case: equal properties -> single-layer film",
      np.max(np.abs(u_two - u_one)) < 1e-12 * (np.max(np.abs(u_one)) + 1e-30),
      f"max|u_2layer - u_1layer| = {np.max(np.abs(u_two - u_one)):.2e} m/s")

# (b) Zero pressure gradient -> the present case already has dp/dx = 0; verify
#     the gravity-only wall shear equals the total weight component per width.
Dw0 = -rho_w * g * np.sin(theta)
Do0 = -rho_o * g * np.sin(theta)
tau_wall0 = -(Dw0 * h_w + Do0 * h_o)
weight_x = (rho_w * h_w + rho_o * h_o) * g * np.sin(theta)   # per unit width
check("limiting case: dp/dx = 0 -> tau_wall = total weight/width * sin(theta)",
      abs(tau_wall0 - weight_x) < 1e-9 * (abs(weight_x) + 1e-30),
      f"tau_wall = {tau_wall0:.4e} Pa,  weight_x = {weight_x:.4e} Pa")

# (c) Horizontal plate (theta = 0) -> both driving terms equal dp/dx.
aw, bw, cw, ao, bo, co, Ht = solve_generic(rho_w, rho_o, mu_w, mu_o,
                                           h_w, h_o, -50.0, 0.0)
tau_wall_h = mu_w * bw
check("limiting case: horizontal (theta=0) -> tau_wall = -dp/dx * H",
      abs(tau_wall_h - (-(-50.0) * (h_w + h_o))) < 1e-9 *
      (abs(50.0 * (h_w + h_o)) + 1e-30),
      f"tau_wall = {tau_wall_h:.4e} Pa,  -dp/dx*H = {50.0*(h_w+h_o):.4e} Pa")

# (d) Vanishing oil layer (h_o -> 0) -> single water film of thickness h_w.
aw, bw, cw, ao, bo, co, Ht = solve_generic(rho_w, rho_o, mu_w, mu_o,
                                           h_w, 1e-12, dpdx, theta)
a_wsingle = (dpdx - rho_w * g * np.sin(theta)) / mu_w
yv = np.linspace(0.0, h_w, 4001)
u_thin = 0.5 * aw * yv**2 + bw * yv + cw
u_wsingle = a_wsingle * (0.5 * yv**2 - h_w * yv)   # single water film, no-slip+free
check("limiting case: h_o -> 0 -> single water film of thickness h_w",
      np.max(np.abs(u_thin - u_wsingle)) < 1e-6 *
      (np.max(np.abs(u_wsingle)) + 1e-30),
      f"max|diff| = {np.max(np.abs(u_thin - u_wsingle)):.2e} m/s")

print("\n" + "-" * 78)
if all(checks):
    print(f"  ALL {len(checks)} CHECKS PASSED -- the analytical two-phase "
          "solution is verified.")
else:
    print(f"  {checks.count(False)} of {len(checks)} CHECKS FAILED.")
assert all(checks), "two-phase analytical solution failed its verification"


# ===========================================================================
# Plots:  velocity profile (both layers, coloured) + shear-stress distribution
# ===========================================================================
yw = np.linspace(0.0, h_w, 400)
yo = np.linspace(h_w, H, 400)
uw, uo = u_water(yw), u_oil(yo)
tw, to = tau_water(yw), tau_oil(yo)

WATER_C, OIL_C = "#1f77b4", "#d4a017"   # blue water, amber oil
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.0),
                               constrained_layout=True)

# -- velocity profile --
ax1.plot(uw, yw * 1e3, color=WATER_C, lw=2.4, label="water  $u_w(y)$")
ax1.plot(uo, yo * 1e3, color=OIL_C, lw=2.4, label="oil  $u_o(y)$")
ax1.axhspan(0.0, h_w * 1e3, color=WATER_C, alpha=0.08)
ax1.axhspan(h_w * 1e3, H * 1e3, color=OIL_C, alpha=0.10)
ax1.axhline(h_w * 1e3, color="0.35", lw=1.2, ls="--", label="interface")
ax1.axhline(H * 1e3, color="0.15", lw=1.6, label="free surface")
ax1.plot(u_interface, h_w * 1e3, "o", color="0.2", ms=6, zorder=5)
ax1.axvline(0.0, color="0.7", lw=0.8)
ax1.set_xlabel("velocity  u (m/s)")
ax1.set_ylabel("wall-normal coordinate  y (mm)")
ax1.set_title("Velocity profile across both layers")
ax1.legend(frameon=False, fontsize=9, loc="lower right")
ax1.grid(alpha=0.3)

# -- shear-stress distribution --
ax2.plot(tw, yw * 1e3, color=WATER_C, lw=2.4, label=r"water  $\tau_w(y)$")
ax2.plot(to, yo * 1e3, color=OIL_C, lw=2.4, label=r"oil  $\tau_o(y)$")
ax2.axhline(h_w * 1e3, color="0.35", lw=1.2, ls="--")
ax2.axhline(H * 1e3, color="0.15", lw=1.6)
ax2.axvline(0.0, color="0.7", lw=0.8)
ax2.set_xlabel(r"shear stress  $\tau = \mu\,du/dy$  (Pa)")
ax2.set_ylabel("wall-normal coordinate  y (mm)")
ax2.set_title("Shear-stress distribution")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
ax2.grid(alpha=0.3)

fig.suptitle("Stratified two-phase (water/oil) flow over an inclined plate "
             "with a free surface", fontsize=13)
fig.savefig("fig7_4_two_phase.png", dpi=150, bbox_inches="tight")
print("\n  Wrote fig7_4_two_phase.png")
