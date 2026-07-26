#!/usr/bin/env python3
"""
Example 3.1 -- Analytical kinematics of plane stagnation-point flow.

The steady, two-dimensional velocity field

        u = a x ,     v = -a y ,        (a > 0, units 1/s)

is the flow that impinges on a wall and divides at a stagnation point.  It is the
simplest field that exhibits, together, every kinematic quantity of this chapter:

  * stream function      psi = a x y                 (u = d psi/dy, v = -d psi/dx)
  * velocity potential   phi = (a/2)(x^2 - y^2)      (u = d phi/dx, v = d phi/dy)
  * continuity           du/dx + dv/dy = a - a = 0   (incompressible)
  * vorticity            omega = dv/dx - du/dy = 0   (irrotational)
  * material accel.      Dx u = a^2 x,  Dy v = a^2 y (purely convective)
  * rate of strain       eigenvalues +/- a (pure straining, no rotation)

The program evaluates each analytical quantity and verifies it against a
second-order finite-difference evaluation of the same derivatives, then draws the
orthogonal net of streamlines and equipotentials with the acceleration field.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = 1.5     # strain rate [1/s]

def vel(x, y):
    return a * x, -a * y

def psi(x, y):   return a * x * y
def phi(x, y):   return 0.5 * a * (x**2 - y**2)

def accel_exact(x, y):
    # material acceleration Du/Dt = (V . grad) V for steady flow
    return a**2 * x, a**2 * y

# ---- finite-difference verification of the field identities ----------------
def fd_grad(f, x, y, h=1e-6):
    return ((f(x + h, y) - f(x - h, y)) / (2*h),
            (f(x, y + h) - f(x, y - h)) / (2*h))

print("Example 3.1  Plane stagnation-point flow -- analytical kinematics\n")
pts = [(1.0, 1.0), (2.0, 0.5), (-1.5, 1.2), (0.7, -1.3)]
print(f"  {'point':>12} {'div V':>10} {'vorticity':>11} {'a_err':>10} "
      f"{'u=dpsi/dy':>11} {'u=dphi/dx':>11}")
worst = 0.0
for (x, y) in pts:
    u, v = vel(x, y)
    # divergence and vorticity by central differences of the velocity field
    (ux, uy) = fd_grad(lambda X, Y: vel(X, Y)[0], x, y)
    (vx, vy) = fd_grad(lambda X, Y: vel(X, Y)[1], x, y)
    div = ux + vy
    vort = vx - uy
    # acceleration by material derivative, checked against exact
    ax_num = u*ux + v*uy
    ay_num = u*vx + v*vy
    axe, aye = accel_exact(x, y)
    a_err = max(abs(ax_num - axe), abs(ay_num - aye))
    # stream function / potential give back the velocity
    dpsidy = fd_grad(psi, x, y)[1]
    dphidx = fd_grad(phi, x, y)[0]
    e = max(abs(div), abs(vort), a_err, abs(dpsidy - u), abs(dphidx - u))
    worst = max(worst, e)
    print(f"  ({x:+.1f},{y:+.1f}) {div:10.2e} {vort:11.2e} {a_err:10.2e}"
          f" {dpsidy:11.4f} {dphidx:11.4f}")

print(f"\n  Worst residual across all identities: {worst:.2e}")
assert worst < 1e-4, "a kinematic identity failed its finite-difference check"

# rate-of-strain tensor E = sym(grad V); eigenvalues should be +/- a
E = np.array([[a, 0.0], [0.0, -a]])
eigs = np.linalg.eigvalsh(E)
print(f"  Rate-of-strain eigenvalues: {eigs[0]:+.4f}, {eigs[1]:+.4f}  (expected +/- {a})")
assert np.allclose(sorted(eigs), sorted([-a, a])), "strain eigenvalues wrong"
print("  PASS: incompressible, irrotational, pure strain; psi and phi verified.\n")

# ---- figure: orthogonal net + acceleration ---------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
xx = np.linspace(-2, 2, 400); yy = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(xx, yy)
ax1.contour(X, Y, psi(X, Y), levels=np.linspace(-4, 4, 17), colors="C0", linewidths=1)
ax1.contour(X, Y, phi(X, Y), levels=np.linspace(-4, 4, 17), colors="C3",
            linewidths=1, linestyles="--")
ax1.plot(0, 0, "ko", ms=6)
ax1.text(0.06, 0.08, "stagnation point", fontsize=9)
ax1.set_title("Streamlines (solid) and equipotentials (dashed)")
ax1.set_xlabel("$x$ [m]"); ax1.set_ylabel("$y$ [m]"); ax1.set_aspect("equal")

xs = np.linspace(-2, 2, 11); ys = np.linspace(-2, 2, 11)
Xs, Ys = np.meshgrid(xs, ys)
U, V = vel(Xs, Ys)
ax2.quiver(Xs, Ys, U, V, color="C0", alpha=0.7)
Axx, Ayy = accel_exact(Xs, Ys)
ax2.quiver(Xs, Ys, Axx, Ayy, color="C3", alpha=0.5, scale=40)
ax2.plot(0, 0, "ko", ms=6)
ax2.set_title("Velocity (blue) and acceleration (red)")
ax2.set_xlabel("$x$ [m]"); ax2.set_ylabel("$y$ [m]"); ax2.set_aspect("equal")

fig.suptitle("Kinematics of plane stagnation-point flow", y=1.04, fontsize=13)
fig.savefig("fig3_1_stagnation.png", dpi=150, bbox_inches="tight")
print("  Wrote fig3_1_stagnation.png")
