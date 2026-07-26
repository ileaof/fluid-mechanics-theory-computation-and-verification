#!/usr/bin/env python3
"""
Example 8.1 -- The Blasius laminar boundary layer by the shooting method.

For steady laminar flow over a flat plate at zero incidence, Prandtl's
boundary-layer equations admit a similarity solution.  With the similarity
variable eta = y sqrt(U/(nu x)) and stream function giving u/U = f'(eta), the
problem reduces to the third-order nonlinear Blasius ordinary differential
equation

        f''' + (1/2) f f'' = 0 ,   f(0)=0 , f'(0)=0 , f'(inf)=1 .

There is no elementary closed form, but the equation is solved exactly (to machine
precision) by SHOOTING: guess the wall curvature s = f''(0), integrate the ODE as
an initial-value problem with fourth-order Runge-Kutta, and adjust s by the secant
method until f'(eta_max) = 1.  The program recovers the celebrated constant
f''(0) = 0.33206 and the boundary-layer integral parameters, and verifies the
engineering results delta/x, C_f, and the drag coefficient.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

eta_max, N = 10.0, 20000
eta = np.linspace(0.0, eta_max, N+1)
h = eta[1]-eta[0]

def rhs(Y):
    f, fp, fpp = Y
    return np.array([fp, fpp, -0.5*f*fpp])

def integrate(s):
    """RK4 integrate the Blasius ODE from eta=0 with f''(0)=s; return full history."""
    Y = np.array([0.0, 0.0, s])
    hist = np.empty((N+1, 3)); hist[0] = Y
    for i in range(N):
        k1 = rhs(Y); k2 = rhs(Y+0.5*h*k1)
        k3 = rhs(Y+0.5*h*k2); k4 = rhs(Y+h*k3)
        Y = Y + h/6.0*(k1+2*k2+2*k3+k4); hist[i+1] = Y
    return hist

def shoot():
    """Secant iteration on s=f''(0) so that f'(eta_max)=1."""
    s0, s1 = 0.30, 0.35
    g0 = integrate(s0)[-1,1] - 1.0
    for _ in range(60):
        g1 = integrate(s1)[-1,1] - 1.0
        if abs(g1-g0) < 1e-15: break
        s2 = s1 - g1*(s1-s0)/(g1-g0)
        s0, g0, s1 = s1, g1, s2
        if abs(s1-s0) < 1e-13: break
    return s1

print("Example 8.1  Blasius laminar boundary layer (shooting method)\n")
s = shoot()
H = integrate(s)
f, fp, fpp = H[:,0], H[:,1], H[:,2]
print(f"  Wall curvature  f''(0) = {s:.6f}   (accepted value 0.33206)")
print(f"  f'(eta_max)          = {fp[-1]:.8f}   (target 1)")
assert abs(s-0.33206) < 1e-3, "shooting did not recover the Blasius constant"
assert abs(fp[-1]-1.0) < 1e-6

# boundary-layer integral thicknesses (in eta units)
disp = trapezoid(1-fp, eta)                 # displacement thickness delta*/ (x/sqrt(Re_x))... in eta
mom  = trapezoid(fp*(1-fp), eta)            # momentum thickness (eta units)
shape = disp/mom
# eta at u/U = 0.99
i99 = np.argmax(fp >= 0.99)
eta99 = eta[i99]
print(f"\n  Integral parameters (similarity units):")
print(f"    displacement thickness  int(1-f') d_eta = {disp:.4f}  (1.7208)")
print(f"    momentum   thickness  int f'(1-f') d_eta = {mom:.4f}  (0.6641)")
print(f"    shape factor H = delta*/theta          = {shape:.4f}  (2.591)")
print(f"    eta at u/U=0.99                        = {eta99:.3f}  (~4.91)")
assert abs(disp-1.7208) < 1e-2 and abs(mom-0.6641) < 1e-2 and abs(shape-2.591) < 1e-2

# engineering results
print(f"\n  Engineering formulas that follow:")
print(f"    delta99/x = {eta99:.2f}/sqrt(Re_x)         (~5.0/sqrt(Re_x))")
print(f"    C_f(x)    = {2*s:.4f}/sqrt(Re_x)        (0.664/sqrt(Re_x))")
print(f"    C_D(plate)= {4*s:.4f}/sqrt(Re_L)        (1.328/sqrt(Re_L))")
assert abs(2*s-0.664) < 2e-3
print("\n  PASS: Blasius constant, integral thicknesses, and C_f verified.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,4.5),constrained_layout=True)
ax1.plot(fp, eta, "C0-", lw=2, label="$u/U=f'(\\eta)$")
ax1.axhline(eta99, color="C3", ls="--", lw=1.2)
ax1.text(0.05, eta99+0.15, r"$\eta_{99}\approx4.91$", color="C3")
ax1.set_xlabel("$u/U$"); ax1.set_ylabel(r"$\eta=y\sqrt{U/\nu x}$")
ax1.set_title("Blasius velocity profile"); ax1.legend(frameon=False); ax1.grid(alpha=0.3)
ax1.set_ylim(0,8)
Rex=np.logspace(3,6,100)
ax2.loglog(Rex, 2*s/np.sqrt(Rex), "C0-", lw=2, label=r"$C_f=0.664/\sqrt{\mathrm{Re}_x}$")
ax2.loglog(Rex, eta99/np.sqrt(Rex), "C3--", lw=2, label=r"$\delta/x=5.0/\sqrt{\mathrm{Re}_x}$")
ax2.set_xlabel(r"$\mathrm{Re}_x$"); ax2.set_ylabel("$C_f$,  $\\delta/x$")
ax2.set_title("Laminar flat-plate scaling"); ax2.legend(frameon=False); ax2.grid(alpha=0.3,which="both")
fig.suptitle("Blasius boundary layer on a flat plate", y=1.04, fontsize=13)
fig.savefig("fig8_1_blasius.png", dpi=150, bbox_inches="tight")
print("  Wrote fig8_1_blasius.png")
