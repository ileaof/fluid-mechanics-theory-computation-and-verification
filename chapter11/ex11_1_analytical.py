#!/usr/bin/env python3
"""
Example 11.1 -- The convection-diffusion equation and the cell Peclet number.

The steady one-dimensional convection-diffusion equation

        rho u dphi/dx = Gamma d^2 phi/dx^2 ,   phi(0)=0 , phi(L)=1 ,

is the model problem of computational fluid dynamics: it contains, in one dimension,
the competition between convection (which transports phi with the flow) and diffusion
(which smooths it) that dominates the Navier-Stokes equations.  Its exact solution,

        phi(x) = ( exp(Pe x/L) - 1 ) / ( exp(Pe) - 1 ) ,   Pe = rho u L / Gamma ,

develops a thin boundary layer of thickness ~ L/Pe near the outflow as the Peclet
number grows.  This program verifies the exact solution against the differential
equation, examines the boundary-layer structure, and introduces the CELL Peclet
number Pe_cell = rho u dx/Gamma that governs whether a numerical scheme remains
bounded -- the central concern of Example 11.2.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L, rho, u = 1.0, 1.0, 1.0

def phi_exact(x, Pe):
    return (np.exp(Pe*x/L) - 1.0)/(np.exp(Pe) - 1.0)

print("Example 11.1  Convection-diffusion: exact solution and cell Peclet number\n")

# verify the exact solution satisfies rho u phi' = Gamma phi'' and the BCs
print("  Verify exact solution satisfies the ODE (residual by finite differences):")
print(f"    {'Pe':>6} {'phi(0)':>9} {'phi(L)':>9} {'max ODE residual':>18}")
for Pe in (1.0, 10.0, 50.0):
    Gamma = rho*u*L/Pe
    x = np.linspace(0, L, 40001); dx = x[1]-x[0]
    p = phi_exact(x, Pe)
    d1 = np.gradient(p, dx)
    d2 = np.gradient(d1, dx)
    res = rho*u*d1 - Gamma*d2
    interior = slice(50, -50)
    print(f"    {Pe:6.1f} {p[0]:9.5f} {p[-1]:9.5f} {np.max(np.abs(res[interior])):18.2e}")
    assert abs(p[0]) < 1e-12 and abs(p[-1]-1) < 1e-12
    assert np.max(np.abs(res[interior])) < 1e-3
print("  PASS: exact solution satisfies the convection-diffusion equation and BCs.\n")

# boundary-layer thickness: distance over which phi rises from ~0 to 0.99
print("  Outflow boundary-layer thickness (phi = 0.01 to 0.99):")
print(f"    {'Pe':>6} {'delta_99/L (numeric)':>22} {'~ln(100)/Pe':>14}")
for Pe in (10.0, 50.0, 100.0):
    x = np.linspace(0, L, 200001)
    p = phi_exact(x, Pe)
    x01 = x[np.argmax(p >= 0.01)]
    x99 = x[np.argmax(p >= 0.99)]
    delta = x99 - x01
    print(f"    {Pe:6.1f} {delta/L:22.5f} {np.log(100)/Pe:14.5f}")
    assert abs(delta/L - np.log(100)/Pe) < 0.02
print("\n  As Pe grows the solution is flat over most of the domain and rises steeply")
print("  only in a thin outflow layer -- the feature that defeats naive schemes.\n")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,4.4),constrained_layout=True)
xx = np.linspace(0, L, 400)
for Pe in (0.1, 1, 5, 20, 100):
    ax1.plot(xx, phi_exact(xx, Pe), lw=2, label=f"Pe = {Pe:g}")
ax1.set_xlabel("x/L"); ax1.set_ylabel(r"$\phi$")
ax1.set_title("Convection-diffusion profiles"); ax1.legend(frameon=False, fontsize=8)
ax1.grid(alpha=0.3)
# cell-Peclet illustration: nodes for N=10 at Pe=50 -> Pe_cell=5 (>2)
N=10; Pe=50; xc=np.linspace(0,L,N+1)
ax2.plot(xx, phi_exact(xx,Pe), "k-", lw=2, label="exact (Pe=50)")
ax2.plot(xc, phi_exact(xc,Pe), "C0o", ms=6, label=f"nodes (N={N}, Pe_cell={Pe/N:g})")
ax2.axhline(0, color="0.6", lw=0.8)
ax2.set_xlabel("x/L"); ax2.set_ylabel(r"$\phi$")
ax2.set_title(r"Coarse grid: $Pe_{cell}=5>2$"); ax2.legend(frameon=False, fontsize=8)
ax2.grid(alpha=0.3)
fig.suptitle("The convection-diffusion model problem", y=1.04, fontsize=13)
fig.savefig("fig11_1_convdiff.png", dpi=150, bbox_inches="tight")
print("  Wrote fig11_1_convdiff.png")
