#!/usr/bin/env python3
"""
Example 6.1 -- The Buckingham Pi theorem computed from the dimensional matrix.

For the drag force F on a smooth sphere the relevant variables are the force F,
the fluid density rho, the free-stream velocity V, the diameter D, and the
viscosity mu.  In the M-L-T system their dimensions are

        [F]=M L T^-2,  [rho]=M L^-3,  [V]=L T^-1,  [D]=L,  [mu]=M L^-1 T^-1.

The Buckingham Pi theorem states that these n = 5 variables in k = 3 independent
dimensions combine into n - k = 2 independent dimensionless groups.  This program
builds the dimensional matrix, finds its rank, computes a basis for its null space
(each null vector is the exponent set of a dimensionless Pi group), and verifies
that every group is truly dimensionless.  It recovers the drag coefficient
C_D = F/(rho V^2 D^2) and the Reynolds number Re = rho V D / mu.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction

variables = ["F", "rho", "V", "D", "mu"]
# rows are exponents of (M, L, T)
A = np.array([
    [ 1,  1,  0,  0,  1],   # M
    [ 1, -3,  1,  1, -1],   # L
    [-2,  0, -1,  0, -1],   # T
], dtype=float)

print("Example 6.1  Buckingham Pi theorem for sphere drag\n")
print("  Variables:", variables)
print("  Dimensional matrix (rows M,L,T):")
for r, lab in zip(A, "MLT"):
    print(f"    {lab}: {r.astype(int)}")

n = len(variables)
k = np.linalg.matrix_rank(A)
npi = n - k
print(f"\n  n = {n} variables, k = rank = {k} dimensions -> {npi} Pi groups\n")

# null space via SVD
U, S, Vt = np.linalg.svd(A)
null = Vt[k:]                      # rows span the null space
# verify each null vector is dimensionless: A @ v = 0
print("  Null-space basis vectors (exponents of F,rho,V,D,mu) and dimensionlessness:")
for v in null:
    resid = np.max(np.abs(A @ v))
    print(f"    v = {np.round(v,3)}   max|A v| = {resid:.2e}")
    assert resid < 1e-10

# Physically meaningful, integer Pi groups (known forms) and their verification
def check(name, exps):
    exps = np.array(exps, float)
    d = A @ exps
    s = "  ".join(f"{var}^{int(e)}" for var, e in zip(variables, exps) if e != 0)
    print(f"    {name:6s}= {s:32s} dims (M,L,T) = {d.astype(int)}  -> {'DIMENSIONLESS' if np.allclose(d,0) else 'NOT'}")
    return np.allclose(d, 0)

print("\n  Recovering the classical dimensionless groups:")
ok1 = check("C_D", [1, -1, -2, -2, 0])     # F/(rho V^2 D^2)
ok2 = check("Re",  [0,  1,  1,  1, -1])     # rho V D / mu
assert ok1 and ok2
# confirm they are independent and span the null space (rank of stacked = npi)
groups = np.array([[1,-1,-2,-2,0],[0,1,1,1,-1]], float)
assert np.linalg.matrix_rank(np.vstack([null, groups])) == npi
print("\n  PASS: 2 independent dimensionless groups; C_D and Re verified and span")
print("        the null space, so F/(rho V^2 D^2) = phi(rho V D / mu).\n")

# figure: the resulting universal relation C_D = phi(Re) (standard drag curve)
Re = np.logspace(-1, 6, 400)
# a standard smooth drag correlation (Clift-Gauvin style) for illustration
CD = 24/Re*(1+0.15*Re**0.687) + 0.42/(1+4.25e4*Re**-1.16)
fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
ax.loglog(Re, CD, "C0-", lw=2)
ax.loglog(Re, 24/Re, "C3--", lw=1.6, label=r"Stokes $C_D=24/\mathrm{Re}$")
ax.set_xlabel(r"Reynolds number $\mathrm{Re}=\rho V D/\mu$")
ax.set_ylabel(r"drag coefficient $C_D=F/(\rho V^2 D^2)$")
ax.set_title("Dimensional analysis collapses 5 variables to $C_D=\\phi(\\mathrm{Re})$")
ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
fig.savefig("fig6_1_buckingham.png", dpi=150, bbox_inches="tight")
print("  Wrote fig6_1_buckingham.png")
