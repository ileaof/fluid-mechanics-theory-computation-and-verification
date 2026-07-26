#!/usr/bin/env python3
"""
Example 12.1 -- Richardson extrapolation and the observed order of accuracy.

Any convergent numerical method has a discretisation error that behaves, in the
asymptotic range, as a power of the mesh spacing,

        f_h = f_exact + C h^p + higher-order terms,

where p is the order of accuracy.  Two consequences follow, and both are the
foundation of solution verification.  First, from three solutions on systematically
refined grids the OBSERVED order of accuracy can be measured,

        p = ln[ (f_2h - f_4h)/(f_h - f_2h) ] / ln(r) ,

without knowing the exact answer.  Second, the leading error term can be eliminated
by RICHARDSON EXTRAPOLATION,

        f_exact ~ f_h + (f_h - f_2h)/(r^p - 1) ,

producing an estimate far more accurate than either grid.  This program demonstrates
both with the trapezoidal rule (order p = 2): it measures the observed order, applies
Richardson extrapolation to boost the accuracy to fourth order (the Romberg method),
and verifies the extrapolated value against the exact integral.
Only numpy and matplotlib are used; no random numbers are involved.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# test integral  I = integral_0^1 e^x dx = e - 1  (exact)
def f(x): return np.exp(x)
I_exact = np.e - 1.0

def trapezoid_rule(N):
    x = np.linspace(0, 1, N+1); h = 1.0/N
    return h*(0.5*f(x[0]) + f(x[1:-1]).sum() + 0.5*f(x[-1]))

print("Example 12.1  Richardson extrapolation and the observed order of accuracy\n")
print(f"  Exact integral  I = e - 1 = {I_exact:.12f}\n")

# --- observed order of accuracy from successive grids (no exact needed) ------
print("  Trapezoidal rule: error and OBSERVED order of accuracy")
print(f"    {'N':>6} {'T(h)':>16} {'error':>13} {'obs. order p':>13}")
Ns = [4, 8, 16, 32, 64, 128]
T = {N: trapezoid_rule(N) for N in Ns}
prev_err = None
for N in Ns:
    err = T[N] - I_exact
    # observed order from three grids ending at N
    if N//4 in T and N//2 in T:
        p_obs = np.log(abs((T[N//4]-I_exact)/(T[N]-I_exact)))/np.log(4)  # ref order
        num = T[N//4]-T[N//2]; den = T[N//2]-T[N]
        p_meas = np.log(abs(num/den))/np.log(2)
        print(f"    {N:6d} {T[N]:16.10f} {err:13.3e} {p_meas:13.4f}")
    else:
        print(f"    {N:6d} {T[N]:16.10f} {err:13.3e} {'--':>13}")
p_final = np.log(abs((T[32]-T[64])/(T[64]-T[128])))/np.log(2)
print(f"\n  Observed order (finest triple): p = {p_final:.4f}  (trapezoidal design p = 2)")
assert abs(p_final-2.0) < 0.05

# --- Richardson extrapolation: boost order 2 -> 4 (Romberg) -----------------
print("\n  Richardson extrapolation  R = (r^p f_h - f_2h)/(r^p - 1),  r=2, p=2:")
print(f"    {'N':>6} {'Richardson value':>20} {'error':>13} {'obs. order':>12}")
prevR = None; Rerrs=[]
for N in [8,16,32,64,128]:
    R = (4*T[N] - T[N//2])/3.0        # eliminate the h^2 term
    errR = R - I_exact
    pR = np.log(abs(prevR/errR))/np.log(2) if prevR else float("nan")
    print(f"    {N:6d} {R:20.14f} {errR:13.3e} {pR:12.3f}")
    Rerrs.append((1.0/N, abs(errR))); prevR=errR
pR_final = np.log(Rerrs[-2][1]/Rerrs[-1][1])/np.log(Rerrs[-2][0]/Rerrs[-1][0])
R_best = (4*T[128]-T[64])/3.0
print(f"\n  Richardson-extrapolated value = {R_best:.14f}")
print(f"  Exact value                   = {I_exact:.14f}")
print(f"  Extrapolation error           = {abs(R_best-I_exact):.2e}")
print(f"  Observed order of Richardson value: p = {pR_final:.3f}  (boosted to 4)")
assert 3.5 < pR_final < 4.5
assert abs(R_best-I_exact) < 1e-8
print("  PASS: observed order = 2; Richardson boosts it to 4 and matches exact.\n")

# figure
fig,ax=plt.subplots(figsize=(7.2,5.0),constrained_layout=True)
hs=np.array([1.0/N for N in Ns]); errs=np.array([abs(T[N]-I_exact) for N in Ns])
ax.loglog(hs, errs, "C0o-", lw=1.8, label="trapezoidal (order 2)")
hr=np.array([r[0] for r in Rerrs]); er=np.array([r[1] for r in Rerrs])
ax.loglog(hr, er, "C3s-", lw=1.8, label="Richardson / Romberg (order 4)")
ax.loglog(hs, errs[-1]*(hs/hs[-1])**2, "k:", lw=1.2, label="slope 2")
ax.loglog(hr, er[-1]*(hr/hr[-1])**4, "k--", lw=1.2, label="slope 4")
ax.set_xlabel("mesh spacing $h$"); ax.set_ylabel("absolute error")
ax.set_title("Richardson extrapolation boosts the order of accuracy")
ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
fig.savefig("fig12_1_richardson.png", dpi=150, bbox_inches="tight")
print("  Wrote fig12_1_richardson.png")
