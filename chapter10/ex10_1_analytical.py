#!/usr/bin/env python3
"""
Example 10.1 -- The law of the wall.

Near a solid wall a turbulent shear flow has a universal mean-velocity structure
when expressed in wall units, u+ = u/u_tau and y+ = y u_tau/nu, where the friction
velocity is u_tau = sqrt(tau_w/rho).  It has three regions:

  * viscous sublayer (y+ < 5):     u+ = y+
  * log-law region (y+ > 30):      u+ = (1/kappa) ln(y+) + B
  * buffer layer (5 < y+ < 30):    a smooth blend

with the (near-universal) constants kappa = 0.41 (von Karman) and B = 5.0.  The log
law follows from Prandtl's mixing-length hypothesis: with mixing length l = kappa y
and constant total stress tau_w = rho l^2 (du/dy)^2, one gets du/dy = u_tau/(kappa y),
which integrates to the logarithm.  This program builds the composite profile from
Spalding's single implicit formula, verifies that it reduces to the sublayer and the
log law in the two limits, and confirms the mixing-length consistency of the log law.
Uses numpy, matplotlib, and scipy.optimize.brentq (via the bundled shim).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

kappa, B = 0.41, 5.0

def spalding_yplus(up):
    """Spalding's law: y+ as a function of u+ (valid across sublayer, buffer, log)."""
    ku = kappa*up
    return up + np.exp(-kappa*B)*(np.exp(ku) - 1 - ku - ku**2/2 - ku**3/6)

def uplus_of_yplus(yp):
    return brentq(lambda up: spalding_yplus(up) - yp, 0.0, 100.0, xtol=1e-12)

print("Example 10.1  The law of the wall (kappa=0.41, B=5.0)\n")

# verify limits
print("  Limiting behaviour of Spalding's composite profile:")
yp_small = 0.5
up_small = uplus_of_yplus(yp_small)
print(f"    y+={yp_small}: u+={up_small:.4f}  (sublayer u+=y+ gives {yp_small})  "
      f"diff={abs(up_small-yp_small):.2e}")
yp_large = 1000.0
up_large = uplus_of_yplus(yp_large)
loglaw = (1/kappa)*np.log(yp_large) + B
print(f"    y+={yp_large}: u+={up_large:.4f}  (log law gives {loglaw:.4f})  "
      f"diff={abs(up_large-loglaw):.2e}")
assert abs(up_small - yp_small) < 1e-2
assert abs(up_large - loglaw) < 5e-2
print("  PASS: composite profile reduces to sublayer and log law in the limits.\n")

# verify mixing-length consistency: du+/dy+ = 1/(kappa y+) in the log region
print("  Mixing-length consistency  du+/dy+ = 1/(kappa y+) in the log region:")
for yp in (500.0, 1000.0, 3000.0):
    up1 = uplus_of_yplus(yp*1.001); up0 = uplus_of_yplus(yp*0.999)
    dudy = (up1-up0)/(yp*0.002)
    print(f"    y+={yp:6.0f}: numeric du+/dy+={dudy:.5e}, 1/(kappa y+)={1/(kappa*yp):.5e}, "
          f"diff={abs(dudy-1/(kappa*yp)):.2e}")
    assert abs(dudy - 1/(kappa*yp))/(1/(kappa*yp)) < 3e-2
print("  PASS: log-law slope matches the mixing-length prediction.\n")

# figure: universal profile
yp = np.logspace(-1, 3.3, 300)
up = np.array([uplus_of_yplus(y) for y in yp])
fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
ax.semilogx(yp, up, "C0-", lw=2.2, label="Spalding composite")
ax.semilogx(yp[yp<8], yp[yp<8], "C3--", lw=1.6, label="sublayer $u^+=y^+$")
ypl = yp[yp>25]
ax.semilogx(ypl, (1/kappa)*np.log(ypl)+B, "C2-.", lw=1.6,
            label=r"log law $\frac{1}{\kappa}\ln y^+ + B$")
ax.axvspan(0, 5, color="0.9"); ax.axvspan(5, 30, color="0.95")
ax.text(1.2, 16, "viscous\nsublayer", fontsize=8, ha="center")
ax.text(13, 3, "buffer", fontsize=8, ha="center")
ax.text(200, 8, "log-law region", fontsize=8, ha="center")
ax.set_xlabel("$y^+ = y u_\\tau/\\nu$"); ax.set_ylabel("$u^+ = u/u_\\tau$")
ax.set_title("The universal law of the wall")
ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.3, which="both")
ax.set_xlim(0.1, 2000); ax.set_ylim(0, 25)
fig.savefig("fig10_1_lawofwall.png", dpi=150, bbox_inches="tight")
print("  Wrote fig10_1_lawofwall.png")
