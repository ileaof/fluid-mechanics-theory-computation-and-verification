"""Literature reference data for the flow-past-a-circular-cylinder benchmark.

The numbers below are the classical 2-D incompressible values used to validate
a bluff-body Navier-Stokes solver.  They are drawn from the standard corpus
(Tritton; Coutanceau & Bouard; Dennis & Chang; Fornberg; Henderson; Norberg;
Roshko; Williamson) and are quoted to the precision that a *staircase*
direct-forcing implementation on a Cartesian mesh can be expected to match --
i.e. a few to ~15 % at the finer meshes.  Use :func:`benchmark_for` to look up
the reference tuple for a Reynolds number; the runner computes the percent
difference against the simulated values.

Each entry is::

    (Re, Cd_mean, Cl_rms, St, Lr/D, theta_sep_deg)

* ``Cd_mean``  -- mean drag coefficient (time-averaged; for Re<=40 the steady Cd).
* ``Cl_rms``   -- rms lift coefficient (0 for the steady regime Re <= ~47).
* ``St``       -- Strouhal shedding number (0 for the steady regime).
* ``Lr/D``     -- recirculation length in cylinder diameters (steady regime).
* ``theta_sep``-- mean separation angle measured from the forward stagnation
                  point, in degrees (steady regime; ``nan`` when unsteady).

The ``nan``/zero entries mark quantities that are undefined in the steady
regime (no shedding) or the unsteady regime (no fixed recirculation bubble).
"""

from __future__ import annotations

import math
from typing import NamedTuple


class Bench(NamedTuple):
    Re: float
    Cd: float
    Cl_rms: float
    St: float
    Lr_D: float
    theta_sep: float


# Re  :  Cd    Cl_rms  St     Lr/D   theta_sep
_TABLE = {
    20:   (2.05, 0.0,   0.0,    0.90,  44.0),   # steady, Coutanceau & Bouard / Dennis & Chang
    40:   (1.52, 0.0,   0.0,    2.20,  53.5),   # steady, classic value
    100:  (1.33, 0.094, 0.165,  math.nan, math.nan),  # von Karman shedding (Norberg / Henderson)
    200:  (1.34, 0.44,  0.197,  math.nan, math.nan),
    300:  (1.28, 0.60,  0.203,  math.nan, math.nan),
    1000: (1.00, 0.20,  0.21,   math.nan, math.nan),  # subcritical drag bucket
}


def benchmark_for(Re: float) -> Bench:
    """Return the literature reference closest to ``Re`` (exact key preferred)."""
    Re = float(Re)
    if Re in _TABLE:
        c = _TABLE[Re]
        return Bench(Re, *c)
    # nearest available key (so an off-table Re still gets a comparison point)
    nearest = min(_TABLE.keys(), key=lambda k: abs(k - Re))
    c = _TABLE[nearest]
    return Bench(Re, *c)


def all_benchmarks() -> dict[float, Bench]:
    return {Re: Bench(Re, *c) for Re, c in _TABLE.items()}


def pct(sim: float, ref: float) -> float:
    """Signed percent difference (sim vs ref); ``nan`` if ref is nan/zero."""
    if ref is None or (isinstance(ref, float) and math.isnan(ref)):
        return math.nan
    if abs(ref) < 1e-12:
        return math.nan
    return 100.0 * (sim - ref) / abs(ref)