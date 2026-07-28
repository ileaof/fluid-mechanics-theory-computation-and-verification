"""Probe helper kept for reuse: run a cylinder case and print the Cd/Cl trajectory.

Usage: edit the `data.update({...})` block below for the desired mesh / Re /
tfinal, then `python _probe.py`. Used during validation to inspect convergence
behaviour without the full report writer.
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from config import Config
from main import Simulation
import numpy as np

# ---- edit here --------------------------------------------------------- #
NX, NY = 200, 80
RE = 40.0
TFINAL = 20.0
# ------------------------------------------------------------------------ #

data = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
data.update({
    "Nx": NX, "Ny": NY,
    "mu": 1.0 / float(RE),
    "tfinal": float(TFINAL),
    "output_interval": 0.05,
    "name": f"cyl_probe_{NX}x{NY}_Re{int(RE)}",
    "output_dir": f"outputs/cyl_probe_{NX}x{NY}_Re{int(RE)}",
    "rhie_chow": True,
    "compute_forces": True,
    "save_csv": False, "save_hdf5": False, "save_tecplot": False,
    "save_png": False, "save_mp4": False,
})
cfg = Config.from_dict(data)
sim = Simulation(cfg)
t0 = time.perf_counter()
sim.run()
wall = time.perf_counter() - t0

ts = np.array([r.get("t", 0.0) for r in sim.history], float)
Cd = np.array([r.get("Cd", np.nan) for r in sim.history], float)
Cl = np.array([r.get("Cl", np.nan) for r in sim.history], float)
div = np.array([r.get("div", 0.0) for r in sim.history], float)
fin = np.isfinite(Cd)
ts, Cd, Cl, div = ts[fin], Cd[fin], Cl[fin], div[fin]
print(f"\n=== {NX}x{NY} Re={RE}: {len(ts)} rows, t in [{ts[0]:.2f},{ts[-1]:.2f}], wall={wall:.1f}s, steps={sim.step_count} ===")
print(f"  max|div|={np.abs(div).max():.3e}  last10 div={np.abs(div[-10:]).mean():.3e}")
for i in range(0, len(Cd), max(1, len(Cd)//25)):
    print(f"    t={ts[i]:6.2f}  Cd={Cd[i]:8.3f}  Cl={Cl[i]:8.3f}")
print(f"  Cd last20% = {Cd[int(0.8*len(Cd)):].mean():.4f}   Cl last20% = {Cl[int(0.8*len(Cl)):].mean():.4f}")
fc = sim.forces_calc
if fc is not None:
    fr = fc.forces(sim.state.u, sim.state.v, sim.state.p)
    Lr = fc.recirculation_length(sim.state.u)
    print(f"  final: Fp_x={fr['Fp_x']:.4f} Fv_x={fr['Fv_x']:.4f} Cd={fr['Cd']:.4f} Cl={fr['Cl']:.4f} Lr/D={Lr/1.0:.4f}")