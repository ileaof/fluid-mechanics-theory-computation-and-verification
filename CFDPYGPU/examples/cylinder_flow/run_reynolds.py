"""Reynolds-sweep driver for the flow-past-a-circular-cylinder benchmark.

For each case ``(Re, (Nx, Ny), tfinal)`` this script builds a :class:`Config`
from ``config.json``, sets the viscosity from the Reynolds number
(``mu = rho * U_inf * D / Re``), runs the simulation, post-processes the
Cd / Cl history for the benchmark quantities (mean Cd, Cl_rms, Strouhal,
recirculation length, separation angle), and writes a Markdown report
(``cylinder_report.md``) with the per-case block and the percent difference
against the literature table in :mod:`benchmarks`.

Usage
-----
::

    python examples/cylinder_flow/run_reynolds.py            # validation subset
    python examples/cylinder_flow/run_reynolds.py --full      # full sweep
    python examples/cylinder_flow/run_reynolds.py --cases 40,100 \
        --mesh 400x160 --tfinal 40,80

The full sweep is expensive (see the printed wall-clock estimate); run the
validation subset first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo root importable when run from the example directory.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import Config                              # noqa: E402
from main import Simulation                            # noqa: E402
from benchmarks import benchmark_for, pct              # noqa: E402

# Geometry / reference constants (kept in sync with config.json).
U_INF = 1.0
D = 1.0                          # cylinder diameter (radius 0.5)
RHO = 1.0

# Validation subset (matches the plan: Re=40 steady, Re=100 shedding @ 400x160).
VALIDATION = [
    (40,  (400, 160), 40.0),
    (100, (400, 160), 80.0),
]

# Full sweep as requested in the spec.
FULL = [
    (20,   (400, 160), 40.0),
    (40,   (400, 160), 40.0),
    (100,  (400, 160), 80.0),
    (200,  (400, 160), 120.0),
    (300,  (400, 160), 120.0),
    (1000, (800, 320), 200.0),
]

# Mesh-independence study at Re=40 (steady, cheap): the staircase geometry
# error is first order, so Cd / Lr should converge under refinement.
MESH_STUDY = [
    (40, (200,  80), 40.0),
    (40, (400, 160), 40.0),
    (40, (800, 320), 40.0),
    (40, (1600, 640), 40.0),
]


def build_config(Re: float, mesh: tuple[int, int], tfinal: float,
                 render: bool, tag: str) -> Config:
    """Build a per-case config from the base ``config.json``."""
    data = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    Nx, Ny = mesh
    mu = RHO * U_INF * D / float(Re)
    data.update({
        "Nx": Nx, "Ny": Ny,
        "mu": mu,
        "tfinal": float(tfinal),
        # Fine history cadence so the Cl FFT resolves the shedding peak; the
        # heavy per-frame outputs are disabled below to keep the run cheap.
        "output_interval": 0.05,
        "name": tag,
        "output_dir": f"outputs/{tag}",
        "rhie_chow": True,
        "compute_forces": True,
    })
    if not render:
        data.update({
            "save_csv": False, "save_hdf5": False, "save_tecplot": False,
            "save_png": False, "save_mp4": False,
        })
    return Config.from_dict(data)


# --------------------------------------------------------------------------- #
def _stromhart(t, cl, U_inf, D):
    """Strouhal from the Cl time history via a windowed FFT.

    Returns ``(St, f_peak, peak_power)``.  ``St = f_peak * D / U_inf``.
    Returns ``(0.0, 0.0, 0.0)`` when the signal is too short or has no
    significant peak above its noise floor (the steady regime).
    """
    t = np.asarray(t, dtype=float)
    cl = np.asarray(cl, dtype=float)
    mask = np.isfinite(cl)
    t = t[mask]; cl = cl[mask]
    if t.size < 8 or (t[-1] - t[0]) < 1.0:
        return 0.0, 0.0, 0.0
    cl = cl - cl.mean()
    if np.std(cl) < 1e-8:
        return 0.0, 0.0, 0.0
    # Uniform-sample resample (adaptive dt gives near-uniform history anyway).
    n = t.size
    tu = np.linspace(t[0], t[-1], n)
    clu = np.interp(tu, t, cl)
    dt = (t[-1] - t[0]) / (n - 1)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(clu * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    # Ignore the DC/bin-0 and the very-low-frequency drift; consider the
    # physically plausible Strouhal range 0.05 .. 0.30.
    fmin = 0.05 * U_inf / D
    fmax = 0.30 * U_inf / D
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0, 0.0, 0.0
    spec_b = spec.copy()
    spec_b[~band] = 0.0
    spec_b[0] = 0.0
    peak = int(np.argmax(spec_b))
    f_peak = freqs[peak]
    noise = np.median(spec_b[band]) if band.sum() > 3 else 0.0
    power = spec_b[peak]
    if power < 3.0 * (noise + 1e-12):       # below the noise floor -> no shedding
        return 0.0, 0.0, 0.0
    return f_peak * D / U_inf, f_peak, power


def post_process(sim: Simulation, Re: float) -> dict:
    """Extract the benchmark quantities from a finished simulation."""
    h = sim.history
    # In-memory history rows carry Cd/Cl only at output frames; pull the
    # finite ones.
    ts = np.array([r.get("t", 0.0) for r in h], dtype=float)
    Cd = np.array([r.get("Cd", np.nan) for r in h], dtype=float)
    Cl = np.array([r.get("Cl", np.nan) for r in h], dtype=float)
    dt = np.array([r.get("dt", 0.0) for r in h], dtype=float)
    div = np.array([r.get("div", 0.0) for r in h], dtype=float)
    fin = np.isfinite(Cd)
    Cd_f = Cd[fin]; Cl_f = Cl[fin]; t_f = ts[fin]

    steady = Re <= 47.0
    if steady:
        # Steady regime: Cd is the late-time mean; Cl_rms ~ 0; no Strouhal.
        tail = Cd_f[max(0, Cd_f.size // 5):] if Cd_f.size else np.array([])
        Cd_mean = float(tail.mean()) if tail.size else float("nan")
        Cl_rms = float(np.sqrt(np.mean(Cl_f[-max(1, Cl_f.size // 5):]**2))
                       if Cl_f.size else 0.0)
        St = 0.0
    else:
        # Unsteady: Cd_mean over the last 60 % (after transients), Cl_rms
        # over the same window, St from the FFT of the full Cl history.
        cut = max(1, int(0.4 * Cd_f.size))
        Cd_mean = float(Cd_f[cut:].mean()) if Cd_f[cut:].size else float("nan")
        Cl_rms = float(np.sqrt(np.mean(Cl_f[cut:]**2))) if Cl_f[cut:].size else 0.0
        St, _, _ = _stromhart(t_f, Cl_f, U_INF, D)

    # Geometry-derived quantities from the final field.
    Lr_D = 0.0
    theta_sep = float("nan")
    if sim.forces_calc is not None:
        Lr = sim.forces_calc.recirculation_length(sim.state.u)
        Lr_D = float(Lr / D)
        if steady:
            theta_sep = sim.forces_calc.separation_angle_deg(sim.state.u,
                                                            sim.state.v)

    res = {
        "Re": Re,
        "Nx": sim.cfg.Nx, "Ny": sim.cfg.Ny,
        "cells": sim.cfg.Nx * sim.cfg.Ny,
        "steps": sim.step_count,
        "mean_dt": float(np.mean(dt[np.isfinite(dt) & (dt > 0)])) if np.any(dt > 0) else 0.0,
        "max_div": float(np.abs(div).max()) if div.size else 0.0,
        "Cd_mean": Cd_mean,
        "Cl_rms": float(Cl_rms),
        "St": float(St),
        "Lr_D": Lr_D,
        "theta_sep": theta_sep,
        "steady": steady,
    }
    return res


# --------------------------------------------------------------------------- #
def run_case(Re, mesh, tfinal, render=False):
    tag = f"cylinder_Re{int(Re)}_{mesh[0]}x{mesh[1]}"
    cfg = build_config(Re, mesh, tfinal, render, tag)
    t0 = time.perf_counter()
    sim = Simulation(cfg)
    sim.run()
    wall = time.perf_counter() - t0
    res = post_process(sim, Re)
    res["wall_s"] = wall
    res["tag"] = tag
    bench = benchmark_for(Re)
    res["Cd_pct"] = pct(res["Cd_mean"], bench.Cd)
    res["St_pct"] = pct(res["St"], bench.St)
    res["Lr_pct"] = pct(res["Lr_D"], bench.Lr_D)
    res["ref"] = bench
    _print_case(res)
    return res


def _print_case(r):
    print(f"\n--- {r['tag']} (Re={r['Re']}, {r['Nx']}x{r['Ny']}, "
          f"{r['cells']} cells, {r['steps']} steps, {r['wall_s']:.1f}s) ---")
    print(f"  mean dt = {r['mean_dt']:.4e} s,  max|div| = {r['max_div']:.3e}")
    print(f"  Cd_mean = {r['Cd_mean']:.4f}  (ref {r['ref'].Cd:.3f}, "
          f"{r['Cd_pct']:+.1f} %)")
    if r["steady"]:
        print(f"  Lr/D    = {r['Lr_D']:.4f}  (ref {r['ref'].Lr_D:.3f}, "
              f"{r['Lr_pct']:+.1f} %)")
        if not math.isnan(r["theta_sep"]):
            print(f"  theta_sep = {r['theta_sep']:.1f} deg  (ref {r['ref'].theta_sep:.1f})")
    else:
        print(f"  Cl_rms  = {r['Cl_rms']:.4f}  (ref {r['ref'].Cl_rms:.3f})")
        print(f"  St      = {r['St']:.4f}  (ref {r['ref'].St:.3f}, "
              f"{r['St_pct']:+.1f} %)")


# --------------------------------------------------------------------------- #
def write_report(results, out_path: Path):
    lines = []
    lines.append("# Flow Past a Circular Cylinder — CFDPy validation report\n")
    lines.append(f"Generated by `examples/cylinder_flow/run_reynolds.py`.\n")
    lines.append("Geometry: D=1, Lx=22D, Ly=4.1D, cylinder centre (5.0, 2.05); "
                "west inlet U∞=1, east outlet, top/bottom slip, cylinder no-slip "
                "(staircase direct-forcing). Rhie-Chow coupling on.\n")
    lines.append("\n## Summary table\n")
    lines.append("| Re | mesh | cells | steps | mean dt | max\\|div\\| | "
                "Cd_mean | Cd ref | Cd % | Cl_rms | St | St ref | St % | "
                "Lr/D | Lr ref | Lr % | wall (s) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        b = r["ref"]
        lines.append(
            f"| {int(r['Re'])} | {r['Nx']}×{r['Ny']} | {r['cells']} | "
            f"{r['steps']} | {r['mean_dt']:.2e} | {r['max_div']:.1e} | "
            f"{r['Cd_mean']:.3f} | {b.Cd:.3f} | {r['Cd_pct']:+.1f} | "
            f"{r['Cl_rms']:.3f} | {r['St']:.4f} | {b.St:.3f} | {r['St_pct']:+.1f} | "
            f"{r['Lr_D']:.3f} | {b.Lr_D:.3f} | {r['Lr_pct']:+.1f} | "
            f"{r['wall_s']:.0f} |"
        )

    lines.append("\n## Per-case blocks\n")
    for r in results:
        b = r["ref"]
        lines.append(f"### Re = {int(r['Re'])}  ({r['Nx']}×{r['Ny']}, "
                     f"{r['cells']} cells)\n")
        lines.append(f"- time steps: **{r['steps']}**, mean dt = "
                     f"{r['mean_dt']:.4e} s, compute time = {r['wall_s']:.1f} s")
        lines.append(f"- max divergence residual (face-averaged, collocated): "
                     f"{r['max_div']:.3e}")
        lines.append(f"- Cd_mean = **{r['Cd_mean']:.4f}**  "
                     f"(literature {b.Cd:.3f}; {r['Cd_pct']:+.1f} %)")
        if r["steady"]:
            lines.append(f"- Lr/D = **{r['Lr_D']:.4f}**  "
                         f"(literature {b.Lr_D:.3f}; {r['Lr_pct']:+.1f} %)")
            if not math.isnan(r["theta_sep"]):
                lines.append(f"- separation angle = {r['theta_sep']:.1f}°  "
                             f"(literature {b.theta_sep:.1f}°)")
            lines.append("- Cl_rms / Strouhal: n/a (steady regime, no shedding)")
        else:
            lines.append(f"- Cl_rms = **{r['Cl_rms']:.4f}**  "
                         f"(literature {b.Cl_rms:.3f})")
            lines.append(f"- Strouhal = **{r['St']:.4f}**  "
                         f"(literature {b.St:.3f}; {r['St_pct']:+.1f} %)")
        q = _quality(r)
        lines.append(f"- quality assessment: {q}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}")


def _quality(r):
    cd_ok = abs(r["Cd_pct"]) <= 25.0
    if r["steady"]:
        lr_ok = abs(r["Lr_pct"]) <= 40.0
        parts = []
        parts.append("Cd " + ("OK" if cd_ok else "OFF"))
        parts.append("Lr " + ("OK" if lr_ok else "OFF"))
        return ", ".join(parts) + " (steady)"
    st_ok = abs(r["St_pct"]) <= 15.0
    parts = []
    parts.append("Cd " + ("OK" if cd_ok else "OFF"))
    parts.append("St " + ("OK" if st_ok else "OFF"))
    return ", ".join(parts) + " (shedding)"


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="run the full Re sweep (expensive)")
    ap.add_argument("--mesh-study", action="store_true",
                    help="run the Re=40 mesh-independence study")
    ap.add_argument("--cases", default=None,
                    help="comma-separated Re list, e.g. 40,100")
    ap.add_argument("--mesh", default=None,
                    help="Nx x Ny mesh, e.g. 400x160")
    ap.add_argument("--tfinal", default=None,
                    help="comma-separated tfinal per case, e.g. 40,80")
    ap.add_argument("--render", action="store_true",
                    help="render MP4 / PNG / Tecplot / HDF5 outputs (slow)")
    ap.add_argument("--no-report", action="store_true",
                    help="skip writing cylinder_report.md")
    args = ap.parse_args(argv)

    if args.mesh_study:
        cases = MESH_STUDY
        tag = "mesh_study"
    elif args.full:
        cases = FULL
        tag = "full_sweep"
    elif args.cases:
        Res = [float(x) for x in args.cases.split(",")]
        if args.mesh:
            Nx, Ny = (int(v) for v in args.mesh.lower().split("x"))
            mesh = (Nx, Ny)
        else:
            mesh = (400, 160)
        tfs = ([float(x) for x in args.tfinal.split(",")]
               if args.tfinal else [80.0] * len(Res))
        cases = [(Re, mesh, tf) for Re, tf in zip(Res, tfs)]
        tag = "custom"
    else:
        cases = VALIDATION
        tag = "validation"

    print(f"Running {len(cases)} case(s): {[(int(r), m, tf) for r, m, tf in cases]}")
    results = []
    total = 0.0
    for (Re, mesh, tf) in cases:
        r = run_case(Re, mesh, tf, render=args.render)
        results.append(r)
        total += r["wall_s"]
    print(f"\nTotal wall time: {total:.1f} s ({total/60:.1f} min)")
    if not args.no_report:
        out = ROOT / "examples" / "cylinder_flow" / "cylinder_report.md"
        write_report(results, out)


if __name__ == "__main__":
    main()