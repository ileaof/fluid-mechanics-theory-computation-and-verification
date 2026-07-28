"""Compute-only profiler for CFDPy: identify the per-step hotspots.

Loads a case config, disables every I/O / plotting back-end so only the
numerical work is measured, runs a fixed number of time steps under cProfile,
and prints the cumulative-time ranking.  Run from the CFDPYGPU root::

    python tools/profile_hotspots.py examples/natural_convection_2D/config.json 40

The case path is relative to the current directory, so run it from the
CFDPYGPU root.
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

import numpy as np

# Put the package root (this file's parent dir) on sys.path so the
# ``config``/``main`` package imports resolve regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from main import Simulation


def _strip_io(cfg):
    """Turn off all output/plotting so the profile reflects compute only."""
    cfg.save_csv = False
    cfg.save_hdf5 = False
    cfg.save_tecplot = False
    cfg.save_png = False
    cfg.save_mp4 = False
    cfg.verbose = False
    return cfg


def main(argv):
    path = argv[1] if len(argv) > 1 else "examples/natural_convection_2D/config.json"
    nsteps = int(argv[2]) if len(argv) > 2 else 40
    cfg = load_config(path)
    cfg = _strip_io(cfg)
    sim = Simulation(cfg)
    sim.initialize()

    # warm-up a couple of steps (JIT compile, caches, lazy matrix builds)
    for _ in range(3):
        sim.step()

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(nsteps):
        sim.step()
    pr.disable()
    wall = time.perf_counter() - t0

    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print(f"\n=== Profile: {path}  ({nsteps} steps, {wall:.3f}s, "
          f"{1000*wall/nsteps:.2f} ms/step) ===")
    st.print_stats(35)
    st.sort_stats("tottime")
    print("\n=== By tottime ===")
    st.print_stats(25)
    print(f"\nper-step wall = {1000*wall/nsteps:.3f} ms   "
          f"(grid {sim.mesh.Nx}x{sim.mesh.Ny}x{sim.mesh.Nz})")
    return sim


if __name__ == "__main__":
    main(sys.argv)