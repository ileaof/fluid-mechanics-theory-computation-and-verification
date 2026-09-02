"""Re-export Tecplot ``.dat`` frames from the HDF5 snapshots of a case.

Utility for refreshing (or back-filling) the Tecplot export after a change to
:class:`visualization.tecplot_writer.TecplotExporter` -- e.g. the 3-D axis
convention (Tecplot is Z-up; the framework is y-up) -- without re-running the
simulation.  Every ``frame_*.h5`` in the case's ``output_dir`` is rewritten in
place as ``frame_XXXXXX.dat``.

Usage (from the package root)::

    python -m visualization.reexport_tecplot examples/liquid_drop_splash_3D/config.json
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

from visualization.tecplot_writer import TecplotExporter


class _FrameMesh:
    """Minimal mesh view reconstructed from a frame's coordinate grids."""

    is_2d = False

    def __init__(self, Xg: np.ndarray) -> None:
        self.Nx, self.Ny, self.Nz = Xg.shape
        self.Xc, self.Yc, self.Zc = Xg, None, None  # filled per frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-export Tecplot .dat frames from HDF5 snapshots.")
    ap.add_argument("config", help="Path to the case config JSON.")
    args = ap.parse_args(argv)

    import json
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out_dir = cfg["output_dir"]
    frames = sorted(f for f in (os.path.join(out_dir, b) for b in
                                os.listdir(out_dir) if b.endswith(".h5")
                                and b.startswith("frame_")))
    if not frames:
        print(f"no frame_*.h5 in {out_dir}", file=sys.stderr)
        return 1

    for p in frames:
        with h5py.File(p, "r") as fh:
            t = float(fh.attrs["time"])
            Xg = fh["X"][...]
            Yg = fh["Y"][...]
            Zg = fh["Z"][...]
            u = fh["u"][...]
            v = fh["v"][...]
            w = fh["w"][...]
            pv = fh["p"][...]
            tv = fh["T"][...]
            av = fh["alpha"][...]
        mesh = _FrameMesh(Xg)
        mesh.Yc, mesh.Zc = Yg, Zg
        fname = os.path.basename(p).replace(".h5", ".dat")
        path = TecplotExporter(mesh, out_dir).write(
            t, u, v, w, pv, tv, av, fname=fname)
        print(f"{path}", flush=True)
    print(f"re-exported {len(frames)} frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())