"""3-D VOF frame renderer: HDF5 frames -> PNG stills -> MP4 animation.

The 2-D :mod:`visualization.matplotlib_view` pipeline (contours / quiver /
streamplot) does not extend to volumetric data, so this module renders the
3-D free surface directly:

* the **pool / free-surface height map** -- for every (x, z) column the
  topmost cell with ``alpha > 0.5`` gives the interface height, drawn as a
  shaded surface (this is the classic "crown" view of a splash);
* the **interface shell** -- every cell with ``0.05 < alpha < 0.95`` (the
  PLIC-reconstructed interface band) is drawn as a point cloud coloured by
  the local flow speed, which shows the drop, the ejecta sheet and the
  crown jets as they detach from the surface;
* a wireframe of the closed tank for spatial reference.

The displayed axes follow the same Z-up convention as the Tecplot export
(:mod:`visualization.tecplot_writer`): the fluid level is drawn in the X-Y
plane and the height along Z (display X <- framework x, Y <- framework z,
Z <- framework y).

Usage (from the package root)::

    python -m visualization.render_vof_3d examples/liquid_drop_splash_3D/config.json

Reads ``frame_XXXXXX.h5`` from the case's ``output_dir`` (as written by
: meth:`main.Simulation._save_frame` with ``save_hdf5: true``) and writes
``render_3d/frame_XXXXXX.png`` plus, when ffmpeg is available, a single
``<name>_3d.mp4`` (H.264, yuv420p) at ``--fps`` frames per second.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _find_ffmpeg() -> str | None:
    """Locate a capable ffmpeg executable.

    On Windows the first ``ffmpeg`` on PATH is often an application-bundled
    build (e.g. Tecplot 360 ships a 2008-era binary without ``-framerate``),
    so every candidate from the PATH search is probed and a modern one is
    preferred (full/winget builds first, then bare ``ffmpeg``).
    """
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = shutil.which(exe)          # first on PATH
    try:                               # ... and all of them (`where` lists all)
        out = subprocess.run(["where", exe] if os.name == "nt"
                             else ["which", "-a", exe],
                             capture_output=True, text=True)
        cands = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        cands = []
    if found and found not in cands:
        cands.insert(0, found)
    if not cands:
        return None
    # Prefer a modern build: winget / full_build installs first, PATH first
    # entry next, the rest in listing order.
    def rank(c: str) -> int:
        low = c.lower()
        if "winget" in low or "full_build" in low:
            return 0
        if low.endswith(exe) and c == shutil.which(exe):
            return 1
        return 2
    cands.sort(key=rank)
    return cands[0]


def _load_frame(path: str, source: str) -> dict:
    """Load one snapshot from an HDF5 or a Tecplot ``.dat`` frame.

    Both sources return the *framework-convention* fields: ``(x, y, z)``
    arrays with y vertical, so the rest of the renderer is source-agnostic.
    The Tecplot zone is in the exported Z-up order ``(I, J, K) = (x, z, y)``
    with ``V <- w``, ``W <- v`` (see :mod:`visualization.tecplot_writer`),
    which is inverted here.
    """
    if source == "h5":
        with h5py.File(path, "r") as fh:
            t = float(fh.attrs["time"])
            alpha = fh["alpha"][...]
            u, v, w = fh["u"][...], fh["v"][...], fh["w"][...]
            # X/Y/Z are stored as full (Nx, Ny, Nz) coordinate grids; the 1-D
            # cell-centre axes are constant along the other two directions.
            xc = np.asarray(fh["X"][...])[:, 0, 0]
            yc = np.asarray(fh["Y"][...])[0, :, 0]
            zc = np.asarray(fh["Z"][...])[0, 0, :]
        return dict(t=t, alpha=alpha, u=u, v=v, w=w, xc=xc, yc=yc, zc=zc)

    # Tecplot ASCII ORDERED POINT zone: 3 header lines, then one record per
    # node ordered I (x) fastest, J, K.  Only Alpha (and the velocity
    # components) are needed; coordinates come from the axis extents.
    with open(path, encoding="utf-8") as fh:
        header = [next(fh) for _ in range(3)]
    m = re.search(r't=([\d.eE+-]+)', header[2])
    t = float(m.group(1)) if m else 0.0
    # The zone data may be followed by embedded style commands ($!Global...);
    # cut the record list at the first macro line before parsing.
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.startswith("$")),
               len(lines))
    data = np.loadtxt(lines[3:end])
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
    U, V, W = data[:, 3], data[:, 4], data[:, 5]
    nx = np.unique(X).size          # I axis: framework x
    nj = np.unique(Y).size          # J axis: framework z (Tecplot Y)
    nk = np.unique(Z).size          # K axis: framework y (Tecplot Z, vertical)

    def fw(col):                    # (nx, nj, nk) Z-up -> framework (x, y, z)
        return np.ascontiguousarray(
            col.reshape((nx, nj, nk), order="F").transpose(0, 2, 1))

    alpha = fw(data[:, 8])
    u = fw(U)
    v = fw(W)                       # W column holds framework v (V/W swapped)
    w = fw(V)                       # V column holds framework w
    return dict(t=t, alpha=alpha, u=u, v=v, w=w,
                xc=np.unique(X), yc=np.unique(Z), zc=np.unique(Y))


def render_frame(path: str, ax, source: str = "h5"):
    """Draw one frame (HDF5 or Tecplot) into the (already created) 3-D axis.

    Returns ``(t, surface_artist, scatter_artist_or_None)`` for colour bars.
    """
    f = _load_frame(path, source)
    t, alpha = f["t"], f["alpha"]
    u, v, w = f["u"], f["v"], f["w"]
    xc, yc, zc = f["xc"], f["yc"], f["zc"]
    ny = alpha.shape[1]

    # ---- free-surface height map (topmost alpha>0.5 cell per (x,z) column),
    # restricted to the water *connected to the tank floor* (pool + crown).
    # Without this, a column under an airborne drop takes the drop cell as its
    # top surface and plot_surface draws a vertical curtain joining the drop
    # to the pool -- the drop looks fused with the reservoir.  Airborne water
    # (the falling drop, ejecta) is shown by the interface point cloud below.
    from scipy import ndimage
    wet = alpha > 0.5
    lab, _ = ndimage.label(wet)                     # 6-connectivity
    bottom_ids = np.unique(lab[:, 0, :])
    bottom_ids = bottom_ids[bottom_ids != 0]
    pool = np.isin(lab, bottom_ids) if bottom_ids.size else wet
    any_col = pool.any(axis=1)
    jtop = ny - 1 - np.argmax(pool[:, ::-1, :], axis=1)
    height = np.where(any_col, yc[jtop], np.nan)

    # ---- display convention: match the Tecplot export (Z-up).  The level
    # plane is drawn as X-Y and the height along Z: display X <- framework x,
    # Y <- framework z, Z <- framework y (the vertical).
    XX, YY = np.meshgrid(xc, zc, indexing="ij")
    ax.clear()
    surf = ax.plot_surface(XX, YY, height, cmap="YlGnBu", vmin=0.0,
                           vmax=float(np.nanmax(height)), rstride=1, cstride=1,
                           linewidth=0, antialiased=True, alpha=0.9)

    # ---- interface shell (interface-band cells), coloured by flow speed
    band = (alpha > 0.05) & (alpha < 0.95)
    ii, jj, kk = np.nonzero(band)
    if ii.size:
        speed = np.sqrt(u[ii, jj, kk] ** 2 + v[ii, jj, kk] ** 2
                        + w[ii, jj, kk] ** 2)
        s = ax.scatter(xc[ii], zc[kk], yc[jj], c=speed, cmap="inferno",
                       s=4, vmin=0.0, vmax=4.0, depthshade=False)
        return t, surf, s
    return t, surf, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render 3-D VOF HDF5 frames to PNG stills + MP4.")
    ap.add_argument("config", help="Path to the case config JSON.")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="MP4 frame rate (default 8).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Render every Nth frame (default all).")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--source", choices=("h5", "dat"), default="h5",
                    help="render from the HDF5 snapshots (default) or "
                         "directly from the exported Tecplot .dat frames.")
    ap.add_argument("--name", default=None,
                    help="movie file name (default <case>_3d.mp4).")
    args = ap.parse_args(argv)

    import json
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out_dir = cfg["output_dir"]
    ext = ".h5" if args.source == "h5" else ".dat"
    frames = sorted(
        f for f in (os.path.join(out_dir, b) for b in
                    os.listdir(out_dir) if b.endswith(ext)
                    and b.startswith("frame_")))
    frames = frames[::max(1, args.stride)]
    if not frames:
        print(f"no frame_*{ext} in {out_dir}", file=sys.stderr)
        return 1

    lx, ly, lz = cfg.get("Lx", 1.0), cfg.get("Ly", 1.5), cfg.get("Lz", 1.0)
    rend_dir = os.path.join(out_dir, "render_3d")
    os.makedirs(rend_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 9})
    for n, path in enumerate(frames):
        fig = plt.figure(figsize=(7.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        t, surf, sc = render_frame(path, ax, source=args.source)
        # Display box follows the Tecplot Z-up convention: level plane X-Y,
        # height Z (display Y <- framework z, display Z <- framework y).
        ax.set_box_aspect((lx, lz, ly))
        ax.set_xlim(0, lx); ax.set_ylim(0, lz); ax.set_zlim(0, ly)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"{cfg.get('name', 'case')}   t = {t:.3f} s")
        c1 = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)
        c1.set_label("free-surface height (m)")
        if sc is not None:
            c2 = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.12)
            c2.set_label("|u| (m/s)")
        png = os.path.join(rend_dir, f"f_{n:04d}.png")
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[{n+1}/{len(frames)}] {png}", flush=True)

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("ffmpeg not found -- PNG stills only.", file=sys.stderr)
        return 0
    mp4 = os.path.join(out_dir, args.name
                       or f"{cfg.get('name', 'case')}_3d.mp4")
    pat = os.path.join(rend_dir, "f_%04d.png")
    # yuv420p needs even dimensions: crop the (bbox-cropped) PNGs down.
    subprocess.run([ffmpeg, "-y", "-framerate", f"{args.fps}", "-i", pat,
                    "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", mp4], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())