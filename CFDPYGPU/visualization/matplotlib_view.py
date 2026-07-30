"""Matplotlib visualisation / animation back-end.

:class:`MatplotlibViewer` accumulates snapshots (pressure, temperature,
velocity, VOF) during a simulation and can:

* render single-frame contour/quiver/streamline plots;
* save PNG snapshots;
* assemble an MP4 animation of any field.

The viewer holds no simulation state: it only consumes the snapshot
dictionaries produced by :class:`Simulation`.  This keeps the visualisation
fully decoupled from the solver and lets the same viewer be reused for
post-processing a saved HDF5 history.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Cached path to the first *usable* ffmpeg found, or "" if none (see
# :func:`_usable_ffmpeg`).  ``None`` = not yet probed.
_ffmpeg_path_cache: str | None = None


def _probe_ffmpeg(path: str) -> bool:
    """Return ``True`` if ``path`` is an ffmpeg that accepts ``-framerate``.

    Matplotlib's :class:`~matplotlib.animation.FFMpegWriter` drives ffmpeg with
    the ``-framerate`` input option, which only modern builds accept.  Some
    third-party bundles ship an ancient ffmpeg (e.g. Tecplot 360's
    ``SVN-r434``, circa 2008) that rejects it with *"Unrecognized option
    'framerate'"*, prints that to the console, and makes a successful GIF
    fallback look like a failure.  ``-framerate 1 -h`` exits 0 on a modern
    build and non-zero on the ancient one; output is captured so the
    rejection never leaks.
    """
    try:
        r = subprocess.run([path, "-framerate", "1", "-h"],
                           capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _usable_ffmpeg() -> str | None:
    """Return the path of the first *usable* ffmpeg, or ``None``.

    We must NOT trust plain ``shutil.which("ffmpeg")``: on Windows the machine
    PATH is prepended to the user PATH, so a broken bundled ffmpeg (Tecplot)
    can shadow a good one (Gyan) that the user installed on their user PATH.
    Instead every ``ffmpeg``/``ffmpeg.exe`` on ``PATH`` is probed in order and
    the first that accepts ``-framerate`` wins.  An explicit ``CFDPY_FFMPEG``
    environment variable, if set, is tried first and overrides the search.

    The result is cached for the process.  When a usable ffmpeg is found it is
    also published to ``matplotlib.rcParams['animation.ffmpeg_path']`` so the
    :class:`FFMpegWriter` uses *that* binary rather than re-resolving PATH.
    """
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is not None:
        return _ffmpeg_path_cache or None
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates: list[str] = []
    override = os.environ.get("CFDPY_FFMPEG")
    if override:
        candidates.append(override)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        cand = os.path.join(d, exe)
        if os.path.isfile(cand):
            candidates.append(os.path.abspath(cand))
    chosen: str | None = None
    seen: set[str] = set()
    for cand in candidates:
        key = os.path.normcase(os.path.abspath(cand))
        if key in seen:
            continue
        seen.add(key)
        if _probe_ffmpeg(cand):
            chosen = cand
            break
    _ffmpeg_path_cache = chosen or ""
    if chosen:
        try:
            import matplotlib as mpl
            mpl.rcParams["animation.ffmpeg_path"] = chosen
        except Exception:
            pass
    return chosen


def _ffmpeg_usable() -> bool:
    """Return ``True`` if a usable ffmpeg was found (cached; see
    :func:`_usable_ffmpeg`)."""
    return _usable_ffmpeg() is not None


@dataclass
class Frame:
    """A single saved snapshot."""
    time: float
    fields: dict[str, np.ndarray]


class MatplotlibViewer:
    """Plot and animate simulation snapshots with Matplotlib."""

    def __init__(self, mesh, output_dir: str = "outputs", dpi: int = 150,
                 save_png: bool = True, save_mp4: bool = True,
                 fps: int = 20) -> None:
        self.mesh = mesh
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dpi = dpi
        self.save_png = save_png
        self.save_mp4 = save_mp4
        self.fps = fps
        self.frames: list[Frame] = []
        self._mpl = None
        self._plt = None

    # ------------------------------------------------------------------ #
    def _ensure_mpl(self):
        if self._mpl is None:
            import matplotlib
            matplotlib.use("Agg")          # headless-safe backend
            import matplotlib.pyplot as plt
            self._mpl = matplotlib
            self._plt = plt
        return self._plt

    # ------------------------------------------------------------------ #
    def add_frame(self, time: float, fields: dict[str, np.ndarray]) -> None:
        """Store a snapshot (a copy of each field)."""
        self.frames.append(Frame(time=time, fields={k: np.array(v)
                                                     for k, v in fields.items()}))

    # ------------------------------------------------------------------ #
    @property
    def X(self) -> np.ndarray:
        return self.mesh.Xc[:, :, 0]

    @property
    def Y(self) -> np.ndarray:
        return self.mesh.Yc[:, :, 0]

    # ------------------------------------------------------------------ #
    def plot_field(self, field: np.ndarray, title: str, fname: str,
                   cmap: str = "viridis", vectors: tuple | None = None,
                   streamlines: bool = False, alpha: np.ndarray | None = None,
                   vmin=None, vmax=None) -> str:
        """Render one 2D field to a PNG file; return the file path.

        ``self.X`` and ``self.Y`` are the cell-centre coordinate grids
        ``(Nx, Ny)`` (axis 0 = x, axis 1 = y); fields are kept in the same
        layout and plotted with ``shading='nearest'`` so non-square grids are
        handled correctly.
        """
        plt = self._ensure_mpl()
        fig, ax = plt.subplots(figsize=(6, 5))
        F = field[:, :, 0] if field.ndim == 3 else field
        pcm = ax.pcolormesh(self.X, self.Y, F, shading="nearest", cmap=cmap,
                            vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=title)
        if vectors is not None:
            u, v = vectors
            U = u[:, :, 0] if u.ndim == 3 else u
            V = v[:, :, 0] if v.ndim == 3 else v
            step = max(1, self.mesh.Nx // 25)
            ax.quiver(self.X[::step, ::step], self.Y[::step, ::step],
                      U[::step, ::step], V[::step, ::step],
                      color="white", scale=20.0, width=0.004)
        if streamlines and vectors is not None:
            u, v = vectors
            U = u[:, :, 0] if u.ndim == 3 else u
            V = v[:, :, 0] if v.ndim == 3 else v
            # Matplotlib streamplot wants u,v of shape (Ny, Nx) with 1-D x (len Nx)
            # and y (len Ny); our cell-centred fields are (Nx, Ny) so transpose.
            try:
                ax.streamplot(self.X[:, 0], self.Y[0, :],
                              U.T, V.T, color="white", density=1.2,
                              linewidth=0.8, arrowsize=0.8)
            except Exception:
                pass
        if alpha is not None:
            A = alpha[:, :, 0] if alpha.ndim == 3 else alpha
            ax.contour(self.X, self.Y, A, levels=[0.5], colors="red",
                       linewidths=1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.tight_layout()
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def animate(self, key: str, fname: str, cmap: str = "viridis",
                title: str | None = None, fixed_scale: bool = True) -> str:
        """Build an MP4 animation of the field ``key`` over all stored frames."""
        plt = self._ensure_mpl()
        if not self.frames:
            return ""
        fig, ax = plt.subplots(figsize=(6, 5))
        first = self.frames[0].fields[key]
        if fixed_scale:
            allv = np.concatenate([f.fields[key].ravel() for f in self.frames])
            vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        else:
            vmin, vmax = None, None
        pcm = ax.pcolormesh(self.X, self.Y, first[:, :, 0], shading="nearest",
                            cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=title or key)
        txt = ax.set_title("")

        def update(i):
            F = self.frames[i].fields[key][:, :, 0]
            pcm.set_array(F.ravel())
            txt.set_text(f"{title or key}   t = {self.frames[i].time:.3f} s")
            return pcm, txt

        try:
            import matplotlib.animation as animation
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(self.frames),
                                           interval=1000 / self.fps,
                                           blit=False)
            path = os.path.join(self.output_dir, fname)
            # Prefer an MP4 (mpeg4 codec) when a usable ffmpeg is present; fall
            # back to a dependency-free Pillow GIF otherwise.  The capability
            # probe avoids invoking a broken bundled ffmpeg (which would print
            # a misleading "Unrecognized option 'framerate'" to the console).
            written = False
            if self.save_mp4 and _ffmpeg_usable():
                try:
                    writer = animation.FFMpegWriter(fps=self.fps, codec="mpeg4")
                    anim.save(path, dpi=self.dpi, writer=writer)
                    written = True
                except Exception:
                    written = False
            if not written:
                path = os.path.join(self.output_dir,
                                    fname.rsplit(".", 1)[0] + ".gif")
                writer = animation.PillowWriter(fps=self.fps)
                anim.save(path, dpi=self.dpi, writer=writer)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"[viewer] animation failed: {exc}")
            path = ""
        finally:
            plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def _flow_bg(self, fields: dict, background: str) -> np.ndarray:
        """Return the scalar background field for a flow animation.

        ``background`` is one of ``"speed"``, ``"u"``, ``"v"``, ``"pressure"``,
        ``"temperature"`` or ``"vorticity"`` (the last needs the postprocessor
        and is computed on the fly).  Default is the velocity magnitude.
        """
        u = fields["u"]; v = fields["v"]
        U = u[:, :, 0] if u.ndim == 3 else u
        V = v[:, :, 0] if v.ndim == 3 else v
        W = None
        if "w" in fields and fields["w"] is not None:
            w = fields["w"]
            W = w[:, :, 0] if w.ndim == 3 else w
        if background == "u":
            return U
        if background == "v":
            return V
        if background == "pressure":
            p = fields["p"]
            return p[:, :, 0] if p.ndim == 3 else p
        if background == "temperature":
            T = fields["T"]
            return T[:, :, 0] if T.ndim == 3 else T
        # default: speed magnitude |u|
        spd = U * U + V * V
        if W is not None:
            spd = spd + W * W
        return np.sqrt(spd)

    # ------------------------------------------------------------------ #
    def animate_flow(self, fname: str, background: str = "speed",
                     cmap: str = "viridis", title: str | None = None,
                     with_quiver: bool = True, with_streamlines: bool = True,
                     fixed_scale: bool = True, quiver_step: int | None = None,
                     quiver_scale: float | None = None) -> str:
        """Animate the velocity field over all stored frames.

        A scalar *background* (default the velocity magnitude ``|u|``) is
        shown as a colour map; the instantaneous velocity field is overlaid as
        white quiver arrows and/or streamlines so recirculation zones,
        reattachment points and the development of the flow are all visible
        in the animation.

        Parameters
        ----------
        background:
            Scalar field rendered as the colour map: ``"speed"`` (|u|),
            ``"u"``, ``"v"``, ``"pressure"``, ``"temperature"``.
        with_quiver, with_streamlines:
            Toggle the vector/quiver and streamline overlays.
        fixed_scale:
            Fix the colour-map range across all frames (recommended so the
            eye sees the field grow / decay consistently).
        quiver_step, quiver_scale:
            Sub-sampling stride and arrow scale for the quiver overlay
            (defaults to ~25 arrows across the domain and ``scale=20``).
        """
        plt = self._ensure_mpl()
        if not self.frames:
            return ""
        fig, ax = plt.subplots(figsize=(7, 5))

        # Pre-compute the background per frame (and the fixed colour range).
        if fixed_scale:
            bgs = [self._flow_bg(f.fields, background) for f in self.frames]
            allv = np.concatenate([b.ravel() for b in bgs])
            vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        else:
            bgs = None
            vmin, vmax = None, None

        bg0 = (bgs[0] if fixed_scale
               else self._flow_bg(self.frames[0].fields, background))
        pcm = ax.pcolormesh(self.X, self.Y, bg0, shading="nearest", cmap=cmap,
                            vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=title or background)

        # Quiver overlay (created once, updated each frame with set_UVC).
        step = quiver_step or max(1, self.mesh.Nx // 25)
        scale = quiver_scale or 20.0
        u0 = self.frames[0].fields["u"]
        v0 = self.frames[0].fields["v"]
        U0 = u0[:, :, 0] if u0.ndim == 3 else u0
        V0 = v0[:, :, 0] if v0.ndim == 3 else v0
        Q = None
        if with_quiver:
            Q = ax.quiver(self.X[::step, ::step], self.Y[::step, ::step],
                          U0[::step, ::step], V0[::step, ::step],
                          color="white", scale=scale, width=0.004)

        # Streamline overlay (redrawn each frame -- streamplot has no in-place
        # update, so the previous artists are removed first).
        stream_arts: list = []
        txt = ax.set_title("")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def _clear_stream() -> None:
            for art in stream_arts:
                try:
                    art.remove()
                except Exception:
                    pass
            stream_arts.clear()

        def update(i):
            f = self.frames[i].fields
            bg = bgs[i] if fixed_scale else self._flow_bg(f, background)
            pcm.set_array(bg.ravel())
            U = f["u"][:, :, 0] if f["u"].ndim == 3 else f["u"]
            V = f["v"][:, :, 0] if f["v"].ndim == 3 else f["v"]
            if Q is not None:
                Q.set_UVC(U[::step, ::step], V[::step, ::step])
            if with_streamlines:
                _clear_stream()
                try:
                    sp = ax.streamplot(self.X[:, 0], self.Y[0, :],
                                       U.T, V.T, color="white", density=1.2,
                                       linewidth=0.8, arrowsize=0.8)
                    stream_arts.extend([sp.lines, sp.arrows])
                except Exception:
                    pass
            txt.set_text(f"{title or background}   t = {self.frames[i].time:.3f} s")
            return pcm, txt

        path = ""
        try:
            import matplotlib.animation as animation
            anim = animation.FuncAnimation(fig, update,
                                           frames=len(self.frames),
                                           interval=1000 / self.fps,
                                           blit=False)
            path = os.path.join(self.output_dir, fname)
            written = False
            if self.save_mp4 and _ffmpeg_usable():
                try:
                    writer = animation.FFMpegWriter(fps=self.fps, codec="mpeg4")
                    anim.save(path, dpi=self.dpi, writer=writer)
                    written = True
                except Exception:
                    written = False
            if not written:
                path = os.path.join(self.output_dir,
                                    fname.rsplit(".", 1)[0] + ".gif")
                writer = animation.PillowWriter(fps=self.fps)
                anim.save(path, dpi=self.dpi, writer=writer)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"[viewer] flow animation failed: {exc}")
            path = ""
        finally:
            plt.close(fig)
        return path

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._ensure_mpl()
        self._plt.close("all")