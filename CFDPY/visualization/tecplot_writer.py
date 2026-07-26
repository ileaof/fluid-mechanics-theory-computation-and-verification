"""Tecplot 360 ASCII ``.dat`` exporter.

Writes a structured IJ-ordered (2D) or IJK-ordered (3D) FE/point dataset in the
Tecplot ASCII format understood by Tecplot 360.  Each frame is exported with the
variables::

    X  Y  Z  U  V  W  Pressure  Temperature  Alpha

The exporter is intentionally ASCII-only (no binary Tecplot variant) to match
the project specification and remain trivially diffable.  Files are written
with a single zone per file (one time step) so they can be appended into a
time-sequence by the :class:`Simulation` loop.
"""

from __future__ import annotations

import os

import numpy as np


class TecplotExporter:
    """Write Tecplot 360 ASCII ``.dat`` files for each snapshot."""

    def __init__(self, mesh, output_dir: str = "outputs") -> None:
        self.mesh = mesh
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def write(self, time: float, u, v, w, p, T, alpha=None,
              fname: str | None = None) -> str:
        """Export one snapshot to ``fname`` (auto-named if omitted)."""
        mesh = self.mesh
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        if fname is None:
            fname = f"frame_{time:08.4f}.dat"
        path = os.path.join(self.output_dir, fname)

        if w is None:
            w = np.zeros_like(u)
        if alpha is None:
            alpha = np.zeros_like(u)

        lines: list[str] = []
        title = f"CFDPY snapshot t={time:.6f}"
        lines.append(f'TITLE = "{title}"')
        varnames = ["X", "Y", "Z", "U", "V", "W", "Pressure", "Temperature", "Alpha"]
        lines.append('VARIABLES = "' + '" "'.join(varnames) + '"')
        if mesh.is_2d:
            lines.append(
                f"ZONE I={Nx}, J={Ny}, K=1, F=POINT, "
                f'DATAPACKING=POINT, SOLUTIONTIME={time:.6f}')
        else:
            lines.append(
                f"ZONE I={Nx}, J={Ny}, K={Nz}, F=POINT, "
                f'DATAPACKING=POINT, SOLUTIONTIME={time:.6f}')

        # Flatten in Fortran order to match I,J,K ordering expected by Tecplot.
        X = mesh.Xc.ravel(order="F")
        Y = mesh.Yc.ravel(order="F")
        Z = mesh.Zc.ravel(order="F")
        U = u.ravel(order="F")
        V = v.ravel(order="F")
        W = w.ravel(order="F")
        P = p.ravel(order="F")
        TT = T.ravel(order="F")
        A = alpha.ravel(order="F")

        rows = np.stack([X, Y, Z, U, V, W, P, TT, A], axis=1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
            for r in rows:
                fh.write(" ".join(f"{val:.6e}" for val in r) + "\n")
        return path

    # ------------------------------------------------------------------ #
    def write_csv(self, time: float, u, v, w, p, T, alpha=None,
                  fname: str | None = None) -> str:
        """Export one snapshot as a flat CSV (for spreadsheets / quick plots)."""
        mesh = self.mesh
        if w is None:
            w = np.zeros_like(u)
        if alpha is None:
            alpha = np.zeros_like(u)
        if fname is None:
            fname = f"frame_{time:08.4f}.csv"
        path = os.path.join(self.output_dir, fname)
        X = mesh.Xc.ravel(order="F")
        Y = mesh.Yc.ravel(order="F")
        Z = mesh.Zc.ravel(order="F")
        rows = np.stack([X, Y, Z, u.ravel(order="F"), v.ravel(order="F"),
                         w.ravel(order="F"), p.ravel(order="F"),
                         T.ravel(order="F"), alpha.ravel(order="F")], axis=1)
        header = "x,y,z,u,v,w,p,T,alpha"
        np.savetxt(path, rows, delimiter=",", header=header, comments="")
        return path

    # ------------------------------------------------------------------ #
    def write_hdf5(self, time: float, fields: dict, fname: str | None = None
                   ) -> str:
        """Export one snapshot to an HDF5 file."""
        import h5py
        if fname is None:
            fname = f"frame_{time:08.4f}.h5"
        path = os.path.join(self.output_dir, fname)
        with h5py.File(path, "w") as fh:
            fh.attrs["time"] = time
            for key, val in fields.items():
                if val is not None and isinstance(val, np.ndarray):
                    fh.create_dataset(key, data=val)
            fh.create_dataset("X", data=self.mesh.Xc)
            fh.create_dataset("Y", data=self.mesh.Yc)
            fh.create_dataset("Z", data=self.mesh.Zc)
        return path