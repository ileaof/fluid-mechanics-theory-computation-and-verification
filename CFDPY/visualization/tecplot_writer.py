"""Tecplot 360 ASCII ``.dat`` exporter.

Writes a structured IJ-ordered (2D) or IJK-ordered (3D) point dataset in the
modern Tecplot 360 ASCII format -- the same dialect emitted by the
``py2tec``/``tec2py`` tools (https://github.com/luohancfd/py2tec), which is
what current Tecplot 360 reads reliably.  Each frame is exported with the
variables::

    X  Y  Z  U  V  W  Pressure  Temperature  Alpha

The file layout is::

    TITLE = "CFDPY snapshot t=..."
    VARIABLES = "X","Y","Z","U","V","W","Pressure","Temperature","Alpha"
    ZONE T="t=..." ZONETYPE=ORDERED I=Nx J=Ny [K=Nz]
         DATAPACKING=POINT STRANDID=1 SOLUTIONTIME=...
    <one line per node, all variables per line, I varies fastest>

The zone is a structured **ORDERED** zone with **POINT** data packing (one
record per node, all variables on the same line).  This replaces the legacy
``F=POINT`` token -- a deprecated finite-element specifier that conflicts with
``DATAPACKING`` and is rejected by current Tecplot 360.  Node values are
flattened in Fortran order so the I (x) index varies fastest, then J (y), then
K (z), which is the ordering Tecplot expects for an ORDERED zone.

The exporter is intentionally ASCII-only (no binary Tecplot variant) to match
the project specification and remain trivially diffable.  Files are written
with a single zone per file (one time step) so they can be appended into a
time-sequence by the :class:`Simulation` loop, and each file round-trips
through ``py2tec.tec2py``.
"""

from __future__ import annotations

import os

import numpy as np

# Single source of truth for SI unit labels (see units.py).  Import is defensive
# so the exporter still works if the module is ever vendored without units.py.
try:
    from units import label as _si_label, FIELD_UNITS as _FIELD_UNITS
except Exception:                       # pragma: no cover
    _FIELD_UNITS = {}

    def _si_label(name, pretty=None):   # type: ignore[misc]
        return pretty if pretty is not None else name


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

        # Variable names carry their coherent-SI unit in brackets (policy: every
        # exported dimensional quantity states its SI unit).  Tecplot 360 and the
        # py2tec round-trip both accept quoted names containing brackets.
        varnames = [_si_label(n) for n in ("X", "Y", "Z", "U", "V", "W",
                                           "Pressure", "Temperature", "Alpha")]
        zonename = f"t={time:.6f}"
        # Modern ORDERED zone header (py2tec dialect): ZONETYPE=ORDERED with
        # DATAPACKING=POINT -- no legacy F=POINT token.  STRANDID groups the
        # per-file time steps into one strand for Tecplot's time animation.
        dims = f"I={Nx} J={Ny}" if mesh.is_2d else f"I={Nx} J={Ny} K={Nz}"
        zone = (f'ZONE T="{zonename}" ZONETYPE=ORDERED {dims} '
                f'DATAPACKING=POINT STRANDID=1 SOLUTIONTIME={time:.6f}')

        # Flatten in Fortran order so I (x) varies fastest, then J (y), then
        # K (z) -- the ordering an ORDERED POINT zone expects.
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
            fh.write(f'TITLE = "CFDPY snapshot t={time:.6f}"\n')
            fh.write('VARIABLES = ' + ','.join(f'"{n}"' for n in varnames) + '\n')
            fh.write(zone + '\n')
            for r in rows:
                fh.write(" ".join(f"{val:.6e}" for val in r) + "\n")
            fh.write("\n")
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
        # Column headers annotated with coherent-SI units (alpha is the
        # dimensionless VOF fraction -> "[-]").
        header = ",".join(_si_label(c) for c in
                          ("x", "y", "z", "u", "v", "w", "p", "T", "alpha"))
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
            # Record the SI unit of the time attribute and every field dataset
            # as HDF5 metadata.  Dataset *names* are unchanged, so the restart
            # path (which reads u/v/w/p/T/alpha by name) is unaffected; the units
            # are attached non-intrusively as a `units` attribute per dataset.
            fh.attrs["time"] = time
            fh.attrs["time_units"] = "s"
            fh.attrs["unit_system"] = "SI (coherent)"
            for key, val in fields.items():
                if val is not None and isinstance(val, np.ndarray):
                    ds = fh.create_dataset(key, data=val)
                    ds.attrs["units"] = _FIELD_UNITS.get(key, "")
            for axis in ("X", "Y", "Z"):
                ds = fh.create_dataset(axis, data=getattr(self.mesh, f"{axis}c"))
                ds.attrs["units"] = "m"
        return path