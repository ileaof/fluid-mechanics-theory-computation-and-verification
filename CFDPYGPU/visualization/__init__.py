"""Visualisation and post-processing subsystem.

Two independent output back-ends:

* :class:`visualization.matplotlib_view.MatplotlibViewer` -- interactive /
  animated plots (pressure, temperature, velocity, vectors, streamlines, VOF);
* :class:`visualization.tecplot_writer.TecplotExporter` -- Tecplot 360 ASCII
  ``.dat`` export of the primary fields.

Both consume the same field snapshots produced by the :class:`Simulation`
loop, so adding a third back-end (VTK/PyVista, Paraview) only requires writing
a new consumer of the snapshot dictionaries.
"""

from .matplotlib_view import MatplotlibViewer
from .tecplot_writer import TecplotExporter
from .postprocessor import PostProcessor

__all__ = ["MatplotlibViewer", "TecplotExporter", "PostProcessor"]