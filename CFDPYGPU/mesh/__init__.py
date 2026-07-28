"""Mesh subsystem.

Provides the structured Cartesian *staggered* (marker-and-cell) mesh on which
the whole finite-volume discretisation is built.  Cell-centred fields
(pressure, temperature, VOF) live at ``Mesh.cell``, face fields (velocity
components) at the faces normal to their axis.
"""

from .mesh import Mesh

__all__ = ["Mesh"]