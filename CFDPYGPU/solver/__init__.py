"""Solver package.

Finite-volume solvers built on the staggered mesh:

* :mod:`solver.boundary`     -- boundary-condition application;
* :mod:`solver.linear_solver`-- CG / BiCGSTAB / GMRES with ILU preconditioner;
* :mod:`solver.momentum`    -- Navier-Stokes momentum predictor;
* :mod:`solver.pressure`     -- pressure-Poisson solve;
* :mod:`solver.projection`  -- pressure-velocity coupling (projection method,
  architecturally ready for SIMPLE);
* :mod:`solver.energy`      -- energy (heat) transport;
* :mod:`solver.vof`         -- Volume-of-Fluid free-surface transport.
* :mod:`solver.forces`      -- drag / lift force integration on a body;
* :mod:`solver.ibm`         -- mirror-point ghost-cell immersed-boundary forcing;
* :mod:`solver.cut_cell`    -- cut-cell fluid-volume / face-aperture geometry
  for a curved immersed boundary (places the no-penetration wall at the true
  surface in the pressure-Poisson divergence).
"""

from .boundary import BoundaryCondition
from .linear_solver import LinearSolver
from .momentum import MomentumSolver
from .pressure import PressureSolver
from .projection import ProjectionMethod
from .energy import EnergySolver
from .vof import VOFSolver
from .forces import ForcesCalculator
from .ibm import IBMForcing
from .cut_cell import CutCellGeometry

__all__ = [
    "BoundaryCondition", "LinearSolver", "MomentumSolver", "PressureSolver",
    "ProjectionMethod", "EnergySolver", "VOFSolver", "ForcesCalculator",
    "IBMForcing", "CutCellGeometry",
]