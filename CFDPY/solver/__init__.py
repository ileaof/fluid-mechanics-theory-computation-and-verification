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
"""

from .boundary import BoundaryCondition
from .linear_solver import LinearSolver
from .momentum import MomentumSolver
from .pressure import PressureSolver
from .projection import ProjectionMethod
from .energy import EnergySolver
from .vof import VOFSolver

__all__ = [
    "BoundaryCondition", "LinearSolver", "MomentumSolver", "PressureSolver",
    "ProjectionMethod", "EnergySolver", "VOFSolver",
]