"""Physical-model package.

Holds the thermophysical state and the body-force models:

* :class:`physics.fluid.Fluid`      -- single-phase properties and accessors;
* :class:`physics.material.Material`-- generic material (reused for solids /
  multi-species extension);
* :mod:`physics.gravity`           -- gravity body force vector;
* :mod:`physics.buoyancy`          -- Boussinesq buoyancy source term.
"""

from .fluid import Fluid
from .material import Material
from .gravity import Gravity
from .buoyancy import BoussinesqBuoyancy

__all__ = ["Fluid", "Material", "Gravity", "BoussinesqBuoyancy"]