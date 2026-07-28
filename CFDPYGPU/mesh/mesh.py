"""Structured Cartesian staggered (MAC) mesh.

Arrangement of the unknowns
----------------------------
The solver uses the classic *marker-and-cell* (Harlow & Welch, 1965) staggered
layout on a structured Cartesian grid:

* **Cell-centred** scalars (pressure ``p``, temperature ``T``, VOF ``alpha``)
  are stored at the geometric centre of each control volume.
* **Face-centred** velocities are stored at the faces normal to their axis:

  - ``u`` lives on ``x``-faces   -> shape ``(Nx+1, Ny, Nz)``
  - ``v`` lives on ``y``-faces   -> shape ``(Nx, Ny+1, Nz)``
  - ``w`` lives on ``z``-faces   -> shape ``(Nx, Ny, Nz+1)``

This staggering places the pressure-velocity coupling on a *natural* staggered
stencil: the discrete divergence of a cell is the sum of the four (six in 3D)
face fluxes, and the pressure gradient at a face is a simple centred
difference between the two neighbouring cell pressures.  It decouples the
pressure between adjacent cells (no checkerboard instability) and keeps the
convective fluxes naturally conservative.

A 2D problem is represented as a 3D mesh with ``Nz == 1``; all operators are
written once in 3D and degrade transparently to 2D because the ``z``-extent
is unity and the ``w``-fluxes vanish.
"""

from __future__ import annotations

import numpy as np


class Mesh:
    """Structured Cartesian staggered mesh.

    Parameters
    ----------
    Nx, Ny, Nz:
        Number of cells along each axis.  ``Nz == 1`` selects 2D.
    Lx, Ly, Lz:
        Physical domain lengths.

    The grid is uniform along each axis.  ``dx``, ``dy``, ``dz`` are the cell
    sizes and the boolean :attr:`is_2d` records the dimensionality.
    """

    def __init__(self, Nx: int, Ny: int, Nz: int,
                 Lx: float, Ly: float, Lz: float) -> None:
        if min(Nx, Ny, Nz) < 1:
            raise ValueError("Cell counts must be >= 1.")
        if min(Lx, Ly, Lz) <= 0.0:
            raise ValueError("Domain lengths must be positive.")

        self.Nx: int = int(Nx)
        self.Ny: int = int(Ny)
        self.Nz: int = int(Nz)
        self.Lx: float = float(Lx)
        self.Ly: float = float(Ly)
        self.Lz: float = float(Lz)

        # Uniform spacing.
        self.dx: float = self.Lx / self.Nx
        self.dy: float = self.Ly / self.Ny
        self.dz: float = self.Lz / max(self.Nz, 1)

        self.is_2d: bool = self.Nz <= 1
        self.dim: int = 2 if self.is_2d else 3

        # --- Coordinate arrays -------------------------------------------------
        # Cell centres (shape (Nx, Ny, Nz)).
        self.xc = (np.arange(Nx) + 0.5) * self.dx
        self.yc = (np.arange(Ny) + 0.5) * self.dy
        self.zc = (np.arange(Nz) + 0.5) * self.dz
        self.Xc, self.Yc, self.Zc = np.meshgrid(self.xc, self.yc, self.zc,
                                                indexing="ij")

        # Face centres.
        self.xf = np.arange(Nx + 1) * self.dx       # u-faces
        self.yf = np.arange(Ny + 1) * self.dy       # v-faces
        self.zf = np.arange(Nz + 1) * self.dz       # w-faces

        # Geometric factors ----------------------------------------------------
        self.cell_volume: float = self.dx * self.dy * (self.dz if not self.is_2d else 1.0)
        # Face areas per axis (used for flux bookkeeping).
        self.area_x: float = self.dy * (self.dz if not self.is_2d else 1.0)
        self.area_y: float = self.dx * (self.dz if not self.is_2d else 1.0)
        self.area_z: float = self.dx * self.dy

    # ------------------------------------------------------------------ #
    # Shape helpers
    # ------------------------------------------------------------------ #
    @property
    def cell_shape(self) -> tuple[int, int, int]:
        """Shape of a cell-centred field ``(Nx, Ny, Nz)``."""
        return (self.Nx, self.Ny, self.Nz)

    def face_shape(self, axis: int) -> tuple[int, int, int]:
        """Shape of a face field normal to ``axis`` (0=x,1=y,2=z)."""
        if axis == 0:
            return (self.Nx + 1, self.Ny, self.Nz)
        if axis == 1:
            return (self.Nx, self.Ny + 1, self.Nz)
        return (self.Nx, self.Ny, self.Nz + 1)

    # ------------------------------------------------------------------ #
    # Field factories
    # ------------------------------------------------------------------ #
    def zeros_cell(self, dtype=np.float64) -> np.ndarray:
        """Allocate a cell-centred field of zeros."""
        return np.zeros(self.cell_shape, dtype=dtype)

    def zeros_face(self, axis: int, dtype=np.float64) -> np.ndarray:
        """Allocate a face-centred field along *axis* of zeros."""
        return np.zeros(self.face_shape(axis), dtype=dtype)

    def cell_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the ``(X, Y, Z)`` coordinate grids of cell centres."""
        return self.Xc, self.Yc, self.Zc

    # ------------------------------------------------------------------ #
    # Distances / metrics
    # ------------------------------------------------------------------ #
    def spacing(self, axis: int) -> float:
        return (self.dx, self.dy, self.dz)[axis]

    def __repr__(self) -> str:
        kind = "2D" if self.is_2d else "3D"
        return (f"Mesh({kind}, Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
                f"L=({self.Lx},{self.Ly},{self.Lz}), "
                f"dx={self.dx:.3e}, dy={self.dy:.3e}, dz={self.dz:.3e})")