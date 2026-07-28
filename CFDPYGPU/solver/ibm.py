"""Mirror-point ghost-cell immersed-boundary (IB) forcing.

This module replaces the *staircase* direct-forcing (clamp solid cells to zero,
which puts the no-slip at the jagged cell face and pins flow separation at the
staircase corners) with a **curved-boundary** treatment that places the no-slip
at the true immersed surface.  It is the standard ghost-cell / mirror-point IB
method (Mittal et al. 2008; Uhlmann 2005) specialised to the cell-centred
collocated mesh used by the rest of the framework.

Ghost cells
-----------
A cell is a **ghost cell** when it is solid *and* at least one of its face
neighbours is fluid -- the first solid layer just inside the body.  For each
ghost cell with centre ``C``:

1. Find the closest point ``B`` on the true boundary (for a z-axis cylinder of
   centre ``c`` and radius ``r`` this is ``B = c + r (C - c)/|C - c|``).
2. The **image point** ``I = 2 B - C`` is the reflection of the ghost centre
   across ``B``; it lies in the fluid at the same distance from the wall as the
   ghost, on the other side.
3. Bilinearly interpolate the field at ``I`` from the four surrounding fluid
   cell centres -> ``u_I``.
4. Enforce the wall value by linear interpolation between ``C`` (ghost, value
   ``u_G``) and ``I`` (image, value ``u_I``) at the midpoint ``B``::

       u(B) = (u_G + u_I)/2 = 0   ->   u_G = -u_I.

   This pins the no-slip at the true boundary point ``B`` (the midpoint of
   ``C`` and ``I`` by construction), *not* at the staircase cell face, so the
   near-wall profile and the separation point are no longer tied to the grid.

Deep interior solid cells (no fluid neighbour) are held at zero.  The
no-penetration (zero face flux at the solid/fluid interface) continues to be
enforced by :meth:`BoundaryCondition.mask_solid_faces`, which is left untouched.

The forcing is a direct (post-solve) override applied to the predicted and to
the corrected velocity each time step (the same three sites that previously
clamped ``solid -> 0``).  It is first-order in the boundary treatment but
places the wall at the correct location to ``O(h^2)`` under the bilinear image
interpolation, which is what lets the wake separate correctly.

Only 2-D z-axis cylinders are supported here (the cylinder benchmark).  The
module returns a no-op ``IBMForcing`` when no curved body is configured, so the
staircase code path is the default and IBM is opt-in via
``"immersed_method": "ibm"``.
"""

from __future__ import annotations

import numpy as np


class IBMForcing:
    """Mirror-point ghost-cell IB forcing for one or more curved bodies."""

    def __init__(self, mesh, solid: np.ndarray, bodies: list[dict]) -> None:
        self.mesh = mesh
        self.solid = np.asarray(solid, dtype=bool)
        self.bodies = bodies
        # Precomputed per-ghost-cell data (filled by _precompute; empty if no
        # curved body / no ghost cells).
        self._ghost_flat: np.ndarray = np.empty(0, dtype=np.int64)
        self._deep_flat: np.ndarray = np.empty(0, dtype=np.int64)
        # bilinear stencil: four flat indices and four weights per ghost cell
        self._stencil: np.ndarray = np.empty((0, 4), dtype=np.int64)
        self._weights: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self._has_ib = False
        self._precompute()

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self._has_ib

    # ------------------------------------------------------------------ #
    def _closest_point(self, x: np.ndarray, y: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For points ``(x, y)`` near each configured z-axis cylinder, return
        ``(Bx, By, nx, ny)`` -- the closest boundary point and the outward unit
        normal (body -> fluid)."""
        Bx = x.copy()
        By = y.copy()
        nx = np.ones_like(x)
        ny = np.zeros_like(x)
        for ob in self.bodies:
            if ob.get("shape") != "cylinder":
                continue
            cx, cy = ob["center"]
            r = ob["radius"]
            if ob.get("axis", "z") != "z":
                continue
            dx = x - cx
            dy = y - cy
            d = np.sqrt(dx * dx + dy * dy)
            # only finite, non-degenerate points map to this cylinder
            ok = (d > 1e-12) & np.isfinite(d)
            inv = np.where(ok, 1.0 / d, 0.0)
            nxi = dx * inv
            nyi = dy * inv
            Bx = np.where(ok, cx + r * nxi, Bx)
            By = np.where(ok, cy + r * nyi, By)
            nx = np.where(ok, nxi, nx)
            ny = np.where(ok, nyi, ny)
        return Bx, By, nx, ny

    # ------------------------------------------------------------------ #
    def _precompute(self) -> None:
        mesh = self.mesh
        if mesh.is_2d:
            s2 = self.solid[:, :, 0]
            Nx, Ny = mesh.Nx, mesh.Ny
            shape2 = (Nx, Ny)
            dx, dy = mesh.dx, mesh.dy
            xc = mesh.xc
            yc = mesh.yc
        else:
            # 3-D not supported yet -- fall back to staircase (no ghost forcing).
            self._has_ib = False
            return

        curved = any(ob.get("shape") in ("cylinder", "sphere")
                     for ob in self.bodies)
        if not curved:
            self._has_ib = False
            return

        # ghost = solid cell with at least one fluid 4-neighbour
        fluid = ~s2
        neighbour_fluid = np.zeros_like(s2)
        neighbour_fluid[1:, :] |= fluid[:-1, :]
        neighbour_fluid[:-1, :] |= fluid[1:, :]
        neighbour_fluid[:, 1:] |= fluid[:, :-1]
        neighbour_fluid[:, :-1] |= fluid[:, 1:]
        ghost = s2 & neighbour_fluid
        deep = s2 & ~ghost

        gi, gj = np.nonzero(ghost)
        if gi.size == 0:
            self._has_ib = False
            return

        Cx = xc[gi]
        Cy = yc[gj]
        Bx, By, _, _ = self._closest_point(Cx, Cy)
        # image point I = 2B - C
        Ix = 2.0 * Bx - Cx
        Iy = 2.0 * By - Cy

        # bilinear stencil around I from the four surrounding cell centres
        i0 = np.floor(Ix / dx - 0.5).astype(np.int64)
        j0 = np.floor(Iy / dy - 0.5).astype(np.int64)
        i0 = np.clip(i0, 0, Nx - 2)
        j0 = np.clip(j0, 0, Ny - 2)
        tx = (Ix - (i0 + 0.5) * dx) / dx
        ty = (Iy - (j0 + 0.5) * dy) / dy
        # guard against clipping drift pushing the parameter outside [0,1]
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)

        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        k00 = np.ravel_multi_index((i0, j0), shape2)
        k10 = np.ravel_multi_index((i0 + 1, j0), shape2)
        k01 = np.ravel_multi_index((i0, j0 + 1), shape2)
        k11 = np.ravel_multi_index((i0 + 1, j0 + 1), shape2)

        self._ghost_flat = np.ravel_multi_index((gi, gj, np.zeros_like(gi)),
                                                mesh.cell_shape)
        self._deep_flat = np.ravel_multi_index(
            (np.nonzero(deep)[0], np.nonzero(deep)[1],
             np.zeros(np.count_nonzero(deep), dtype=np.int64)),
            mesh.cell_shape) if deep.any() else np.empty(0, dtype=np.int64)
        # flatten the 2-D stencil indices into the (Nx, Ny, 1) cell layout
        def flat3(i2, j2):
            return np.ravel_multi_index((i2, j2, np.zeros_like(i2)),
                                        mesh.cell_shape)
        self._stencil = np.stack([flat3(i0, j0), flat3(i0 + 1, j0),
                                  flat3(i0, j0 + 1), flat3(i0 + 1, j0 + 1)],
                                 axis=1)
        self._weights = np.stack([w00, w10, w01, w11], axis=1)
        self._has_ib = True

    # ------------------------------------------------------------------ #
    def apply(self, field: np.ndarray) -> None:
        """Override the velocity on the IB ghost cells (and zero deep solid).

        Reads the current field to interpolate the image-point value, then
        writes the ghost and deep cells in place.  No-op when IBM is inactive
        (the caller falls back to the staircase clamp).
        """
        if not self._has_ib:
            return
        flat = field.ravel()
        s = self._stencil
        w = self._weights
        # image-point value from the *current* field (ghost cells keep their
        # value from the previous forcing pass -- the implicit coupling is
        # resolved over time steps)
        u_I = (w[:, 0] * flat[s[:, 0]] + w[:, 1] * flat[s[:, 1]]
               + w[:, 2] * flat[s[:, 2]] + w[:, 3] * flat[s[:, 3]])
        flat[self._ghost_flat] = -u_I
        if self._deep_flat.size:
            flat[self._deep_flat] = 0.0