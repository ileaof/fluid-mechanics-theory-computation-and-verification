"""GPU geometric multigrid for the pressure-Poisson solve (Numba CUDA).

Why
---
The profile (``GPU_PERFORMANCE_REPORT.md``) settled the VOF bottleneck: the
Jacobi-preconditioned BiCGSTAB spends ~1000-2000 iterations/step, and each
iteration is dominated by host-bound dot products, not by the matvec.  A
geometric multigrid removes the *iteration count* problem at its root: the
long-wavelength error a Krylov method crawls through is exactly what a
coarse-grid correction annihilates in O(1) V-cycles, and the smoother does only
local (latency-free) stencil work that saturates the GPU.

Operator
--------
Same operator as the production :class:`solver.pressure.PressureSolver._matrix`
-- the 7-point variable-coefficient Laplacian on a uniform collocated mesh,

    (A p)_c = sum_f c_f (p_nb - p_c),   c_f = (1/rho)_face / h_f^2,

with zero coefficients on domain-boundary faces (pure Neumann).  The
one-dimensional null space (the constant field) is handled analytically, as in
production: the RHS is mean-projected before the solve and the solution is
mean-subtracted afterwards.

Algorithm
---------
Standard non-Galerkin (rediscretised) V-cycle:

* **Smoother** -- red-black Gauss-Seidel, one 3-D kernel launch per colour per
  sweep.  Variable-coefficient capable and race-free (each colour only *reads*
  the other colour while updating its own).
* **Restriction** -- full weighting (separable 1/4-1/2-1/4 per axis,
  renormalised at fine boundaries so odd grid sizes and degenerate axes are
  handled uniformly).
* **Prolongation** -- trilinear interpolation, accumulated into the fine
  correction.
* **Coarse coefficients** -- rediscretisation along the normal direction: the
  coarse face between coarse cells ``I`` and ``I+1`` crosses fine faces
  ``2I+1`` and ``2I+2`` in series, so the conductance combines *harmonically*
  (``c_c = c1 c2 / (2 (c1 + c2))``, the ``1/2`` folding in the ``h^2/h_c^2``
  rescaling) and ``c1/4`` when only one fine face exists (odd coarse size,
  which is why coarse axes are the *ceiling* of half).  The harmonic mean is
  essential: with an arithmetic average the coarse operator overestimates the
  coupling across a sharp density jump, the correction overshoots and the
  V-cycle diverges (measured on rho-ratio-833 two-phase cases).
* **Coarsest solve** -- direct: the tiny pinned CSR matrix is assembled
  vectorised on the host and factorised once per operator refresh
  (``scipy.sparse.linalg.splu``, cached); each V-cycle then costs one
  triangular solve on a few thousand cells -- negligible next to the
  fine-grid work.
* **Hierarchy depth** -- the default ``coarse_max_cells=4096`` caps the
  hierarchy at two coarsenings.  Deeper ladders *stagnate* (numpy-verified,
  see the class docstring), so do not lower the threshold hoping for more
  speed from a deeper hierarchy.
* **Convergence** -- after each V-cycle the residual is recomputed and its
  norm compared against ``tol * ||b||`` (device reduction, one host sync per
  cycle -- ~10 syncs per solve instead of ~500 for Krylov).

Limitations / fallbacks (all decided by the caller)
---------------------------------------------------
* Pure-Neumann only, no Dirichlet outlet rows, no cut-cell apertures -- the
  production caller falls back to the SciPy path in those cases.
* Density contrast: with rediscretised (non-Galerkin) coarse operators this
  V-cycle converges for mild contrasts (measured: rho-ratio 1.2 needs 31
  cycles on a 200x160 grid) but *diverges* for strong ones (ratio >= 10 on
  the same grid).  Every stencil-compatible coarse operator variant tried --
  arithmetic, Galerkin-fitted 7-point, aggregation Galerkin -- diverges worse;
  only a full-Galerkin P^T A P converges, but its coarse matrices have ~63
  nnz/row (long-range interface couplings), which geometric rediscretisation
  cannot express.  Strong-contrast problems (e.g. the water/air VOF cases,
  rho-ratio 833) must use the Krylov fallback.
* Coefficients arrive as *host* arrays each step (the VOF fields are still
  CPU-resident); the upload is O(N) PCIe traffic per step and is the next
  increment's target (GPU-resident VOF).
* A CPU-only machine never imports this module (it imports :mod:`numba`); the
  production path guards on ``gpu.backend.is_gpu_active()``.
"""

from __future__ import annotations

import numpy as np
from numba import cuda

from .backend import get_backend
from . import kernels as K

# Threads per 3-D block for the stencil kernels: 256 threads (8x8x4) keeps the
# shared/register footprint light and gives full warps.
_TB = (8, 8, 4)


# =========================================================================== #
# Device kernels
# =========================================================================== #
@cuda.jit
def _rbgs_kernel(cx, cy, cz, b, p, nx, ny, nz, parity):
    """One red-black Gauss-Seidel sweep (in place on ``p``).

    ``cx[i, j, k]`` is the coefficient of the face between cells ``(i, j, k)``
    and ``(i+1, j, k)`` (shapes ``(nx-1, ny, nz)`` / ``(nx, ny-1, nz)`` /
    ``(nx, ny, nz-1)``); boundary faces have coefficient zero (pure Neumann).
    Cells of colour ``(i+j+k) % 2 == parity`` are updated,
    ``p = (sum_f c_f p_nb - b_c) / sum_f c_f``; the other colour is untouched
    (so the two half-sweeps are race-free without atomics).
    """
    i, j, k = cuda.grid(3)
    if i >= nx or j >= ny or k >= nz:
        return
    if (i + j + k) % 2 != parity:
        return
    c_xm = cx[i - 1, j, k] if i > 0 else 0.0
    c_xp = cx[i, j, k] if i < nx - 1 else 0.0
    c_ym = cy[i, j - 1, k] if j > 0 else 0.0
    c_yp = cy[i, j, k] if j < ny - 1 else 0.0
    c_zm = cz[i, j, k - 1] if (k > 0 and cz.shape[2] > 1) else 0.0
    c_zp = cz[i, j, k] if (k < nz - 1 and cz.shape[2] > 1) else 0.0
    d = c_xm + c_xp + c_ym + c_yp + c_zm + c_zp
    if d == 0.0:
        return                      # isolated cell: keep the old value
    nb = 0.0
    if c_xm != 0.0:
        nb += c_xm * p[i - 1, j, k]
    if c_xp != 0.0:
        nb += c_xp * p[i + 1, j, k]
    if c_ym != 0.0:
        nb += c_ym * p[i, j - 1, k]
    if c_yp != 0.0:
        nb += c_yp * p[i, j + 1, k]
    if c_zm != 0.0:
        nb += c_zm * p[i, j, k - 1]
    if c_zp != 0.0:
        nb += c_zp * p[i, j, k + 1]
    p[i, j, k] = (nb - b[i, j, k]) / d


@cuda.jit
def _residual_kernel(cx, cy, cz, p, b, r, nx, ny, nz):
    """``r = b - A p`` for the 7-point variable-coefficient operator."""
    i, j, k = cuda.grid(3)
    if i >= nx or j >= ny or k >= nz:
        return
    c_xm = cx[i - 1, j, k] if i > 0 else 0.0
    c_xp = cx[i, j, k] if i < nx - 1 else 0.0
    c_ym = cy[i, j - 1, k] if j > 0 else 0.0
    c_yp = cy[i, j, k] if j < ny - 1 else 0.0
    c_zm = cz[i, j, k - 1] if (k > 0 and cz.shape[2] > 1) else 0.0
    c_zp = cz[i, j, k] if (k < nz - 1 and cz.shape[2] > 1) else 0.0
    d = c_xm + c_xp + c_ym + c_yp + c_zm + c_zp
    ap = 0.0
    if c_xm != 0.0:
        ap += c_xm * p[i - 1, j, k]
    if c_xp != 0.0:
        ap += c_xp * p[i + 1, j, k]
    if c_ym != 0.0:
        ap += c_ym * p[i, j - 1, k]
    if c_yp != 0.0:
        ap += c_yp * p[i, j + 1, k]
    if c_zm != 0.0:
        ap += c_zm * p[i, j, k - 1]
    if c_zp != 0.0:
        ap += c_zp * p[i, j, k + 1]
    r[i, j, k] = b[i, j, k] - (ap - d * p[i, j, k])


@cuda.jit
def _restrict_kernel(rf, rc, nx, ny, nz, mx, my, mz):
    """Full-weighting restriction ``rc = R rf`` (fine -> coarse).

    ``R`` is the row-normalised adjoint of the trilinear prolongation
    (``R = D^-1 P^T``), i.e. the classical cell-centred full weighting: per
    axis the weights over fine cells ``{2I-2 .. 2I+1}`` are
    ``(1/4, 3/4, 3/4, 1/4)``, renormalised wherever weights drop outside the
    fine grid.  The row normalisation (dividing by the accumulated weight sum)
    is essential: the plain adjoint ``P^T`` carries the prolongation's column
    sums (~2 per axis) into the coarse RHS, and combined with a rediscretised
    coarse operator the coarse correction then overshoots by that factor
    (observed as divergence).  Row-normalised, ``R 1 = 1`` and the pair
    (restriction, prolongation) is the standard conservative cell-centred one.
    """
    i, j, k = cuda.grid(3)
    if i >= mx or j >= my or k >= mz:
        return
    acc = 0.0
    wsum = 0.0
    for dx in range(4):
        fi = 2 * i + dx - 2            # fine cells 2I-2 .. 2I+1
        if fi < 0 or fi >= nx:
            continue
        bi = fi // 2                   # = i - 1 for the first pair, i for the second
        di = 1 if bi != i else 0       # which coarse neighbour fi interpolates to
        wi0 = 0.75 if fi % 2 == 0 else 0.25
        wx = wi0 if di == 0 else 1.0 - wi0
        sx = 1.0 if bi + 1 < mx else wi0   # axis factor of the fine cell's wsum
        wx = wx / sx
        for dj in range(4):
            fj = 2 * j + dj - 2
            if fj < 0 or fj >= ny:
                continue
            bj = fj // 2
            dj2 = 1 if bj != j else 0
            wj0 = 0.75 if fj % 2 == 0 else 0.25
            wy = wj0 if dj2 == 0 else 1.0 - wj0
            sy = 1.0 if bj + 1 < my else wj0
            wy = wy / sy
            for dk in range(4):
                fk = 2 * k + dk - 2
                if fk < 0 or fk >= nz:
                    continue
                bk = fk // 2
                dk2 = 1 if bk != k else 0
                wk0 = 0.75 if fk % 2 == 0 else 0.25
                wz = wk0 if dk2 == 0 else 1.0 - wk0
                sz = 1.0 if bk + 1 < mz else wk0
                wz = wz / sz
                wv = wx * wy * wz
                acc += wv * rf[fi, fj, fk]
                wsum += wv
    if wsum > 0.0:
        rc[i, j, k] = acc / wsum
    else:
        rc[i, j, k] = 0.0


@cuda.jit
def _prolong_add_kernel(pc, pf, mx, my, mz, nx, ny, nz):
    """Trilinear prolongation, *accumulated*: ``pf += P pc``.

    Fine cell ``i`` interpolates between coarse cells ``i//2`` and its
    neighbour with weights (3/4, 1/4) by parity of ``i`` (the trilinear
    weights of a 2:1 cell-centred grid); offsets outside the coarse grid are
    dropped and the remaining weights renormalised.
    """
    i, j, k = cuda.grid(3)
    if i >= nx or j >= ny or k >= nz:
        return
    acc = 0.0
    wsum = 0.0
    bi = i // 2
    wi0 = 0.75 if i % 2 == 0 else 0.25
    bj = j // 2
    wj0 = 0.75 if j % 2 == 0 else 0.25
    bk = k // 2
    wk0 = 0.75 if k % 2 == 0 else 0.25
    for di in range(2):
        ci = bi + di
        if ci >= mx:
            continue
        wi = wi0 if di == 0 else 1.0 - wi0
        for dj in range(2):
            cj = bj + dj
            if cj >= my:
                continue
            wj = wj0 if dj == 0 else 1.0 - wj0
            for dk in range(2):
                ck = bk + dk
                if ck >= mz:
                    continue
                wv = wi * wj * (wk0 if dk == 0 else 1.0 - wk0)
                acc += wv * pc[ci, cj, ck]
                wsum += wv
    if wsum > 0.0:
        pf[i, j, k] += acc / wsum


@cuda.jit
def _axpy_scalar_kernel(y, x, a, n):
    """``y += a * x`` with ``a`` a host scalar (flat 1-D views, in place)."""
    i = cuda.grid(1)
    if i < n:
        y[i] = y[i] + a * x[i]


def _grid(nx, ny, nz):
    return ((nx + _TB[0] - 1) // _TB[0],
            (ny + _TB[1] - 1) // _TB[1],
            (nz + _TB[2] - 1) // _TB[2])


def _launch1d(kernel, *args):
    """Launch a 1-D element-wise kernel over ``n`` elements (last arg)."""
    n = args[-1]
    kernel[(n + 255) // 256, 256](*args)


def _size(shape) -> int:
    return int(shape[0]) * int(shape[1]) * int(shape[2])


def _dummy_faces(shp, axis) -> np.ndarray:
    """Zero-coefficient face array for a degenerate axis (never coupled).

    The kernels only touch these arrays when the corresponding neighbour
    exists, so the values never matter -- but the arrays must be proper 3-D
    arrays (the kernels read ``cz.shape[2]``) with zero entries so an
    accidental read is harmless.
    """
    shp = list(shp)
    shp[axis] = 1
    return np.zeros(shp)


# =========================================================================== #
# Host-side solver
# =========================================================================== #
class GPUGeometricMultigrid:
    """Geometric multigrid V-cycle solver for the production Poisson operator.

    Parameters
    ----------
    cell_shape:
        ``(Nx, Ny, Nz)`` of the finest level (``Nz == 1`` selects 2D).
    tol:
        Relative residual tolerance ``||r|| / ||b||`` (the production
        ``poisson_tol``).
    max_cycles:
        Maximum V-cycles per solve.
    nu1, nu2:
        Pre-/post-smoothing red-black sweeps per level.  The default 4 is the
        measured time optimum on both production ladders (3-D 64^3: 68 ms
        instead of 147 ms at 2; 2-D 200x160: ~55 ms vs 60 ms) -- the shallower
        (3-level) hierarchy leaves mid-frequency error that extra smoothing
        removes far cheaper than extra V-cycles.
    coarse_max_cells:
        Coarsen until the current level has at most this many cells (the
        coarsest level is then solved directly on the host).  4096 caps the
        hierarchy at two coarsenings: with three or more levels the V-cycle
        *stagnates* -- measured on every production ladder, even single-phase
        (numpy replica of the 64^3 4-level ladder: per-cycle rate decays from
        0.45 to >1, identical to the device; the 3-level hierarchy converges
        at ~0.3-0.5/cycle on the same cases).  This is an algorithmic property
        of deep rediscretised hierarchies with this smoothing/transfer triple,
        not a device bug -- the device reproduces one numpy V-cycle to 3e-16.

    Lifecycle
    ---------
    ``set_face_coefficients(cx, cy, cz)`` refreshes the operator (VOF: every
    step) and (re)builds all coarse levels; ``solve(b)`` then runs V-cycles to
    ``tol``.  Both accept plain NumPy arrays and return one; transfers are
    explicit and per-solve (fields are not yet GPU-resident).
    """

    def __init__(self, cell_shape, tol: float = 1e-6,
                 max_cycles: int = 100, nu1: int = 4, nu2: int = 4,
                 coarse_max_cells: int = 4096) -> None:
        self.shape = tuple(int(s) for s in cell_shape)
        self.tol = float(tol)
        self.max_cycles = int(max_cycles)
        self.nu1 = int(nu1)
        self.nu2 = int(nu2)
        self.coarse_max_cells = int(coarse_max_cells)
        self.backend = get_backend()
        if not self.backend.enabled:
            raise RuntimeError("GPUGeometricMultigrid requires the CUDA backend")
        self._n0 = _size(self.shape)
        # Per-level state, (re)built by set_face_coefficients: each level dict
        # holds device face-coefficient arrays ('cx'/'cy'/'cz') and flat
        # scratch buffers 'p'/'r'/'b'.
        self._levels: list[dict] = []
        self._ones = None           # device ones vector (mean projection)
        # Last-solve diagnostics (for the performance report / history).
        self.last_cycles = 0
        self.last_residual = float("inf")
        self.last_converged = False

    # ------------------------------------------------------------------ #
    # Level construction
    # ------------------------------------------------------------------ #
    def set_face_coefficients(self, cx, cy, cz) -> None:
        """Refresh the operator from *interior* face-coefficient arrays.

        ``cx`` has shape ``(Nx-1, Ny, Nz)`` and holds ``c`` for the face
        between cells ``(i, j, k)`` and ``(i+1, j, k)``; ``cy`` / ``cz``
        analogously.  Values must already include the ``1/h^2`` scaling
        (production's ``(1/rho)_face / h^2``).  ``cz`` may be ``None`` for a
        2D grid; degenerate axes (size 1) may pass any 1-element dummy array
        -- it is replaced by a harmless zero array of the right shape.
        """
        bck = self.backend
        nx, ny, nz = self.shape

        def prep(c, axis):
            if c is None or self.shape[axis] <= 1:
                return _dummy_faces(self.shape, axis)
            c = np.ascontiguousarray(c, dtype=np.float64)
            if c.ndim != 3:                     # squeeze/expand safety
                c = c.reshape(self.shape[:axis]
                              + (self.shape[axis] - 1,)
                              + self.shape[axis + 1:])
            return c

        levels = [{"shape": self.shape,
                   "cx": prep(cx, 0), "cy": prep(cy, 1), "cz": prep(cz, 2)}]
        # Coarsen (host numpy, cheap O(N)) until the coarsest criterion met.
        # The coarse size is the *ceiling* of half: with a floor coarsening of
        # an odd fine axis the last fine cell has no coarse parent -- its
        # error is invisible to the coarse correction and the V-cycle
        # stagnates (measured ~0.6/cycle instead of ~0.24 on a (25,20) pair).
        while True:
            shp = levels[-1]["shape"]
            if _size(shp) <= self.coarse_max_cells:
                break
            nxt = (max((shp[0] + 1) // 2, 1), max((shp[1] + 1) // 2, 1),
                   max((shp[2] + 1) // 2, 1))
            if nxt == shp:
                break
            cur = levels[-1]
            levels.append({"shape": nxt,
                           "cx": self._coarsen_axis(cur["cx"], 0, shp, nxt),
                           "cy": self._coarsen_axis(cur["cy"], 1, shp, nxt),
                           "cz": self._coarsen_axis(cur["cz"], 2, shp, nxt)})
        # Device arrays + flat buffers for every level.
        self._levels = []
        for lv in levels:
            d = {"shape": lv["shape"]}
            for key in ("cx", "cy", "cz"):
                d[key] = bck.asarray(np.ascontiguousarray(lv[key]))
            n = _size(lv["shape"])
            d["p"] = bck.zeros(n)
            d["r"] = bck.zeros(n)
            d["b"] = bck.zeros(n)
            self._levels.append(d)
        # The coarsest matrix + LU factorisation only depend on the face
        # coefficients, so they are (re)built here -- once per operator
        # refresh, not once per V-cycle.
        self._coarsest_setup(self._levels[-1])
        if self._ones is None or self._ones.size != self._n0:
            self._ones = bck.zeros(self._n0)
            K.fill(self._ones, 1.0, self._n0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coarsen_axis(c: np.ndarray, axis: int, shp, cshp) -> np.ndarray:
        """Rediscretise one axis' face coefficients onto the coarse level.

        ``c`` is the fine interior-face array with the face dimension at
        ``axis`` (length ``shp[axis] - 1``).  A coarse face spans two fine
        faces in series, so the conductance combines *harmonically*
        (``c_c = c1 c2 / (2 (c1 + c2))``): across a sharp density jump the
        arithmetic mean would massively overestimate the coarse coupling and
        the coarse correction then overshoots (observed as divergence on
        two-phase problems).  Returns the coarse interior-face array, or a
        zero dummy when the coarse axis is degenerate.
        """
        cn = cshp[axis]
        if cn <= 1:
            return _dummy_faces(cshp, axis)
        # Move the face axis to the front; coarsen it, then trim the trailing
        # (transverse) dims to the coarse sizes with a stride-2 slice.
        lead = np.moveaxis(np.asarray(c), axis, 0)
        m = cn - 1
        c1 = lead[1:2 * m + 1:2]        # fine faces 2I+1,  I = 0..m-1
        c2 = lead[2:2 * m + 1:2]        # fine faces 2I+2 (may be one short)
        out = np.empty_like(c1)
        k2 = c2.shape[0]
        s = c1[:k2] + c2
        out[:k2] = np.where(s > 0.0, c1[:k2] * c2 / (2.0 * s), 0.0)
        out[k2:] = c1[k2:] / 4.0
        # trim transverse dims to the coarse sizes (representative even index)
        sl = [slice(None)]
        for a in range(3):
            if a == axis:
                continue
            sl.append(slice(0, 2 * cshp[a], 2) if shp[a] != cshp[a]
                      else slice(None))
        # put the face axis back in its natural position (x, y, z order)
        return np.ascontiguousarray(np.moveaxis(out[tuple(sl)], 0, axis))

    # ------------------------------------------------------------------ #
    # V-cycle machinery
    # ------------------------------------------------------------------ #
    def _v_cycle(self, l: int) -> None:
        lvl = self._levels[l]
        if l == len(self._levels) - 1:
            self._coarsest_solve(lvl)
            return
        nxt = self._levels[l + 1]
        nx, ny, nz = lvl["shape"]
        mx, my, mz = nxt["shape"]
        p3 = lvl["p"].reshape(lvl["shape"])
        r3 = lvl["r"].reshape(lvl["shape"])
        b3 = lvl["b"].reshape(lvl["shape"])
        # pre-smooth (red then black)
        for _ in range(self.nu1):
            _rbgs_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                            b3, p3, nx, ny, nz, 0)
            _rbgs_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                            b3, p3, nx, ny, nz, 1)
        # r = b - A p ; restrict: b_{l+1} = R r ; p_{l+1} = 0
        _residual_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                            p3, b3, r3, nx, ny, nz)
        _restrict_kernel[_grid(mx, my, mz), _TB](r3, nxt["b"].reshape(nxt["shape"]),
                                            nx, ny, nz, mx, my, mz)
        K.fill(nxt["p"], 0.0, _size(nxt["shape"]))
        self._v_cycle(l + 1)
        # prolongate + add: p_l += P p_{l+1} ; post-smooth
        _prolong_add_kernel[_grid(nx, ny, nz), _TB](nxt["p"].reshape(nxt["shape"]),
                                               p3, mx, my, mz, nx, ny, nz)
        for _ in range(self.nu2):
            _rbgs_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                            b3, p3, nx, ny, nz, 0)
            _rbgs_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                            b3, p3, nx, ny, nz, 1)

    # ------------------------------------------------------------------ #
    def _coarsest_setup(self, lvl) -> None:
        """Assemble the tiny coarsest system and cache its LU factorisation.

        The matrix is the coarsest level's 7-point operator assembled
        *vectorised* (flat index arithmetic, no Python cell loop) with one row
        pinned to identity (removing the null space); the resulting constant is
        irrelevant because the fine solution is mean-subtracted.  Called from
        :meth:`set_face_coefficients` -- once per operator refresh, not once
        per V-cycle (the per-cycle ``spsolve`` this replaces dominated the
        3-D cycle time).
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        mx, my, mz = lvl["shape"]
        n = mx * my * mz
        cx = self.backend.to_host(lvl["cx"]).reshape(-1)
        cy = self.backend.to_host(lvl["cy"]).reshape(-1)
        cz = self.backend.to_host(lvl["cz"]).reshape(-1)

        C = np.arange(n).reshape(mx, my, mz)
        entries = []
        if mx > 1:
            entries.append((C[:-1, :, :].ravel(), C[1:, :, :].ravel(), cx))
        if my > 1:
            entries.append((C[:, :-1, :].ravel(), C[:, 1:, :].ravel(), cy))
        if mz > 1:
            entries.append((C[:, :, :-1].ravel(), C[:, :, 1:].ravel(), cz))
        R = np.concatenate([e[0] for e in entries] + [e[1] for e in entries]) \
            if entries else np.zeros(0, dtype=np.int64)
        Cc = np.concatenate([e[1] for e in entries] + [e[0] for e in entries]) \
            if entries else np.zeros(0, dtype=np.int64)
        Vv = np.concatenate([e[2] for e in entries] + [e[2] for e in entries]) \
            if entries else np.zeros(0)
        # diagonal: minus the sum of each cell's couplings
        d = np.zeros(n)
        np.add.at(d, R, Vv)
        R = np.concatenate([R, np.arange(n)])
        Cc = np.concatenate([Cc, np.arange(n)])
        Vv = np.concatenate([Vv, -d])
        A = sp.csr_matrix((Vv, (R, Cc)), shape=(n, n))
        # pin row 0 (constant null space); keep the matrix symmetric-looking
        # for splu by zeroing the row's off-diagonals explicitly
        A = A.tolil()
        A.rows[0] = [0]
        A.data[0] = [1.0]
        self._coarsest_lu = splu(A.tocsc())
        self._coarsest_n = n

    # ------------------------------------------------------------------ #
    def _coarsest_solve(self, lvl) -> None:
        """Direct solve of the tiny coarsest system using the cached LU."""
        mx, my, mz = lvl["shape"]
        b = self.backend.to_host(lvl["b"]).reshape(mx, my, mz)
        b = b - b.mean()
        x = self._coarsest_lu.solve(b.ravel())
        lvl["p"].copy_to_device(np.ascontiguousarray(x).ravel())

    # ------------------------------------------------------------------ #
    def solve(self, b) -> np.ndarray:
        """Solve ``A x = b`` (pure Neumann) from a zero initial guess.

        ``b`` is a host array shaped like the finest grid; the return value is
        the host solution, mean-subtracted (the constant null space removed).
        """
        lvl = self._levels[0]
        nx, ny, nz = self.shape
        b = np.ascontiguousarray(b, dtype=np.float64)
        lvl["b"].copy_to_device(b.ravel())
        # mean-project the RHS (constant null space); reductions run on the
        # flat views, the stencil kernels on the 3-D views of the same buffer.
        bf = lvl["b"]
        mean = K.dot(bf, self._ones, self._n0) / self._n0
        _launch1d(_axpy_scalar_kernel, bf, self._ones, -mean, self._n0)
        bnorm = K.norm2(bf, self._n0)
        if bnorm == 0.0:
            return np.zeros(self.shape)
        K.fill(lvl["p"], 0.0, self._n0)
        converged = False
        it = 0
        rnorm = float("inf")
        for it in range(1, self.max_cycles + 1):
            self._v_cycle(0)
            _residual_kernel[_grid(nx, ny, nz), _TB](lvl["cx"], lvl["cy"], lvl["cz"],
                                                lvl["p"].reshape(self.shape),
                                                lvl["b"].reshape(self.shape),
                                                lvl["r"].reshape(self.shape),
                                                nx, ny, nz)
            rnorm = K.norm2(lvl["r"], self._n0)
            if rnorm < self.tol * bnorm:
                converged = True
                break
        self.last_cycles = it
        self.last_residual = rnorm / bnorm
        self.last_converged = converged
        # mean-subtract the solution (null-space constant removed)
        pf = lvl["p"]
        mean = K.dot(pf, self._ones, self._n0) / self._n0
        _launch1d(_axpy_scalar_kernel, pf, self._ones, -mean, self._n0)
        self.backend.synchronize()
        return self.backend.to_host(pf).reshape(self.shape)