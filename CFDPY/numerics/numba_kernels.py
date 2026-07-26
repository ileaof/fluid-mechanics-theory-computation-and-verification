"""Numba-JIT kernels for the performance-critical inner loops.

The framework is written so that its hot numerics are pure, vectorised NumPy
and therefore already fast.  A few routines, however, are *element-wise with
branches* — the TVD flux limiters, which mix comparisons and arithmetic per
element.  Numba's ``@njit`` compiles those into tight scalar loops that avoid
NumPy's temporary-array allocation, which is measurable on the large arrays
touched every convection/VOF step.

The module degrades gracefully: if Numba is not installed (it is an *optional*
dependency), :func:`limiter_kernel` falls back to a pure-NumPy implementation
with identical numerics, so the framework runs with or without Numba.

Limiter identifier codes (used so the JIT kernel can dispatch without a
string argument, which Numba cannot take):

    0 = vanleer, 1 = minmod, 2 = superbee, 3 = beamwarming, 4 = osher
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    import numba as _numba_mod  # noqa: F401  (presence test)
    _HAS_NUMBA = True
except Exception:              # pragma: no cover - depends on environment
    _HAS_NUMBA = False
    njit = None                # type: ignore[assignment]


# Map of limiter name -> integer code (kept here so the kernel stays numeric).
LIMITER_CODES: dict[str, int] = {
    "vanleer": 0,
    "minmod": 1,
    "superbee": 2,
    "beamwarming": 3,
    "beam": 3,
    "osher": 4,
}


# --------------------------------------------------------------------------- #
# JIT kernel (compiled on first call, then reused for the whole run)
# --------------------------------------------------------------------------- #
if _HAS_NUMBA:

    @njit(cache=True)
    def _limiter_kernel(r: np.ndarray, kind: int) -> np.ndarray:
        """Element-wise TVD limiter, compiled by Numba.

        Operates on a flat or any-shape ``r`` array and returns the limiter
        value with the same shape.  ``r <= 0`` or non-finite entries yield 0
        (the scheme degrades to first-order upwind at extrema, as required for
        TVD monotonicity).

        The input is coerced to a C-contiguous array first: ``r`` arrives as a
        ``moveaxis`` view (often F-ordered) when interpolating along a non-zero
        axis, and the element-wise index loop below assumes C-order flattening.
        """
        r = np.ascontiguousarray(r)
        out = np.empty_like(r)
        flat_r = r.ravel()
        flat_o = out.ravel()
        n = flat_r.size
        for i in range(n):
            ri = flat_r[i]
            if not np.isfinite(ri) or ri <= 0.0:
                flat_o[i] = 0.0
            elif kind == 0:                       # vanleer
                flat_o[i] = 2.0 * ri / (ri + 1.0)
            elif kind == 1:                       # minmod
                flat_o[i] = ri if ri < 1.0 else 1.0
            elif kind == 2:                       # superbee
                a = 2.0 * ri if 2.0 * ri < 1.0 else 1.0
                b = ri if ri < 2.0 else 2.0
                flat_o[i] = a if a > b else b
            elif kind == 3:                       # beamwarming
                flat_o[i] = ri
            else:                                 # osher
                v = 2.0 * ri if 2.0 * ri < 2.0 else 2.0
                flat_o[i] = v if v > 0.0 else 0.0
        return out

    _limiter_kernel_impl = _limiter_kernel

else:

    def _limiter_kernel_impl(r: np.ndarray, kind: int) -> np.ndarray:  # type: ignore[no-redef]
        """Pure-NumPy fallback (identical numerics) used without Numba."""
        r = np.where(np.isfinite(r), r, 0.0)
        if kind == 0:
            return np.where(r > 0.0, 2.0 * r / (r + 1.0), 0.0)
        if kind == 1:
            return np.where(r > 0.0, np.minimum(r, 1.0), 0.0)
        if kind == 2:
            a = np.minimum(2.0 * r, 1.0)
            b = np.minimum(r, 2.0)
            return np.where(r > 0.0, np.maximum(a, b), 0.0)
        if kind == 3:
            return np.where(r > 0.0, r, 0.0)
        return np.where(r > 0.0, np.maximum(0.0, np.minimum(2.0 * r, 2.0)), 0.0)


def limiter_kernel(r: np.ndarray, name: str) -> np.ndarray:
    """Dispatch to the (JIT or fallback) limiter kernel by ``name``.

    This is the public entry point used by :mod:`numerics.interpolation`.
    The result is reshaped to the *original* ``r`` shape so the caller always
    sees the expected layout (the JIT kernel may return a C-contiguous copy
    when the input was a transposed view).
    """
    kind = LIMITER_CODES.get(name.lower(), 0)
    r_arr = np.asarray(r, dtype=np.float64)
    out = _limiter_kernel_impl(r_arr, kind)
    return out.reshape(r_arr.shape)


def has_numba() -> bool:
    """Whether Numba is available and the JIT kernel is in use."""
    return _HAS_NUMBA