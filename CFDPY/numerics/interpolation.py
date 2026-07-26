"""Face-value interpolation for the convective terms.

The convective term of a transport equation is

.. math::
    \\nabla \\cdot (\\mathbf{u}\\,\\phi) \\;\\approx\\;
    \\frac{1}{V}\\sum_f \\,(\\mathbf{u}\\cdot\\mathbf{n})_f\\,\\phi_f .

The choice of how the face value :math:`\\phi_f` is reconstructed from the
neighbouring cell values is what distinguishes the convection schemes:

* **Upwind**  -- first order, monotone but numerically diffusive;
* **Central** -- second order, no numerical diffusion, oscillatory on sharp
  gradients;
* **QUICK**   -- third-order upwind-biased (Leonard, 1979), sharp but not
  monotone;
* **TVD**     -- second-order upwind with a flux limiter that restores
  monotonicity near discontinuities (ideal for VOF / shocks).

All schemes operate along a single axis at a time.  ``phi`` is a cell-centred
field and ``uf`` the face velocity along *axis*; the returned face values have
the same shape as ``uf`` (``Ncells+1`` along *axis*).  The reconstruction uses
a one-cell reflective padding so the high-order stencils (QUICK/TVD) stay valid
up to the domain boundary, where the boundary-condition layer overwrites the
face value afterwards.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# TVD limiters
# ---------------------------------------------------------------------------
def limiter(r: np.ndarray, name: str = "vanleer") -> np.ndarray:
    """Symmetric TVD flux limiter evaluated on the ratio ``r``.

    ``r`` is the ratio of consecutive gradients (upwind / downwind).  Where the
    limiter is undefined (``r<=0``) the result is zero, which degrades the
    scheme to first-order upwind at extrema -- exactly the desired TVD
    behaviour.

    The element-wise evaluation is dispatched to :mod:`numerics.numba_kernels`,
    which compiles the per-element branches with Numba when available (a
    genuine speed-up for the TVD scheme, which mixes comparisons and
    arithmetic per element) and falls back to a pure-NumPy implementation
    otherwise.  Both paths produce identical numerics.
    """

    from numerics.numba_kernels import limiter_kernel
    return limiter_kernel(r, name)


# ---------------------------------------------------------------------------
# Face interpolation along one axis
# ---------------------------------------------------------------------------
def face_interpolate(phi: np.ndarray, uf: np.ndarray, axis: int,
                     scheme: str = "upwind",
                     limiter_name: str = "vanleer") -> np.ndarray:
    """Reconstruct ``phi`` at the faces normal to *axis*.

    Parameters
    ----------
    phi:
        Cell-centred field.
    uf:
        Face velocity along *axis*, shape ``Ncells+1`` along *axis*.
    axis:
        ``0``, ``1`` or ``2``.
    scheme:
        ``upwind``, ``central``, ``quick`` or ``tvd``.
    limiter_name:
        TVD limiter (only used by the ``tvd`` scheme).

    Returns
    -------
    numpy.ndarray
        Face values of ``phi`` with the same shape as ``uf``.
    """

    scheme = scheme.lower()
    if scheme not in ("upwind", "central", "quick", "tvd"):
        raise ValueError(f"Unknown convection scheme: {scheme}")

    P = np.moveaxis(phi, axis, 0)          # (n, ...)
    U = np.moveaxis(uf, axis, 0)           # (n+1, ...)
    n = P.shape[0]
    phif = np.zeros_like(U)

    # Reflective padding so the far-upwind stencil is always defined.
    Pp = np.concatenate([P[0:1], P, P[-1:]], axis=0)   # (n+2, ...)
    # internal faces indices 1..n  (face k between cells k-1 and k)
    L = Pp[1:n]        # left  cell of each internal face  -> P[0:n-1]
    R = Pp[2:n + 1]    # right cell of each internal face -> P[1:n]
    pos = U[1:n] > 0.0  # sign on internal faces, shape (n-1, ...)

    # central
    central = 0.5 * (L + R)

    if scheme == "central":
        phif[1:n] = central
        phif[0], phif[-1] = P[0], P[-1]
        return np.moveaxis(phif, 0, axis)

    # upwind
    upwind = np.where(pos, L, R)
    phif[1:n] = upwind
    phif[0], phif[-1] = P[0], P[-1]
    if scheme == "upwind":
        return np.moveaxis(phif, 0, axis)

    # far-upwind cells (one cell beyond the upwind cell)
    far_pos = Pp[0:n - 1]      # cell i-1 for positive flow
    far_neg = Pp[3:n + 2]      # cell i+2 for negative flow
    far = np.where(pos, far_pos, far_neg)

    if scheme == "quick":
        # Leonard QUICK: third-order upwind-biased.
        quick_pos = L + 0.125 * (3.0 * R - 2.0 * L + far_pos)
        quick_neg = R - 0.125 * (3.0 * L - 2.0 * R + far_neg)
        phif[1:n] = np.where(pos, quick_pos, quick_neg)
        phif[0], phif[-1] = P[0], P[-1]
        return np.moveaxis(phif, 0, axis)

    # TVD:  phi_f = phi_upwind + 0.5 * lim(r) * (phi_downwind - phi_upwind)
    downwind = np.where(pos, R, L)
    d = downwind - upwind
    r = np.where(np.abs(d) > 1e-30, (upwind - far) / np.where(d == 0, 1.0, d), 0.0)
    r = np.where(np.isfinite(r), r, 0.0)
    lim = limiter(r, limiter_name)
    phif[1:n] = upwind + 0.5 * lim * d
    phif[0], phif[-1] = P[0], P[-1]
    return np.moveaxis(phif, 0, axis)


def tvd_face_value(phi: np.ndarray, uf: np.ndarray, axis: int,
                   limiter_name: str = "vanleer") -> np.ndarray:
    """Convenience alias for the TVD scheme."""
    return face_interpolate(phi, uf, axis, scheme="tvd", limiter_name=limiter_name)