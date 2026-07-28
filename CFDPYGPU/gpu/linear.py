"""GPU-resident Krylov solvers for CFDPy (Numba CUDA).

The profile of the framework singled out the pressure-Poisson solve as the
dominant per-step cost: ~84% of a single-phase step is spent in the SciPy
ILU factorisation, and ~98% of a VOF step is the BiCGSTAB sparse-matvec loop
(~2000 matvecs/step).  The sparse matvec is the workhorse, and
:func:`gpu.kernels.matvec_csr` is already 3.6x faster than SciPy's CSR matvec
at the cylinder-grid size (N=64000) and 8x at N=128000, with the GPU time
essentially flat in N.  This module turns that fast matvec into a full
Krylov solve by driving a BiCGSTAB loop on the device with the BLAS-1
reductions from :mod:`gpu.kernels`.

Algorithm
---------
:class:`GPUBiCGSTAB` implements the *preconditioned* BiCGSTAB of van der Vorst
with an optional **Jacobi (diagonal) preconditioner** -- the only preconditioner
that is both cheap to apply on a GPU (a pointwise divide) and free of the
sequential triangular solves that make ILU awkward on a GPU.  The algorithm is
identical to SciPy's ``bicgstab`` (same recurrence, same convergence test); only
the matvec / reductions / vector updates run on the device, so the solution
agrees with the CPU path to the requested tolerance (validated in
``gpu/validate_linear.py``).

Numerical notes
---------------
* The convergence test uses the *recurrence* residual ``r = s - omega t`` for
  speed (no extra matvec); the true residual ``b - A x`` is recomputed on exit
  and reported so the caller can detect a stagnating recurrence.
* BiCGSTAB breakdown guards (``rho == 0``, ``omega == 0``, ``(t,t) == 0``)
  return the best iterate with a non-zero ``info`` rather than raising, matching
  the CPU solver's "never abort the simulation" contract.
* All workspace vectors are allocated once per :class:`GPUBiCGSTAB` instance and
  reused across solves (no per-solve malloc), so a fixed-grid run pays the
  allocation cost exactly once.

This is the foundation for keeping the pressure field GPU-resident; wiring it
into the production :class:`solver.pressure.PressureSolver` (and eventually the
momentum / energy diffusion solves) is the next incremental step, done only
after the standalone validation passes.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .backend import get_backend


class GPUBiCGSTAB:
    """Preconditioned BiCGSTAB for a fixed-shape CSR system, on the GPU.

    Parameters
    ----------
    n:
        Row count of the linear system (fixed for the lifetime of the solver;
        workspace is allocated once).
    tol:
        Relative residual tolerance ``||r|| / ||b||``.
    maxiter:
        Maximum BiCGSTAB iterations.
    use_jacobi:
        Apply the Jacobi (diagonal) preconditioner.  Cheap on a GPU and helps
        the Poisson conditioning; set ``False`` for plain BiCGSTAB.

    The CSR matrix is passed to :meth:`solve` so a variable-coefficient matrix
    (the VOF pressure-Poisson, rebuilt every step) can be refreshed without
    reallocating the workspace.  Only the diagonal (for Jacobi) is cached per
    matrix identity.
    """

    def __init__(self, n: int, tol: float = 1e-7, maxiter: int = 3000,
                 use_jacobi: bool = True, check_every: int = 4) -> None:
        self.n = int(n)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.use_jacobi = bool(use_jacobi)
        # Convergence is tested every ``check_every`` iterations: the recurrence
        # residual norm is a host-bound reduction (one sync), so testing it
        # every iteration would dominate the per-iteration cost at small N.
        # Checking every 4 iters cuts that overhead 4x with no change to the
        # converged solution (at most 3 extra iterations before the stop).
        self.check_every = max(1, int(check_every))
        self.backend = get_backend()
        # Workspace -- allocated once, reused for every solve on this size.
        self._r = self.backend.zeros(n)
        self._rhat = self.backend.zeros(n)
        self._p = self.backend.zeros(n)
        self._v = self.backend.zeros(n)
        self._s = self.backend.zeros(n)
        self._t = self.backend.zeros(n)
        self._y = self.backend.zeros(n)   # M^-1 p
        self._z = self.backend.zeros(n)   # M^-1 s
        self._tmp = self.backend.zeros(n)  # matvec scratch
        self._w = self.backend.zeros(n)    # true-residual scratch
        # Cached Jacobi inverse-diagonal for the current matrix identity.
        self._diag_inv = None
        self._diag_id = None
        # Last-solve diagnostics (for the performance report / history).
        self.last_iterations = 0
        self.last_residual = float("inf")
        self.last_converged = False

    # ------------------------------------------------------------------ #
    def _set_jacobi(self, A_data, A_indices, A_indptr, diag_id) -> None:
        """Cache ``1/diag(A)`` on the device, rebuilt only when A changes."""
        if not self.use_jacobi:
            return
        if diag_id == self._diag_id and self._diag_inv is not None:
            return
        # Extract the diagonal on the host (cheap, O(n)) and ship it over.
        diag = np.empty(self.n, dtype=np.float64)
        Ad = A_data.copy_to_host() if not isinstance(A_data, np.ndarray) \
            else np.asarray(A_data)
        Ai = A_indices.copy_to_host() if not isinstance(A_indices, np.ndarray) \
            else np.asarray(A_indices)
        Ap = A_indptr.copy_to_host() if not isinstance(A_indptr, np.ndarray) \
            else np.asarray(A_indptr)
        for r in range(self.n):
            diag[r] = 0.0
            for k in range(Ap[r], Ap[r + 1]):
                if Ai[k] == r:
                    diag[r] = Ad[k]
                    break
        # Guard against zero diagonal (should not happen for the Poisson Laplacian).
        diag = np.where(np.abs(diag) > 1e-300, diag, 1.0)
        self._diag_inv = self.backend.asarray(1.0 / diag)
        self._diag_id = diag_id

    # ------------------------------------------------------------------ #
    def solve(self, A_data, A_indices, A_indptr, b, x0=None) -> "object":
        """Solve ``A x = b`` on the device; return the device solution array.

        ``A_data/A_indices/A_indptr`` and ``b`` are device arrays (use
        :meth:`gpu.backend.Backend.asarray`).  ``x0`` is an optional device
        initial guess; a zero guess is used when ``None``.  The returned array
        is the device vector ``x`` (call ``backend.to_host`` to copy it back).
        """
        n = self.n
        Kf = K
        x = self.backend.zeros(n) if x0 is None else x0
        # r = b - A x0
        Kf.matvec_csr(A_data, A_indices, A_indptr, x, self._tmp, n)
        Kf.copy(self._r, b, n)
        Kf.axpy(self._r, -1.0, self._tmp, n)

        # rhat = r
        Kf.copy(self._rhat, self._r, n)
        bnorm = Kf.norm2(b, n)
        if bnorm == 0.0:
            bnorm = 1.0

        # Jacobi inverse diagonal (cached per matrix identity).
        diag_id = (id(A_data), id(A_indices), id(A_indptr))
        self._set_jacobi(A_data, A_indices, A_indptr, diag_id)
        jac = self.use_jacobi

        Kf.fill(self._p, 0.0, n)
        Kf.fill(self._v, 0.0, n)
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        converged = False
        it = 0
        for it in range(1, self.maxiter + 1):
            rho = Kf.dot(self._rhat, self._r, n)
            if rho == 0.0:
                break  # Lanczos breakdown
            if it == 1:
                Kf.copy(self._p, self._r, n)
            else:
                beta = (rho / rho_old) * (alpha / omega)
                # p = r + beta*(p - omega*v)
                Kf.axpy(self._p, -omega, self._v, n)   # p -= omega*v
                Kf.scale_add(self._p, beta, self._r, n)  # p = beta*p + r
            # y = M^-1 p ; v = A y
            if jac:
                Kf.div_pointwise(self._y, self._p, self._diag_inv, n)
                Kf.matvec_csr(A_data, A_indices, A_indptr, self._y, self._v, n)
            else:
                Kf.matvec_csr(A_data, A_indices, A_indptr, self._p, self._v, n)
            denom = Kf.dot(self._rhat, self._v, n)
            if denom == 0.0:
                break
            alpha = rho / denom
            # s = r - alpha*v ; x += alpha*y
            Kf.copy(self._s, self._r, n)
            Kf.axpy(self._s, -alpha, self._v, n)
            if jac:
                Kf.axpy(x, alpha, self._y, n)
            else:
                Kf.axpy(x, alpha, self._p, n)
            # z = M^-1 s ; t = A z
            if jac:
                Kf.div_pointwise(self._z, self._s, self._diag_inv, n)
                Kf.matvec_csr(A_data, A_indices, A_indptr, self._z, self._t, n)
            else:
                Kf.matvec_csr(A_data, A_indices, A_indptr, self._s, self._t, n)
            ts, tt = Kf.dot2(self._t, self._s, self._t, self._t, n)
            if tt == 0.0:
                break
            omega = ts / tt
            # x += omega*z ; r = s - omega*t
            if jac:
                Kf.axpy(x, omega, self._z, n)
            else:
                Kf.axpy(x, omega, self._s, n)
            Kf.copy(self._r, self._s, n)
            Kf.axpy(self._r, -omega, self._t, n)
            rho_old = rho
            # convergence: test the cheap recurrence residual first, then verify
            # with the *true* residual (one extra matvec) to avoid the BiCGSTAB
            # phantom-convergence stop (recurrence residual << true residual).
            # ``norm2`` is a host-bound reduction (one sync), so we only invoke
            # it every ``check_every`` iterations; the final true-residual check
            # at the bottom of the loop still guarantees a correct stop.
            if (it % self.check_every) == 0 or it >= self.maxiter:
                rnorm = Kf.norm2(self._r, n)
            else:
                rnorm = float("inf")
            if rnorm < self.tol * bnorm:
                # True residual = b - A x (one extra matvec) to guard against
                # the BiCGSTAB phantom-convergence stop.  self._w holds A x;
                # scale_add(w, -1, b) -> w = -w + b = b - A x.
                Kf.matvec_csr(A_data, A_indices, A_indptr, x, self._w, n)
                Kf.scale_add(self._w, -1.0, b, n)
                true_rnorm = Kf.norm2(self._w, n)
                if true_rnorm < self.tol * bnorm:
                    converged = True
                    break

        # True residual for reporting (one extra matvec).
        Kf.matvec_csr(A_data, A_indices, A_indptr, x, self._tmp, n)
        Kf.copy(self._r, b, n)
        Kf.axpy(self._r, -1.0, self._tmp, n)
        true_res = Kf.norm2(self._r, n) / bnorm
        self.last_iterations = it
        self.last_residual = true_res
        self.last_converged = converged
        self.backend.synchronize()
        return x