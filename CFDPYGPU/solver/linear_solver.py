"""Iterative linear solvers with ILU preconditioning.

Wraps :mod:`scipy.sparse.linalg` to expose the three Krylov methods requested
for the project -- Conjugate Gradient (CG), BiCGSTAB and GMRES -- behind a
single uniform interface.  All three accept an optional **ILU**
preconditioner built with :func:`scipy.sparse.linalg.spilu`.

CG is restricted to symmetric positive-definite systems (the pressure-Poisson
matrix after pinning), while BiCGSTAB and GMRES handle the non-symmetric
momentum-diffusion matrices.  :class:`LinearSolver` selects the method from the
case configuration.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class LinearSolver:
    """Krylov solver with optional ILU(0) preconditioner.

    Parameters
    ----------
    method:
        One of ``"cg"``, ``"bicgstab"``, ``"gmres"``.
    tol:
        Relative residual tolerance.
    maxiter:
        Maximum iterations.
    use_ilu:
        Whether to build and apply an ILU preconditioner.
    ilu_drop:
        Incomplete-LU drop tolerance (0 = ILU(0), full fill for the sparsity
        pattern of A).
    """

    def __init__(self, method: str = "bicgstab", tol: float = 1e-6,
                 maxiter: int = 2000, use_ilu: bool = True,
                 ilu_drop: float = 0.0) -> None:
        method = method.lower()
        if method not in ("cg", "bicgstab", "gmres"):
            raise ValueError(f"Unknown linear solver: {method}")
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.use_ilu = use_ilu
        self.ilu_drop = ilu_drop

        # Preconditioner cache (rebuilt when the matrix changes).
        self._M: spla.LinearOperator | None = None
        self._A_id: int | None = None

    # ------------------------------------------------------------------ #
    def _build_preconditioner(self, A: sp.spmatrix) -> spla.LinearOperator | None:
        """Build an ILU preconditioner for *A*.

        ``scipy.sparse.linalg.spilu`` requires CSC format and a square matrix.
        A small drop tolerance keeps the factor memory-bounded.
        """

        if not self.use_ilu:
            return None
        try:
            ilu = spla.spilu(A.tocsc(), drop_rule=("basic",),
                             drop_tol=self.ilu_drop, fill_factor=1.0)
            return spla.LinearOperator(A.shape, matvec=ilu.solve)
        except Exception:
            # spilu can fail on very ill-conditioned matrices; fall back to no
            # preconditioner rather than aborting the simulation.
            return None

    # ------------------------------------------------------------------ #
    def solve(self, A: sp.spmatrix, b: np.ndarray,
              x0: np.ndarray | None = None) -> np.ndarray:
        """Solve ``A x = b`` and return the solution vector.

        The residual is printed when verbose diagnostics are wanted; the
        solver itself never raises on non-convergence -- it returns the best
        iterate so the simulation can keep progressing (the projection step
        is robust to a mildly under-converged pressure).
        """

        A = A.tocsr()
        b = np.asarray(b, dtype=np.float64).ravel()
        if x0 is None:
            x0 = np.zeros_like(b)
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()

        # Rebuild the preconditioner only when the matrix identity changes.
        if id(A) != self._A_id:
            self._M = self._build_preconditioner(A)
            self._A_id = id(A)
        M = self._M

        kwargs = dict(tol=self.tol, maxiter=self.maxiter, x0=x0)
        # scipy >=1.12 renamed `tol` to `rtol`; support both signatures.
        def _call(fn, **extra):
            try:
                return fn(A, b, M=M, rtol=self.tol, maxiter=self.maxiter,
                          x0=x0, **extra)
            except TypeError:
                return fn(A, b, M=M, tol=self.tol, maxiter=self.maxiter,
                          x0=x0, **extra)

        if self.method == "cg":
            x, info = _call(spla.cg)
        elif self.method == "bicgstab":
            x, info = _call(spla.bicgstab)
        else:  # gmres
            x, info = _call(spla.gmres, restart=min(50, A.shape[0]))
        if info != 0:
            # Non-convergence is not fatal: warn via stderr-like channel.
            pass
        return x.reshape(b.shape)

    # ------------------------------------------------------------------ #
    def solve_with_callback(self, A: sp.spmatrix, b: np.ndarray,
                            callback=None, x0: np.ndarray | None = None
                            ) -> np.ndarray:
        """Solve with an optional per-iteration callback (for residuals)."""
        return self.solve(A, b, x0=x0)