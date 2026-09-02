"""Validate the GPU geometric multigrid against the production CPU path on the
*real* CFDPy pressure-Poisson operator, and benchmark both.

Exactly the protocol of :mod:`gpu.validate_linear`, extended with a
variable-density (two-phase) case -- the regime multigrid exists for:

* the operator is built by the actual :class:`solver.pressure.PressureSolver`
  (interior-face couplings only -> pure Neumann, constant null space), with
  uniform density (single-phase) and a water/air drop field (VOF);
* the RHS is mean-projected and the solution mean-subtracted, as
  :meth:`PressureSolver.solve` does for a no-outlet case;
* the reference solution is a *direct* sparse solve (pinned row) -- ground
  truth independent of any Krylov tolerance;
* timing compares multigrid against the GPU Jacobi-BiCGSTAB and the CPU
  no-ILU BiCGSTAB (the effective VOF / adaptive-dt regime).

Accuracy is reported as the L2 / L-infinity difference between the multigrid
and the direct solution (both mean-subtracted), plus the true residual
``||A x - b|| / ||b||`` on the production CSR matrix.
"""
from __future__ import annotations

import time

import numpy as np

from config import Config
from mesh import Mesh
from solver.boundary import BoundaryCondition
from solver.pressure import PressureSolver
from solver.linear_solver import LinearSolver
from gpu.backend import init_backend
from gpu.linear import GPUBiCGSTAB
from gpu.multigrid import GPUGeometricMultigrid


def _case(Nx, Ny, Nz, variable_rho):
    """The real production Poisson setup: mesh + BCs + (rho, face coeffs)."""
    cfg = Config(Nx=Nx, Ny=Ny, Nz=Nz, Lx=1.0, Ly=1.0, Lz=1.0,
                 two_d=(Nz == 1))
    mesh = Mesh(Nx, Ny, Nz, 1.0, 1.0, 1.0)
    bc = BoundaryCondition(cfg)          # default no-slip walls -> pure Neumann
    ps = PressureSolver(mesh, bc, cfg, linear_solver=None)
    shape = (Nx, Ny, Nz)
    if variable_rho:
        # a water drop in air, like the splash init: sharp density jump
        i = np.arange(Nx)[:, None, None]
        j = np.arange(Ny)[None, :, None]
        k = np.arange(Nz)[None, None, :]
        r2 = (i / Nx - 0.5) ** 2 + (j / Ny - 0.45) ** 2 \
            + ((k / Nz - 0.5) ** 2 if Nz > 1 else 0.0)
        rho = np.where(r2 < 0.04, 1000.0, 1.2)
    else:
        rho = np.full(shape, 1000.0)
    rho = np.ascontiguousarray(rho)
    A = ps._matrix(rho).tocsr()
    A.sort_indices()
    irx, iry, irz = ps._inv_rho_faces(rho)
    hx2, hy2, hz2 = mesh.dx ** 2, mesh.dy ** 2, mesh.dz ** 2
    cx = irx[1:Nx] / hx2
    cy = iry[:, 1:Ny] / hy2
    cz = irz[:, :, 1:Nz] / hz2 if Nz > 1 else None
    return mesh, A, rho, cx, cy, cz


def _rhs(mesh, kind="smooth"):
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    i = np.arange(Nx)[:, None, None]
    j = np.arange(Ny)[None, :, None]
    k = np.arange(Nz)[None, None, :]
    f = (np.sin(2 * np.pi * i / Nx) * np.cos(2 * np.pi * j / Ny)
         * (np.sin(2 * np.pi * k / Nz) if Nz > 1 else 1.0))
    return f - f.mean()


def _err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    linf = float(np.max(np.abs(a - b))) if a.size else 0.0
    l2 = float(np.sqrt(np.mean((a - b) ** 2)))
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return linf, l2, linf / scale


def _res(A, x, b):
    return float(np.linalg.norm(A @ x.ravel() - b.ravel())
                 / max(np.linalg.norm(b), 1e-300))


def _direct(A, b):
    """Ground truth: direct sparse solve with one pinned row (null space)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    bm = b - b.mean()
    A2 = A.tolil()
    A2.rows[0] = [0]
    A2.data[0] = [1.0]
    x = np.asarray(spsolve(A2.tocsr(), bm.ravel()))
    return x - x.mean()


def solve_one(Nx, Ny, Nz, variable_rho, tol=1e-7, reps=3):
    mesh, A, rho, cx, cy, cz = _case(Nx, Ny, Nz, variable_rho)
    N = A.shape[0]
    b = _rhs(mesh)
    label = f"grid {(Nx, Ny, Nz)}  N={N:7d}  " \
            f"{'two-phase' if variable_rho else 'single-phase'}"

    # Ground truth (direct solve).
    x_ref = _direct(A, b)

    # CPU, no ILU (VOF / adaptive-dt effective regime).
    cpu = LinearSolver(method="bicgstab", tol=tol, maxiter=3000,
                       use_ilu=False)
    t0 = time.perf_counter()
    for _ in range(reps):
        dp_cpu = cpu.solve(A, b, x0=np.zeros(N))
    t_cpu = (time.perf_counter() - t0) / reps
    dp_cpu = dp_cpu - dp_cpu.mean()

    # GPU geometric multigrid.
    bck = init_backend(use_gpu=True)
    mg = GPUGeometricMultigrid((Nx, Ny, Nz), tol=tol, max_cycles=100)
    mg.set_face_coefficients(cx, cy, cz)
    mg.solve(b)                                    # warm up (JIT)
    t0 = time.perf_counter()
    for _ in range(reps):
        dp_mg = mg.solve(b)
    bck.synchronize()
    t_mg = (time.perf_counter() - t0) / reps

    # GPU Jacobi-BiCGSTAB (the increment this replaces).
    Ad = bck.asarray(A.data); Ai = bck.asarray(A.indices)
    Ap_ = bck.asarray(A.indptr)
    bd = bck.asarray((b - b.mean()).ravel())
    kry = GPUBiCGSTAB(N, tol=tol, maxiter=3000, use_jacobi=True)
    kry.solve(Ad, Ai, Ap_, bd)                     # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        xd = kry.solve(Ad, Ai, Ap_, bd)
    bck.synchronize()
    t_kry = (time.perf_counter() - t0) / reps
    dp_kry = bck.to_host(xd)
    dp_kry = dp_kry - dp_kry.mean()

    e_ref = _err(dp_mg, x_ref)
    return dict(label=label, N=N,
                t_cpu=t_cpu * 1e3, t_kry=t_kry * 1e3, t_mg=t_mg * 1e3,
                mg_cycles=mg.last_cycles, mg_conv=mg.last_converged,
                res_mg=_res(A, dp_mg, b),
                err_vs_ref=e_ref, err_vs_kry=_err(dp_mg, dp_kry))


def main():
    print("GPU geometric multigrid vs CPU/GPU Krylov on the *real* pressure-")
    print("Poisson operator (pure-Neumann, mean-projected RHS, mean-subtracted")
    print("solution).  Reference: direct sparse solve.\n")
    tol = 1e-7
    for (Nx, Ny, Nz, vr) in [(60, 60, 1, False), (200, 160, 1, False),
                             (200, 160, 1, True), (64, 64, 32, False),
                             (64, 64, 32, True)]:
        r = solve_one(Nx, Ny, Nz, vr, tol=tol)
        e = r["err_vs_ref"]
        print(f"{r['label']}")
        print(f"  MG cycles={r['mg_cycles']} conv={r['mg_conv']} "
              f"true_res={r['res_mg']:.1e}")
        print(f"  CPU no-ILU {r['t_cpu']:8.1f} ms   "
              f"GPU BiCGSTAB {r['t_kry']:8.1f} ms   "
              f"GPU MG {r['t_mg']:8.2f} ms")
        print(f"  speedup MG vs CPU no-ILU: {r['t_cpu']/r['t_mg']:7.1f}x   "
              f"vs GPU BiCGSTAB: {r['t_kry']/r['t_mg']:6.1f}x")
        print(f"  dp MG vs direct   :  L2={e[1]:.2e}  relLinf={e[2]:.2e}\n")


if __name__ == "__main__":
    main()