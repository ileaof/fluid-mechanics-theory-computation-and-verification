"""Validate the GPU BiCGSTAB against the production CPU solver on the *real*
CFDPy pressure-Poisson operator, and benchmark both.

The operator is built by the actual :class:`solver.pressure.PressureSolver`
(only interior-face couplings -> every row sums to zero -> the constant null
space), with a uniform density, exactly the single-phase production path.  The
RHS is mean-projected and the solution mean-subtracted, as :meth:`PressureSolver
.solve` does for a no-outlet case.

Two CPU baselines are reported:

* **CPU + ILU**  -- the production :class:`LinearSolver` (BiCGSTAB + ILU).  ILU
  is a strong preconditioner, so this converges in few iterations; it is the
  true CPU cost of a single-phase (constant-density, fixed-dt) run, where the
  factorisation is cached.
* **CPU, no ILU** -- BiCGSTAB with *no* preconditioner.  This is the effective
  regime of a VOF / adaptive-dt run, where the matrix changes every step and
  the ILU factorisation is rebuilt to no net benefit (the profile measured
  ~1000 BiCGSTAB iterations/step there).  This is the fair comparison for the
  GPU Jacobi-BiCGSTAB.

Accuracy is reported as the L2 / L-infinity difference between the GPU and CPU
``delta_p`` fields (both mean-subtracted).
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


def _pressure_matrix(Nx, Ny):
    """The real production Poisson operator for a uniform-density no-outlet case."""
    cfg = Config(Nx=Nx, Ny=Ny, Nz=1, Lx=1.0, Ly=1.0, Lz=1.0, two_d=True)
    mesh = Mesh(Nx, Ny, 1, 1.0, 1.0, 1.0)
    bc = BoundaryCondition(cfg)          # default no-slip walls -> pure Neumann
    # PressureSolver only needs mesh/bc/cfg/ls for _matrix; pass a dummy ls.
    ps = PressureSolver(mesh, bc, cfg, linear_solver=None)
    rho = np.full((Nx, Ny, 1), 1000.0, dtype=np.float64)
    A = ps._matrix(rho).tocsr()
    A.sort_indices()
    return A


def _smooth(Nx, Ny):
    i = np.arange(Nx)[:, None]
    j = np.arange(Ny)[None, :]
    return (np.sin(2 * np.pi * i / Nx) * np.cos(2 * np.pi * j / Ny)).ravel()


def _err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    linf = float(np.max(np.abs(a - b))) if a.size else 0.0
    l2 = float(np.sqrt(np.mean((a - b) ** 2)))
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return linf, l2, linf / scale


def _res(A, x, b):
    return float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))


def solve_one(Nx, Ny, tol, maxiter, reps=3):
    A = _pressure_matrix(Nx, Ny)
    N = A.shape[0]
    b = _smooth(Nx, Ny)
    b = b - b.mean()

    # CPU + ILU (production single-phase path).
    cpu_ilu = LinearSolver(method="bicgstab", tol=tol, maxiter=maxiter,
                           use_ilu=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        dp_ilu = cpu_ilu.solve(A, b, x0=np.zeros_like(b))
    t_ilu = (time.perf_counter() - t0) / reps
    dp_ilu = dp_ilu - dp_ilu.mean()

    # CPU, no ILU (VOF / adaptive-dt effective regime).
    cpu_no = LinearSolver(method="bicgstab", tol=tol, maxiter=maxiter,
                          use_ilu=False)
    t0 = time.perf_counter()
    for _ in range(reps):
        dp_no = cpu_no.solve(A, b, x0=np.zeros_like(b))
    t_no = (time.perf_counter() - t0) / reps
    dp_no = dp_no - dp_no.mean()

    # GPU + Jacobi.
    bck = init_backend(use_gpu=True)
    Ad = bck.asarray(A.data); Ai = bck.asarray(A.indices); Ap = bck.asarray(A.indptr)
    bd = bck.asarray(b)
    solver = GPUBiCGSTAB(N, tol=tol, maxiter=maxiter, use_jacobi=True)
    solver.solve(Ad, Ai, Ap, bd)          # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        xd = solver.solve(Ad, Ai, Ap, bd)
    bck.synchronize()
    t_gpu = (time.perf_counter() - t0) / reps
    dp_gpu = bck.to_host(xd)
    dp_gpu = dp_gpu - dp_gpu.mean()

    e_ilu = _err(dp_gpu, dp_ilu)
    e_no = _err(dp_gpu, dp_no)
    return dict(grid=(Nx, Ny), N=N,
               t_ilu=t_ilu * 1e3, t_no=t_no * 1e3, t_gpu=t_gpu * 1e3,
               gpu_iters=solver.last_iterations, gpu_conv=solver.last_converged,
               res_gpu=_res(A, dp_gpu, b),
               err_vs_ilu=e_ilu, err_vs_no=e_no)


def main():
    print("GPU BiCGSTAB (Jacobi) vs CPU BiCGSTAB on the *real* pressure-Poisson")
    print("operator (pure-Neumann, mean-projected RHS, mean-subtracted solution).\n")
    tol, maxiter = 1e-7, 3000
    for (Nx, Ny) in [(60, 60), (200, 160), (400, 160)]:
        r = solve_one(Nx, Ny, tol, maxiter, reps=3)
        ei, en = r["err_vs_ilu"], r["err_vs_no"]
        print(f"grid {r['grid']}  N={r['N']:6d}  (GPU iters={r['gpu_iters']}, "
              f"conv={r['gpu_conv']}, true_res={r['res_gpu']:.1e})")
        print(f"  CPU+ILU  {r['t_ilu']:7.2f} ms   "
              f"CPU no-ILU {r['t_no']:7.2f} ms   GPU {r['t_gpu']:7.2f} ms")
        print(f"  speedup vs CPU+ILU: {r['t_ilu']/r['t_gpu']:4.2f}x   "
              f"vs CPU no-ILU: {r['t_no']/r['t_gpu']:4.2f}x")
        print(f"  dp GPU vs CPU+ILU :  L2={ei[1]:.2e}  relLinf={ei[2]:.2e}")
        print(f"  dp GPU vs CPU noILU: L2={en[1]:.2e}  relLinf={en[2]:.2e}\n")


if __name__ == "__main__":
    main()