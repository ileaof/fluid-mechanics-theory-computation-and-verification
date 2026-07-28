"""Validate gpu.kernels against NumPy references and benchmark them.

Every kernel here implements exactly the same float64 operation as its NumPy
reference, so the L2 / L-infinity errors should be at the round-off level
(~1e-14 relative).  The script prints per-kernel errors and a CPU-vs-GPU
timing for the sparse matvec and the dot reduction on a problem sized like the
cylinder-case Poisson system.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from gpu.backend import init_backend
from gpu import kernels as K


def _err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    linf = float(np.max(np.abs(a - b))) if a.size else 0.0
    denom = max(float(np.max(np.abs(b))), 1e-300)
    return linf, linf / denom


def main():
    b = init_backend(use_gpu=True)
    print(f"backend: {b.name}\n")
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    # Element-wise ops
    # ------------------------------------------------------------------ #
    n = 200_000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    a = 1.2345

    xd, yd = b.asarray(x), b.asarray(y)

    # copy
    yref = x.copy()
    K.copy(yd, xd, n)
    print(f"copy        Linf={_err(b.to_host(yd), yref)[0]:.3e}  "
          f"rel={_err(b.to_host(yd), yref)[1]:.3e}")

    # axpy: y += a*x  (reload the original y first; copy() clobbered yd)
    yd = b.asarray(y)
    yref = y + a * x
    K.axpy(yd, a, xd, n)
    print(f"axpy        Linf={_err(b.to_host(yd), yref)[0]:.3e}  "
          f"rel={_err(b.to_host(yd), yref)[1]:.3e}")

    # scale_add: y = a*y + x  (ref from the post-axpy y to mirror the device)
    yhost = b.to_host(yd)
    yref = a * yhost + x
    K.scale_add(yd, a, xd, n)
    print(f"scale_add   Linf={_err(b.to_host(yd), yref)[0]:.3e}  "
          f"rel={_err(b.to_host(yd), yref)[1]:.3e}")

    # fill
    K.fill(xd, 7.0, n)
    print(f"fill        Linf={_err(b.to_host(xd), np.full(n, 7.0))[0]:.3e}")

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    xd, yd = b.asarray(x), b.asarray(y)

    ref = float(np.dot(x, y))
    val = K.dot(xd, yd, n)
    print(f"dot         ref={ref:.6e} gpu={val:.6e} "
          f"rel={abs(val-ref)/max(abs(ref),1e-300):.3e}")

    ref = float(np.max(np.abs(x)))
    val = K.max_abs(xd, n)
    print(f"max_abs     ref={ref:.6e} gpu={val:.6e} "
          f"rel={abs(val-ref)/max(ref,1e-300):.3e}")

    ref = float(np.linalg.norm(x))
    val = K.norm2(xd, n)
    print(f"norm2       ref={ref:.6e} gpu={val:.6e} "
          f"rel={abs(val-ref)/max(ref,1e-300):.3e}")

    # ------------------------------------------------------------------ #
    # Sparse CSR matvec vs scipy
    # ------------------------------------------------------------------ #
    # Build a 5-diagonal 2-D Poisson-like matrix (matches the solver's stencil).
    Nx, Ny = 200, 160
    N = Nx * Ny
    D = sp.diags([-4 * np.ones(N), np.ones(N), np.ones(N),
                  np.ones(N), np.ones(N)],
                 offsets=[0, 1, -1, Ny, -Ny], shape=(N, N)).tocsr()
    # zero out the wrap-around entries at row/column boundaries so it is a true
    # 2-D Poisson (no periodic coupling across the Ny stride).
    D = D.tolil()
    for r in range(N):
        if (r % Ny) == Ny - 1 and r + 1 < N:
            D[r, r + 1] = 0.0
        if (r % Ny) == 0 and r - 1 >= 0:
            D[r, r - 1] = 0.0
    D = D.tocsr()
    D.sort_indices()

    x = rng.standard_normal(N)
    x_ref = D @ x

    Ad, Ai, Ap = b.asarray(D.data), b.asarray(D.indices), b.asarray(D.indptr)
    xd = b.asarray(x)
    yd = b.zeros(N)
    K.matvec_csr(Ad, Ai, Ap, xd, yd, N)
    print(f"matvec_csr  Linf={_err(b.to_host(yd), x_ref)[0]:.3e}  "
          f"rel={_err(b.to_host(yd), x_ref)[1]:.3e}  (N={N})")

    # ------------------------------------------------------------------ #
    # Benchmark: matvec + dot, CPU vs GPU
    # ------------------------------------------------------------------ #
    b.synchronize()
    reps = 2000

    t0 = time.perf_counter()
    for _ in range(reps):
        K.matvec_csr(Ad, Ai, Ap, xd, yd, N)
    b.synchronize()
    t_gpu_mv = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = D @ x
    t_cpu_mv = (time.perf_counter() - t0) / reps

    # dot benchmark on its own large vectors (independent of the matvec x).
    bx = rng.standard_normal(n)
    by = rng.standard_normal(n)
    bxd, byd = b.asarray(bx), b.asarray(by)
    t0 = time.perf_counter()
    for _ in range(reps):
        K.dot(bxd, byd, n)
    b.synchronize()
    t_gpu_dot = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = np.dot(bx, by)
    t_cpu_dot = (time.perf_counter() - t0) / reps

    print("\n--- benchmark (median of repeated calls) ---")
    print(f"matvec  N={N}:  CPU {t_cpu_mv*1e6:8.1f} us   "
          f"GPU {t_gpu_mv*1e6:8.1f} us   speedup {t_cpu_mv/t_gpu_mv:5.2f}x")
    print(f"dot     n={n}:  CPU {t_cpu_dot*1e6:8.1f} us   "
          f"GPU {t_gpu_dot*1e6:8.1f} us   speedup {t_cpu_dot/t_gpu_dot:5.2f}x")


if __name__ == "__main__":
    main()