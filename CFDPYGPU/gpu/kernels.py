"""Low-level CUDA kernels for CFDPy (Numba ``@cuda.jit``).

This module holds the device kernels reused by the GPU linear solver
(:mod:`gpu.linear`) and, later, by the GPU-resident stencil operators.  They
are deliberately tiny and numerics-faithful: each implements exactly the same
floating-point operation as its NumPy reference, so a CPU-vs-GPU validation can
expect agreement to round-off (no algorithmic change).

Two families:

* **Sparse linear algebra** -- :func:`matvec_csr` (CSR ``y = A x``), the workhorse
  of every Krylov iteration and the dominant cost of the VOF pressure-Poisson
  (the profile measured ~2000 matvecs/step there).
* **BLAS-1 reductions / vector ops** -- :func:`dot`, :func:`max_abs`,
  :func:`norm2`, and the in-place element-wise :func:`copy`, :func:`axpy`,
  :func:`scale_add`, :func:`fill`.  These are what a BiCGSTAB driver is built
  from.

Reduction strategy
------------------
:func:`dot` and :func:`max_abs` use a two-level reduction: each thread block
tree-reduces a grid-stride chunk in shared memory and writes *one* partial sum
to a small ``partials`` array (one entry per block -- no atomics, deterministic).
The host then reduces the ``O(num_blocks)`` partials in plain Python/NumPy.
This avoids float atomics (non-deterministic ordering) and keeps the kernel
register-light.  The partials buffer is allocated per call from the Numba
device pool, so repeated calls of the same size reuse cached memory.

All wrappers accept *device* arrays (from :mod:`gpu.backend`) and synchronise
before returning a host scalar, so the caller sees completed results.
"""

from __future__ import annotations

import numpy as np

# Numba CUDA is required for this module; it is only imported when the backend
# is the CUDA one, so a CPU-only machine never reaches here.
from numba import cuda, float64, int32

# Block size for the 1-D reductions / element-wise kernels.  A power of two so
# the shared-memory tree reduction is exact; 256 threads keeps register pressure
# low and gives full warps on every NVIDIA architecture.
_BLOCK = 256


# =========================================================================== #
# Sparse CSR matrix-vector product:  y = A @ x
# =========================================================================== #
@cuda.jit
def _matvec_csr_kernel(data, indices, indptr, x, y, n):
    """One thread per row, grid-stride over rows.

    Each row is owned by exactly one thread, so the accumulation into ``y[r]``
    is race-free without atomics.  The CSR triple is the standard
    ``data/indices/indptr`` of a :class:`scipy.sparse.csr_matrix`.
    """
    r = cuda.grid(1)
    if r < n:
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += data[k] * x[indices[k]]
        y[r] = acc


def matvec_csr(A_data, A_indices, A_indptr, x, y, n: int) -> None:
    """Compute ``y = A @ x`` for a CSR matrix on the device (in place).

    ``A_data``, ``A_indices``, ``A_indptr``, ``x``, ``y`` are device arrays;
    ``n`` is the row count.  The caller ensures ``y`` is allocated and that
    ``x`` / the CSR arrays are already on the device.
    """
    threads = _BLOCK
    blocks = (n + threads - 1) // threads
    _matvec_csr_kernel[blocks, threads](A_data, A_indices, A_indptr, x, y, n)


# =========================================================================== #
# Element-wise vector ops (in place)
# =========================================================================== #
@cuda.jit
def _copy_kernel(y, x, n):
    i = cuda.grid(1)
    if i < n:
        y[i] = x[i]


@cuda.jit
def _axpy_kernel(y, a, x, n):
    """``y += a * x`` (in place on ``y``)."""
    i = cuda.grid(1)
    if i < n:
        y[i] = y[i] + a * x[i]


@cuda.jit
def _scale_add_kernel(y, a, x, n):
    """``y = a * y + x`` (in place on ``y``)."""
    i = cuda.grid(1)
    if i < n:
        y[i] = a * y[i] + x[i]


@cuda.jit
def _fill_kernel(x, v, n):
    i = cuda.grid(1)
    if i < n:
        x[i] = v


@cuda.jit
def _div_pointwise_kernel(z, x, d, n):
    """``z[i] = x[i] / d[i]`` (pointwise; used by the Jacobi preconditioner)."""
    i = cuda.grid(1)
    if i < n:
        z[i] = x[i] / d[i]


def _launch1d(kernel, *args):
    """Launch a 1-D element-wise kernel over ``n`` elements (last arg)."""
    n = args[-1]
    threads = _BLOCK
    blocks = (n + threads - 1) // threads
    kernel[blocks, threads](*args)


def copy(y, x, n: int) -> None:
    """``y = x`` (device arrays, in place on ``y``)."""
    _launch1d(_copy_kernel, y, x, n)


def axpy(y, a: float, x, n: int) -> None:
    """``y += a * x`` (device arrays, in place on ``y``)."""
    _launch1d(_axpy_kernel, y, a, x, n)


def scale_add(y, a: float, x, n: int) -> None:
    """``y = a * y + x`` (device arrays, in place on ``y``)."""
    _launch1d(_scale_add_kernel, y, a, x, n)


def fill(x, v: float, n: int) -> None:
    """``x[:] = v`` (device array, in place)."""
    _launch1d(_fill_kernel, x, v, n)


def div_pointwise(z, x, d, n: int) -> None:
    """``z = x / d`` pointwise (device arrays; used by Jacobi preconditioning)."""
    _launch1d(_div_pointwise_kernel, z, x, d, n)


# =========================================================================== #
# Reductions: dot product and max(|x|)
# =========================================================================== #
@cuda.jit
def _dot_partials_kernel(x, y, partials, n):
    """Block partial sums of ``x[i] * y[i]``.

    Each thread grid-stride accumulates into a register, the block tree-reduces
    in shared memory, and thread 0 writes the block's partial to
    ``partials[blockIdx.x]``.  ``partials`` has one entry per launched block.
    """
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    i = cuda.grid(1)
    sh = cuda.shared.array(_BLOCK, dtype=float64)

    acc = 0.0
    grid_stride = cuda.gridDim.x * bdim
    while i < n:
        acc += x[i] * y[i]
        i += grid_stride
    sh[tid] = acc
    cuda.syncthreads()

    # Inclusive tree reduction (bdim is a power of two).
    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            sh[tid] += sh[tid + step]
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        partials[cuda.blockIdx.x] = sh[0]


@cuda.jit
def _maxabs_partials_kernel(x, partials, n):
    """Block partial maxima of ``|x[i]|`` (same reduction shape as dot)."""
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    i = cuda.grid(1)
    sh = cuda.shared.array(_BLOCK, dtype=float64)

    acc = 0.0
    grid_stride = cuda.gridDim.x * bdim
    while i < n:
        a = x[i]
        if a < 0.0:
            a = -a
        if a > acc:
            acc = a
        i += grid_stride
    sh[tid] = acc
    cuda.syncthreads()

    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            a = sh[tid]
            b = sh[tid + step]
            sh[tid] = a if a > b else b
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        partials[cuda.blockIdx.x] = sh[0]


def _num_blocks(n: int) -> int:
    return (n + _BLOCK - 1) // _BLOCK


# Cache of device partial-sum buffers keyed by block count.  Reusing the same
# buffer across reduction calls avoids a ``cudaMalloc``/``cudaFree`` round-trip
# per call (the Numba pool already caches, but this also skips the Python-side
# allocation dispatch).  Sized for the largest problem; smaller reductions share
# the head of the buffer.
_PARTIALS_CACHE: dict[int, "object"] = {}


def _partials(nblk: int):
    buf = _PARTIALS_CACHE.get(nblk)
    if buf is None:
        buf = cuda.device_array(nblk, dtype=np.float64)
        _PARTIALS_CACHE[nblk] = buf
    return buf


@cuda.jit
def _sum_partials_kernel(partials, out, nblk):
    """Reduce the ``nblk`` block partials to a single scalar on device.

    One block of ``_BLOCK`` threads tree-reduces the (small) partials array in
    shared memory and writes the result to ``out[0]``.  Keeping the final sum
    on the device means only one float crosses back to the host per reduction.
    """
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    sh = cuda.shared.array(_BLOCK, dtype=float64)
    i = tid
    acc = 0.0
    while i < nblk:
        acc += partials[i]
        i += bdim
    sh[tid] = acc
    cuda.syncthreads()
    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            sh[tid] += sh[tid + step]
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        out[0] = sh[0]


@cuda.jit
def _max_partials_kernel(partials, out, nblk):
    """Reduce the block partials to their maximum on device (see above)."""
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    sh = cuda.shared.array(_BLOCK, dtype=float64)
    i = tid
    acc = 0.0
    while i < nblk:
        a = partials[i]
        if a < 0.0:
            a = -a
        if a > acc:
            acc = a
        i += bdim
    sh[tid] = acc
    cuda.syncthreads()
    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            a = sh[tid]
            b = sh[tid + step]
            sh[tid] = a if a > b else b
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        out[0] = sh[0]


# Single-element device scratch for the final scalar of each reduction.
_REDUCE_OUT = {"sum": None, "max": None, "sum2": None}


def _reduce_out(kind: str):
    if _REDUCE_OUT[kind] is None:
        _REDUCE_OUT[kind] = cuda.device_array(
            2 if kind == "sum2" else 1, dtype=np.float64)
    return _REDUCE_OUT[kind]


def dot(x, y, n: int) -> float:
    """Return ``sum(x[i] * y[i])`` over device arrays (host scalar)."""
    nblk = _num_blocks(n)
    partials = _partials(nblk)
    _dot_partials_kernel[nblk, _BLOCK](x, y, partials, n)
    out = _reduce_out("sum")
    _sum_partials_kernel[1, _BLOCK](partials, out, nblk)
    return float(out.copy_to_host()[0])


# Second partials buffer for the fused two-dot reduction (see ``dot2``).
_PARTIALS_CACHE2: dict[int, "object"] = {}


def _partials2(nblk: int):
    buf = _PARTIALS_CACHE2.get(nblk)
    if buf is None:
        buf = cuda.device_array(nblk, dtype=np.float64)
        _PARTIALS_CACHE2[nblk] = buf
    return buf


@cuda.jit
def _sum2_partials_kernel(p1, p2, out, nblk):
    """Reduce two partials arrays to ``out[0]`` and ``out[1]`` in one block."""
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    sh = cuda.shared.array(_BLOCK, dtype=float64)
    # first sum
    i = tid
    acc = 0.0
    while i < nblk:
        acc += p1[i]
        i += bdim
    sh[tid] = acc
    cuda.syncthreads()
    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            sh[tid] += sh[tid + step]
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        out[0] = sh[0]
    # second sum
    i = tid
    acc = 0.0
    while i < nblk:
        acc += p2[i]
        i += bdim
    sh[tid] = acc
    cuda.syncthreads()
    step = 1
    while step < bdim:
        if tid % (2 * step) == 0:
            sh[tid] += sh[tid + step]
        cuda.syncthreads()
        step *= 2
    if tid == 0:
        out[1] = sh[0]


def dot2(x, y, x2, y2, n: int) -> tuple[float, float]:
    """Return ``(x.y, x2.y2)`` with a *single* host sync (two dots for the price
    of one round-trip).  Used by the BiCGSTAB ``omega`` step, which needs both
    ``(t, s)`` and ``(t, t)`` at the same point.
    """
    nblk = _num_blocks(n)
    p1 = _partials(nblk)
    p2 = _partials2(nblk)
    _dot_partials_kernel[nblk, _BLOCK](x, y, p1, n)
    _dot_partials_kernel[nblk, _BLOCK](x2, y2, p2, n)
    out = _reduce_out("sum2")
    _sum2_partials_kernel[1, _BLOCK](p1, p2, out, nblk)
    o = out.copy_to_host()
    return float(o[0]), float(o[1])


def max_abs(x, n: int) -> float:
    """Return ``max |x[i]|`` over a device array (host scalar)."""
    nblk = _num_blocks(n)
    partials = _partials(nblk)
    _maxabs_partials_kernel[nblk, _BLOCK](x, partials, n)
    out = _reduce_out("max")
    _max_partials_kernel[1, _BLOCK](partials, out, nblk)
    return float(out.copy_to_host()[0])


def norm2(x, n: int) -> float:
    """Return the Euclidean norm ``sqrt(sum x[i]^2)`` (host scalar)."""
    return float(np.sqrt(dot(x, x, n)))