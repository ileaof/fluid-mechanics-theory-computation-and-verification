"""GPU acceleration package for CFDPy.

The package is organised in layers, each depending only on the one below it:

* :mod:`gpu.hardware`  -- detection + the startup hardware report (no state).
* :mod:`gpu.backend`   -- the array/device backend (auto-detect, CPU fallback,
  memory-pool reuse, future multi-GPU/MPI hook).
* :mod:`gpu.kernels`   -- low-level ``@cuda.jit`` kernels (sparse matvec,
  BLAS-1 reductions) used by the GPU linear solver.
* :mod:`gpu.linear`    -- GPU-resident Krylov solvers (BiCGSTAB) for the
  pressure-Poisson system.

Importing the package never touches the GPU: detection is lazy.  The framework
imports :mod:`gpu.hardware` at startup to print the report, and the solver
modules import the backend / kernels only when a GPU is actually in use, so a
CPU-only machine pays no import cost beyond the report.
"""

from __future__ import annotations

from .hardware import (
    GPUInfo,
    detect_gpu,
    gpu_available,
    hardware_report,
    print_hardware_report,
)
from .backend import (
    Backend,
    CUDABackend,
    NumPyBackend,
    init_backend,
    get_backend,
    reset_backend,
    is_gpu_active,
)
from .linear import GPUBiCGSTAB

__all__ = [
    "GPUInfo",
    "detect_gpu",
    "gpu_available",
    "hardware_report",
    "print_hardware_report",
    "Backend",
    "CUDABackend",
    "NumPyBackend",
    "init_backend",
    "get_backend",
    "reset_backend",
    "is_gpu_active",
    "GPUBiCGSTAB",
]