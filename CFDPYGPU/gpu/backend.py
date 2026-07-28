"""Array / device backend for CFDPy (auto-detect, CPU fallback).

The backend is the *only* module that decides "CPU or GPU" for the numerical
core and the only one that imports ``numba.cuda`` for allocation.  Everything
above it (:mod:`gpu.kernels`, :mod:`gpu.linear`, and eventually the solver
modules) asks the backend for arrays and transfers, so the CPU/GPU choice is
made in one place and the surrounding code is device-agnostic.

Two concrete backends implement the same interface:

* :class:`NumPyBackend` -- the original pure-NumPy path (also the fallback).
* :class:`CUDABackend`  -- Numba-CUDA device arrays; allocations live in the
  device memory pool that Numba manages (a per-context dealloc cache that
  reuses freed allocations), so repeated ``zeros``/``empty`` of the same shape
  do not round-trip through ``cudaMalloc``/``cudaFree``.

Selection rule
--------------
``get_backend()`` returns a CUDA backend when *all* of:

1. ``cfg.use_gpu`` is True (the config switch; default True),
2. :func:`gpu.hardware.gpu_available` is True,

hold, otherwise the NumPy backend.  A process-wide singleton is cached after
the first call; tests can reset it with :func:`reset_backend`.

Multi-GPU / MPI extension point
-------------------------------
The backend carries a ``device_index`` and activates its context on first use.
The future multi-GPU / domain-decomposition path will call
``init_backend(device_index=local_rank)`` once per MPI rank so each rank owns
one GPU; the kernels themselves are rank-local and need no change.  Nothing
here assumes a single GPU.
"""

from __future__ import annotations

import numpy as np

from .hardware import detect_gpu, gpu_available

# Numba CUDA is imported lazily inside CUDABackend so a CPU-only machine never
# pays the import cost (and never trips a missing-CUDA-toolkit error).


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #
class Backend:
    """Common interface for the NumPy and CUDA backends.

    The methods are intentionally a small subset of the NumPy / CuPy surface --
    just what the numerical kernels need.  Adding a method here is the only
    change required to give both backends a new capability.
    """

    name: str = "numpy"
    enabled: bool = False                 # True only for the CUDA backend
    device_index: int = 0

    # -- array creation ---------------------------------------------------- #
    def asarray(self, a, dtype=None):
        """Return the array on this backend's device (host for NumPy).

        ``dtype=None`` preserves the input dtype (important for integer index
        arrays such as the CSR ``indices``/``indptr``); pass an explicit dtype
        to coerce.  Python scalars / lists default to float64.
        """
        raise NotImplementedError

    def to_host(self, a):
        """Return a NumPy (host) copy of ``a``."""
        raise NotImplementedError

    def zeros(self, shape, dtype=np.float64):
        raise NotImplementedError

    def empty(self, shape, dtype=np.float64):
        raise NotImplementedError

    def zeros_like(self, a):
        raise NotImplementedError

    # -- introspection ----------------------------------------------------- #
    def is_device_array(self, a) -> bool:
        return False

    def synchronize(self) -> None:
        """Block until all queued device work is done (no-op on CPU)."""
        return None


class NumPyBackend(Backend):
    """The original CPU path: plain NumPy arrays, no transfers."""

    name = "numpy"
    enabled = False

    def asarray(self, a, dtype=None):
        if dtype is None:
            return np.asarray(a)
        return np.asarray(a, dtype=dtype)

    def to_host(self, a):
        return np.asarray(a)

    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float64):
        return np.empty(shape, dtype=dtype)

    def zeros_like(self, a):
        return np.zeros_like(np.asarray(a))

    def is_device_array(self, a) -> bool:
        return False


class CUDABackend(Backend):
    """Numba-CUDA device-array backend.

    Device arrays are ``numba.cuda.ndarray``-like objects (created by
    :func:`numba.cuda.device_array` / ``device_array_like``).  They live in the
    per-context memory pool Numba manages, so reused shapes avoid repeated
    driver allocations.  Host<->device copies are explicit through
    :meth:`asarray` (host->device) and :meth:`to_host` (device->host); the
    numerical kernels never implicit-copy.
    """

    name = "cuda"
    enabled = True

    def __init__(self, device_index: int = 0) -> None:
        from numba import cuda
        self._cuda = cuda
        self.device_index = device_index
        # Activate this device's context lazily on first use; the context is
        # retained for the process (Numba keeps a per-device primary context).
        self._dev = cuda.gpus[device_index]
        self._ctx_entered = False

    # -- internal: ensure the device context is active --------------------- #
    def _ensure_context(self):
        if not self._ctx_entered:
            self._dev.__enter__()           # type: ignore[attr-defined]
            self._ctx_entered = True

    # -- array creation ---------------------------------------------------- #
    def asarray(self, a, dtype=None):
        self._ensure_context()
        cuda = self._cuda
        if dtype is None:
            arr = np.asarray(a)
            if arr.dtype == object:
                arr = np.asarray(a, dtype=np.float64)
            arr = np.ascontiguousarray(arr)
        else:
            arr = np.ascontiguousarray(np.asarray(a), dtype=dtype)
        return cuda.to_device(arr)

    def to_host(self, a):
        if a is None:
            return None
        if isinstance(a, np.ndarray):
            return np.asarray(a)
        # device array -> host
        return a.copy_to_host()

    def zeros(self, shape, dtype=np.float64):
        self._ensure_context()
        return self._cuda.device_array(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float64):
        self._ensure_context()
        return self._cuda.device_array(shape, dtype=dtype)

    def zeros_like(self, a):
        self._ensure_context()
        if isinstance(a, np.ndarray):
            return self._cuda.device_array(a.shape, dtype=a.dtype)
        return self._cuda.device_array_like(a)

    # -- introspection ----------------------------------------------------- #
    def is_device_array(self, a) -> bool:
        return not isinstance(a, np.ndarray) and hasattr(a, "copy_to_host")

    def synchronize(self) -> None:
        self._ensure_context()
        self._cuda.synchronize()


# --------------------------------------------------------------------------- #
# Singleton selection
# --------------------------------------------------------------------------- #
_BACKEND: Backend | None = None


def init_backend(use_gpu: bool = True, device_index: int = 0) -> Backend:
    """Build and cache the process-wide backend.

    A CUDA backend is selected only when ``use_gpu`` is True *and* a CUDA GPU
    was detected by :mod:`gpu.hardware`; otherwise the NumPy backend is used
    and the framework runs the original CPU code path unchanged.
    """
    global _BACKEND
    if use_gpu and gpu_available():
        try:
            _BACKEND = CUDABackend(device_index=device_index)
            return _BACKEND
        except Exception:
            # Any failure to initialise the CUDA context falls back to CPU so
            # the simulation always runs.
            _BACKEND = NumPyBackend()
            return _BACKEND
    _BACKEND = NumPyBackend()
    return _BACKEND


def get_backend() -> Backend:
    """Return the cached backend, initialising it (GPU on) on first call."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = init_backend(use_gpu=True)
    return _BACKEND


def reset_backend() -> None:
    """Drop the cached backend (tests / forced re-detection)."""
    global _BACKEND
    if _BACKEND is not None and isinstance(_BACKEND, CUDABackend):
        try:
            _BACKEND.synchronize()
        except Exception:
            pass
    _BACKEND = None


def is_gpu_active() -> bool:
    """Whether the active backend is the CUDA one."""
    return get_backend().enabled