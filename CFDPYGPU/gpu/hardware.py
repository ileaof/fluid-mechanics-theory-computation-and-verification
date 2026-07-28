"""GPU hardware detection and the CFDPy hardware report.

This module is the *single source of truth* for "is there a usable NVIDIA GPU
on this machine?".  It is imported lazily and degrades gracefully: when no CUDA
runtime, no Numba CUDA support, or no NVIDIA device is present, it reports
``Execution Device : CPU`` and the rest of the framework runs the original
pure-NumPy/SciPy code path unchanged.

Detection is deliberately conservative and side-effect free: :func:`detect_gpu`
only *reads* device attributes; it never allocates device memory or creates a
context that would disturb a later run.  The (limited) CUDA driver/runtime
version query goes through the bundled ``cudart64`` DLL when it can be located;
both versions are optional -- a missing DLL simply leaves the corresponding
field blank rather than failing detection.

The public surface is small and stable:

* :class:`GPUInfo`   -- dataclass holding every detected attribute (or ``None``).
* :func:`detect_gpu` -- build a :class:`GPUInfo` (cached).
* :func:`hardware_report` -- the formatted multi-line string printed at startup.
* :func:`print_hardware_report` -- print it.
* :func:`gpu_available` -- convenience boolean.

Design note (multi-GPU / MPI).  ``GPUInfo`` describes *one* device; the field
``device_index`` records which.  The future multi-GPU / domain-decomposition
extension will instantiate one backend per rank and select the device by
``device_index = local_rank`` (set via ``CUDA_VISIBLE_DEVICES`` or an explicit
``mpi_rank`` argument to :func:`detect_gpu`).  Nothing here assumes a single
GPU; it merely reports the one that would be used by this process.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

# Numba is an optional dependency of the framework; its CUDA sub-module is the
# preferred GPU runtime here.  Importing it must never raise.
try:
    from numba import cuda as _cuda
    _HAS_NUMBA_CUDA = True
except Exception:                       # pragma: no cover - environment dependent
    _cuda = None
    _HAS_NUMBA_CUDA = False


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class GPUInfo:
    """Detected attributes of the CUDA device available to this process.

    Every field is ``None`` when the information could not be obtained, so a
    caller can always format ``f"{info.name or 'unknown'}"`` without special
    cases.  ``available`` is the master switch: ``False`` means "run on CPU".
    """

    available: bool = False
    device_index: int = 0
    name: str | None = None
    compute_capability: tuple[int, int] | None = None
    multiprocessor_count: int | None = None          # number of SMs
    total_memory_mb: float | None = None              # total global memory
    warp_size: int | None = None
    max_threads_per_block: int | None = None
    max_shared_memory_per_block: int | None = None
    max_registers_per_block: int | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    # Reason when detection declined (for the report footer / logs).
    reason: str = ""


# --------------------------------------------------------------------------- #
# CUDA driver / runtime version via the bundled cudart DLL
# --------------------------------------------------------------------------- #
def _find_cudart() -> str | None:
    """Locate a ``cudart64_*.dll`` shipped with the environment.

    Searched in priority order: the ``nvidia/cuda_runtime/bin`` wheel layout
    (used by the ``nvidia-*`` pip packages), then the standard DLL search path.
    Returns the first loadable path or ``None``.
    """
    # 1. pip-installed NVIDIA CUDA runtime wheel layout.
    try:
        base = os.path.dirname(os.path.dirname(sys.executable))
        cands = glob.glob(
            os.path.join(base, "**", "nvidia", "cuda_runtime", "bin",
                         "cudart64*.dll"),
            recursive=True,
        )
        for c in cands:
            if os.path.isfile(c):
                return c
    except Exception:
        pass
    # 2. site-packages of the running interpreter (venv pointing at a base).
    try:
        for p in sys.path:
            if not p or not os.path.isdir(p):
                continue
            cands = glob.glob(
                os.path.join(p, "nvidia", "cuda_runtime", "bin",
                             "cudart64*.dll"))
            for c in cands:
                if os.path.isfile(c):
                    return c
    except Exception:
        pass
    return None


def _cuda_versions() -> tuple[str | None, str | None]:
    """Return ``(driver_version, runtime_version)`` as ``"major.minor"`` strings.

    Both are ``None`` if the cudart DLL could not be loaded or the entry points
    are missing.  ``cudaDriverGetVersion`` returns the *driver* API version
    supported by the loaded driver (independent of the DLL's own runtime), and
    ``cudaRuntimeGetVersion`` returns the runtime version of the DLL.
    """
    dll = _find_cudart()
    if dll is None:
        return None, None
    try:
        cudart = ctypes.CDLL(dll)
    except OSError:
        return None, None

    def fmt(raw: int) -> str:
        return f"{raw // 1000}.{(raw % 1000) // 10}"

    drv = rtl = None
    try:
        v = ctypes.c_int(0)
        if cudart.cudaDriverGetVersion(ctypes.byref(v)) == 0:
            drv = fmt(v.value)
    except Exception:
        pass
    try:
        v = ctypes.c_int(0)
        if cudart.cudaRuntimeGetVersion(ctypes.byref(v)) == 0:
            rtl = fmt(v.value)
    except Exception:
        pass
    return drv, rtl


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def detect_gpu(device_index: int = 0) -> GPUInfo:
    """Detect the CUDA device at ``device_index`` and return a :class:`GPUInfo`.

    Cached: the first call probes the hardware; subsequent calls return the same
    object.  Pass ``device_index`` to select a specific device in a multi-GPU
    node (used by the future MPI/domain-decomposition path: one rank per GPU).
    """
    if not _HAS_NUMBA_CUDA:
        return GPUInfo(available=False,
                       reason="Numba CUDA not available (numba missing or "
                              "built without CUDA support).")
    try:
        if not _cuda.is_available():
            return GPUInfo(available=False,
                           reason="numba.cuda.is_available() is False "
                                  "(no CUDA-capable device or no CUDA toolkit).")
    except Exception as exc:            # pragma: no cover - defensive
        return GPUInfo(available=False, reason=f"CUDA probe failed: {exc}")

    try:
        with _cuda.gpus[device_index]:
            dev = _cuda.get_current_device()
    except Exception as exc:             # pragma: no cover - defensive
        return GPUInfo(available=False, reason=f"Device {device_index} "
                                               f"unavailable: {exc}")

    def attr(name: str):
        try:
            return getattr(dev, name)
        except Exception:
            return None

    name = attr("name")
    if isinstance(name, bytes):
        name = name.decode("ascii", "ignore")
    cc = attr("compute_capability")          # (major, minor)
    sm = attr("MULTIPROCESSOR_COUNT")
    warp = attr("WARP_SIZE")
    maxthr = attr("MAX_THREADS_PER_BLOCK")
    shmem = attr("MAX_SHARED_MEMORY_PER_BLOCK")
    regs = attr("MAX_REGISTERS_PER_BLOCK")

    # Total global memory.  Numba's device object does not expose a portable
    # ``TOTAL_MEMORY`` attribute across builds, so query the *context* memory
    # info, which returns ``(free_bytes, total_bytes)``.  This is the same
    # figure nvidia-smi reports.  Falls back to the legacy device attribute if
    # present.
    total_mb = None
    try:
        ctx = _cuda.current_context()
        _free, total = ctx.get_memory_info()
        if total and total > 0:
            total_mb = float(total) / (1024.0 * 1024.0)
    except Exception:
        pass
    if total_mb is None:
        for cand in ("TOTAL_MEMORY", "total_memory"):
            v = attr(cand)
            if isinstance(v, (int, float)) and v > 0:
                total_mb = float(v) / (1024.0 * 1024.0)
                break

    drv, rtl = _cuda_versions()

    return GPUInfo(
        available=True,
        device_index=device_index,
        name=name,
        compute_capability=cc,
        multiprocessor_count=sm,
        total_memory_mb=total_mb,
        warp_size=warp,
        max_threads_per_block=maxthr,
        max_shared_memory_per_block=shmem,
        max_registers_per_block=regs,
        cuda_runtime_version=rtl,
        cuda_driver_version=drv,
    )


def gpu_available() -> bool:
    """Whether a CUDA GPU was detected by :func:`detect_gpu`."""
    return detect_gpu().available


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #
def _fmt_cc(cc) -> str:
    return "unknown" if cc is None else f"{cc[0]}.{cc[1]}"


def _fmt_mem(mb: float | None) -> str:
    if mb is None:
        return "unknown"
    gb = mb / 1024.0
    if gb >= 1.0:
        return f"{gb:.0f} GB"
    return f"{mb:.0f} MB"


def hardware_report(info: GPUInfo | None = None) -> str:
    """Return the formatted CFDPy hardware report (no trailing newline).

    The layout matches the project specification: a banner, the execution
    device, and -- on GPU -- the device name, compute capability, memory and
    the CUDA runtime / driver versions.  On CPU only the device line is shown.
    """
    if info is None:
        info = detect_gpu()
    bar = "-" * 49
    lines = [bar, "CFDPy Hardware Report", bar]
    if info.available:
        lines.append(f"Execution Device : NVIDIA GPU")
        lines.append(f"GPU             : {info.name or 'unknown'}")
        lines.append(f"Compute Cap.    : {_fmt_cc(info.compute_capability)}")
        lines.append(f"Memory          : {_fmt_mem(info.total_memory_mb)}")
        lines.append(f"CUDA Runtime    : {info.cuda_runtime_version or 'unknown'}")
        lines.append(f"CUDA Driver     : {info.cuda_driver_version or 'unknown'}")
    else:
        lines.append("Execution Device : CPU")
        if info.reason:
            lines.append(f"Note            : {info.reason}")
    lines.append(bar)
    return "\n".join(lines)


def print_hardware_report(info: GPUInfo | None = None) -> None:
    """Print :func:`hardware_report` to stdout."""
    print(hardware_report(info))


if __name__ == "__main__":
    print_hardware_report()