# CFDPyGPU — GPU Acceleration Performance Report

> **Scope.** This report covers the GPU work in [`CFDPYGPU/`](README.md).  The
> framework itself is documented in [`README.md`](README.md); the repository
> overview is in the root [README](../README.md).

> **Methodology reminder (from the spec):** *"Never rewrite the entire solver
> at once. Proceed incrementally. At each step: Profile → Select next hotspot
> → Implement GPU → Validate accuracy → Benchmark CPU vs GPU → Only after
> successful validation continue to the next hotspot."*
>
> This report is the **Profile** step of that loop, plus the validation and
> benchmark of the first two GPU increments (the BLAS-1 / sparse-matvec kernels
> and the GPU BiCGSTAB pressure-Poisson solver). It is written *before* the
> production solvers are touched, exactly as requested.

---

## 1. Profiling methodology

`profile_hotspots.py` drives the real `Simulation` under `cProfile` with **all
I/O disabled** (`save_csv/hdf5/tecplot/png/mp4 = False`, `verbose = False`) so
only numerical work is measured. Three warm-up steps run before the profiler is
armed, so JIT compilation, lazy matrix builds and cache fills are excluded. The
script reports both `cumulative` (call-tree cost) and `tottime` (self cost)
rankings, and the per-step wall time.

Two regimes are profiled, because they have **different hotspots** and therefore
different GPU strategies:

| Case | File | Regime | Grid |
|------|------|--------|------|
| Cylinder flow | `examples/cylinder_flow/config.json` | single-phase, fixed Δt, constant ρ | 400×160×1 |
| Dam break | `examples/dam_break_2D/config.json` | two-phase VOF, variable ρ, adaptive Δt | 80×50×1 |

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9, 6 GB,
CUDA Runtime 12.6 / Driver 13.2. CPU baseline is the same machine (NumPy/SciPy).

---

## 2. Hotspot ranking

### 2.1 Single-phase (cylinder flow) — 1124.97 ms/step

| Rank | Function | tottime | % of step | Role |
|------|----------|---------|-----------|------|
| 1 | `scipy…_superlu.gstrf` (ILU factorization) | 30.88 s | **91.5 %** | build ILU(0) of the Poisson operator |
| 2 | `SuperLU.solve` (triangular solves) | 1.50 s | 4.4 % | apply ILU inside BiCGSTAB |
| 3 | `linear_solver.solve` (cumulative) | 33.10 s | 98.1 % | the whole linear-solve path |
| 4 | `bicgstab` (the Krylov loop itself) | 1.01 s | 3.0 % | matvecs + dot products |
| 5 | `face_interpolate` / `cell_gradient` / `divergence` | ~0.5 s | ~1.5 % | stencil array kernels |

**Reading.** In the single-phase regime the matrix is fixed, so ILU(0) is a
strong preconditioner and BiCGSTAB converges in a handful of iterations — the
Krylov loop is only ~3 % of the step. The cost is the **ILU factorization**
(`gstrf`, 91.5 %), rebuilt for each of the four linear systems per step
(pressure + momentum-x/y + energy). The sparse matvec (`csr_matvec`, 0.045 s
total) is negligible here.

### 2.2 Two-phase VOF (dam break) — 83.25 ms/step

| Rank | Function | tottime/cumtime | % of step | Role |
|------|----------|-----------------|-----------|------|
| 1 | `scipy…iterative.bicgstab` (cumulative) | 1.626 s | **97.7 %** | the Krylov solve |
| 2 | `scipy…csr_matvec` (self) | 0.373 s | 22.4 % | sparse matvec (inside BiCGSTAB) |
| 3 | `LinearOperator.matvec` plumbing | ~1.0 s | ~60 % | SciPy dispatch overhead |
| — | `gstrf` (ILU factorization) | **absent** | 0 % | not used in this regime |

**Matvec count:** 40 120 matvecs / 20 steps = **2006 matvecs/step** (≈2 matvecs
per BiCGSTAB iteration → ~1000 iterations/step). No ILU appears in the top 25:
the density field changes every step, so an ILU factorization would have to be
rebuilt each step to no benefit, and the production path runs unpreconditioned
BiCGSTAB. The cost is **iteration count × matvec**, not factorization.

### 2.3 The dichotomy that defines the GPU strategy

| | Single-phase | VOF / adaptive-Δt |
|---|---|---|
| **Hotspot** | ILU factorization (91.5 %) | BiCGSTAB matvec loop (97.7 %) |
| **Cost driver** | SuperLU `gstrf` | ~2000 matvecs/step |
| **CPU exploit** | ILU cached → few iterations | none (matrix changes each step) |
| **What a GPU must beat** | the factorization + few ILU applies | ~2000 cheap matvecs |
| **GPU lever** | replace ILU with O(1)-iteration multigrid | faster matvec + fewer iterations |

The same physical operator (5-point Poisson on a structured collocated mesh)
dominates both regimes, but the *bottleneck shifts* between them. A GPU port
that only speeds up the matvec helps VOF and **does nothing for single-phase**
(where the matvec is already <0.2 % of the step). Conversely, removing the ILU
factorization helps single-phase enormously but, without a better preconditioner,
leaves VOF at ~2000 iterations/step.

---

## 3. The partial-port trap

A naive GPU port of just the matvec is **net-negative** for this codebase:

* **Transfer overhead dominates at small N.** Each host→device copy of the CSR
  triple + RHS, plus the device→host copy of the solution, costs ~30–60 µs. At
  N = 3600 the CPU matvec is ~15 µs — the transfers alone are 2–4× the work
  being offloaded.
* **The single-phase hotspot is not the matvec.** Speeding up 0.2 % of the step
  by 8× saves ~0.18 % of the step, far less than the transfer + launch overhead
  it introduces.
* **SciPy `LinearOperator` dispatch is half the VOF cost.** Even a perfect GPU
  matvec leaves the ~60 % spent in SciPy's `matvec`/`matmat`/`isscalar` plumbing
  untouched unless the *entire* Krylov loop is moved to the device.

**Conclusion:** fields must live on the GPU for the whole step, and the Krylov
driver (not just the matvec) must run on the device. This is why the
architecture keeps fields GPU-resident and reuses allocations
(`gpu/backend.py` memory pool, `gpu/kernels.py` partials cache,
`gpu/linear.py` workspace allocated once per solver instance).

---

## 4. GPU increments delivered and validated

### 4.1 Increment A — BLAS-1 reductions + sparse CSR matvec (`gpu/kernels.py`)

* **Why selected:** the sparse matvec is the workhorse of every Krylov
  iteration and the single largest `tottime` item in the VOF profile
  (`csr_matvec`, 22 %). Reductions (dot, max-abs, norm2) are needed to drive any
  Krylov or residual-norm loop on the device.
* **CUDA design:** one thread per row, grid-stride accumulation in a register
  for the matvec (race-free, no atomics); two-level shared-memory tree
  reduction (`BLOCK = 256`, power of two) for dot / max-abs, writing one partial
  per block, then a single-block final reduce so only **one float crosses
  D2H** per reduction. `dot2` fuses two dot products into one host sync (the
  BiCGSTAB ω-step needs `(t,s)` and `(t,t)` together). Partial-sum buffers and
  the final-scalar scratch are cached and reused across calls.
* **Validation (`python -m gpu.validate_kernels`):** every kernel matches its
  NumPy reference to round-off — matvec **exact**, reductions ~1e-15 relative.

| Kernel | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| `matvec_csr` N=32000 | — | — | **1.92×** |
| `matvec_csr` N=64000 | — | — | **3.59×** |
| `matvec_csr` N=128000 | — | — | **7.95×** |
| `dot` n=200000 | — | — | 1.17× |

*(The matvec GPU time is essentially flat in N (~35 µs), so speedup grows with
problem size; the dot reduction is sync-bound at this size and barely wins —
expected, since one dot is one tree-reduce + one D2H.)*

### 4.2 Increment B — GPU BiCGSTAB pressure-Poisson solver (`gpu/linear.py`)

* **Why selected:** the Krylov loop is 97.7 % of the VOF step and 3 % + the
  ILU-apply tail of the single-phase step. Moving the *whole* loop to the device
  (matvec + BLAS-1 + convergence test) is the only way to escape the SciPy
  `LinearOperator` dispatch overhead identified in §2.2.
* **CUDA design:** preconditioned BiCGSTAB (van der Vorst) with **Jacobi
  (diagonal) preconditioning** — the only preconditioner that is cheap on a GPU
  (a pointwise divide, `div_pointwise`) and free of the sequential triangular
  solves that make ILU awkward on a GPU. Workspace is allocated **once** per
  solver instance and reused across solves; the Jacobi inverse-diagonal is
  cached per matrix identity and rebuilt only when the matrix changes. The
  convergence test uses the cheap recurrence residual but **verifies with the
  true residual** `b − A x` (one extra matvec) on exit, to catch the BiCGSTAB
  phantom-convergence stop (recurrence residual ≪ true residual). The
  recurrence-residual norm is only evaluated every 4 iterations
  (`check_every=4`) to cut sync overhead, with no change to the converged
  solution.
* **Validation (`python -m gpu.validate_linear`):** solved on the **real
  production operator** built by `PressureSolver._matrix` (pure-Neumann,
  mean-projected RHS, mean-subtracted solution), compared against CPU+ILU and
  CPU-no-ILU.

| Grid | N | GPU iters | conv | true_res | dp GPU vs CPU+ILU (L2 / relL∞) |
|------|---|-----------|------|----------|--------------------------------|
| 60×60 | 3600 | 168 | ✓ | 8.6e-08 | 1.12e-06 / 1.68e-07 |
| 200×160 | 32000 | 408 | ✓ | 7.8e-08 | 1.01e-06 / 1.40e-07 |
| 400×160 | 64000 | 676 | ✓ | 2.9e-08 | 4.51e-07 / 7.18e-08 |

**Accuracy verdict:** the GPU solution agrees with the CPU solution to
L2 ~1e-6, relL∞ ~1e-7 — within the solver tolerance (1e-7) amplified by the
condition number, exactly as expected for two iterative solvers that stop on a
relative residual. ✅ Numerical consistency holds.

### 4.3 Benchmark — honest result

| Grid | N | CPU+ILU | CPU no-ILU | GPU | GPU vs CPU+ILU | GPU vs CPU no-ILU |
|------|---|---------|------------|-----|----------------|-------------------|
| 60×60 | 3600 | 2.95 ms | 4.78 ms | 113.4 ms | 0.03× | 0.04× |
| 200×160 | 32000 | 48.7 ms | 254.6 ms | 272.4 ms | 0.18× | 0.93× |
| 400×160 | 64000 | 108.7 ms | 567.3 ms | 437.8 ms | 0.25× | **1.30×** |

**Interpretation (no spin):**

* **vs CPU+ILU the GPU is slower (0.03–0.25×).** ILU is a much stronger
  preconditioner than Jacobi, so the CPU converges in few iterations and the
  cached factorization wins. This is the single-phase regime of §2.1.
* **vs CPU-no-ILU the GPU wins at production size (1.30× at 400×160, and the
  advantage grows with N).** This is the VOF / adaptive-Δt regime of §2.2, where
  the matrix changes every step and ILU would be rebuilt to no benefit. Here the
  GPU's flat-in-N matvec + on-device Krylov loop beat ~2000 CPU matvecs plus
  SciPy dispatch overhead.
* **The bottleneck is the iteration count, not the per-iteration cost.** At
  N = 64000 the GPU does 676 iterations × ~0.65 ms/iter; the matvec itself is
  ~35 µs, so the dots + syncs dominate each iteration. Cutting iterations is the
  lever, which is exactly what a better preconditioner delivers.

**Status:** Increment B is numerically correct and is a net win **only in the
VOF regime at large N**. It is **not yet** a net win over CPU+ILU for
single-phase. It is deliberately **not wired into the production
`PressureSolver`** until the preconditioner is strong enough to win in both
regimes — per the incremental methodology, an increment that regresses the
production solver is not promoted.

---

## 5. Incremental GPU roadmap

Ordered by expected impact, each step is Profile → Implement → Validate →
Benchmark → promote-only-on-success, as the spec requires.

### Step 1 (NEXT) — GPU geometric multigrid preconditioner
The single change that turns both regimes into net wins. For the structured
5-point Poisson on the collocated mesh, geometric multigrid (V-cycle with
red-black Gauss-Seidel smoothing, full-weighting restriction, bilinear
prolongation, coarsest-level direct solve) converges in **O(1) iterations**
independent of N — eliminating both the 91.5 % ILU factorization (single-phase)
and the ~2000-iteration Jacobi loop (VOF). All operators are structured-stencil
(not CSR), so they map to 2-D CUDA grids with shared-memory halos and need **no
host↔device transfer** during the cycle. Expected: GPU Poisson solve net-faster
than CPU+ILU at all N, and ≫100× fewer iterations than the current GPU Jacobi
path. This is the preconditioner `GPUBiCGSTAB` will call instead of
`div_pointwise`.

### Step 2 — Wire the GPU Poisson solve into the production `PressureSolver`
Only after Step 1 makes the GPU solve a net win in both regimes. Keep the CPU
path as the automatic fallback (`use_gpu` config flag, already added in
`config_loader.py`). Validate a full `Simulation.step()` CPU-vs-GPU on the
cylinder and dam-break cases: L2/relL∞ of pressure, velocity, and the per-step
max divergence / max CFL, plus residual-history agreement.

### Step 3 — Keep the fields GPU-resident across the whole step
Move the velocity/pressure/temperature/density/phase-fraction fields and mesh
coords to the device once at `Simulation.initialize()` and keep them there for
the run. This is what makes the VOF 1.30× (measured at N=64000) become the
*whole-step* speedup rather than just the Poisson-isolated number, by removing
the per-solve host↔device copies that today bracket each linear solve.

### Step 4 — GPU stencil operators (gradient / divergence / face-interpolate)
Once the fields are resident (Step 3), port the `numerics/` stencil kernels that
§2.1 showed at ~1.5 % of the single-phase step. They are pure 2-D stencils —
natural 2-D CUDA grids with shared-memory halos — and, more importantly, they
remove the last D2H/H2D boundaries so the entire `step()` runs without a single
host transfer.

### Step 5 — GPU momentum / energy diffusion solves
The momentum and energy diffusion terms (the other three of the four linear
systems per step flagged in §2.1) are the same class of structured symmetric
operator as the Poisson solve, so they reuse the Step-1 multigrid preconditioner
and the Step-3 resident fields.

### Future — multi-GPU / domain decomposition
The backend was designed for this: `init_backend(device_index=local_rank)` (see
`gpu/backend.py`) selects a per-rank GPU; the kernels are one-thread-per-row /
per-cell with no inter-block coupling, so a halo-exchange layer can be added
later without rewriting the kernels. Not started.

---

## 6. Per-modification deliverable summary (template from the spec)

For Increment B (GPU BiCGSTAB), the six required artifacts:

1. **Why this kernel:** the Krylov loop is the 97.7 % VOF hotspot and the tail
   of the 98 % single-phase linear-solve path; only a device-resident driver
   escapes SciPy `LinearOperator` dispatch overhead.
2. **Original code:** `solver/linear_solver.py` — SciPy `bicgstab` with optional
   ILU(0) (`scipy.sparse.linalg.spilu` / SuperLU `gstrf`).
3. **CUDA implementation:** `gpu/linear.py` — `GPUBiCGSTAB`, preconditioned
   BiCGSTAB driven by `gpu/kernels.py` (matvec + BLAS-1 + reductions), workspace
   allocated once, Jacobi inverse-diagonal cached per matrix identity,
   true-residual convergence guard, `check_every=4` sync throttling.
4. **Validation:** §4.2 — agrees with CPU to L2 ~1e-6 / relL∞ ~1e-7 on the real
   production operator; true residual < tol in every case.
5. **Expected speedup:** 1.30× vs CPU-no-ILU at N=64000 (measured), growing with
   N; **not yet** a win vs CPU+ILU (0.25×) — deferred to Step 1 (multigrid).
6. **Future optimizations:** (a) geometric multigrid preconditioner (Step 1) to
   cut iterations from O(N) to O(1); (b) fuse the `rhat·r` and `rhat·v` dot
   products; (c) defer the convergence test to every K iterations and use a
   Chebyshev-accelerated Jacobi smoother; (d) keep fields resident (Step 3) to
   remove the per-solve host↔device copies that currently bracket `solve`.

---

## 7. Reproducing the numbers

```bash
# Profile (the §2 hotspot rankings)
python profile_hotspots.py examples/cylinder_flow/config.json 30
python profile_hotspots.py examples/dam_break_2D/config.json 20

# Validate + benchmark the GPU kernels (§4.1)
python -m gpu.validate_kernels

# Validate + benchmark the GPU BiCGSTAB on the real Poisson operator (§4.2-4.3)
python -m gpu.validate_linear

# Print the hardware report (§1)
python -c "from gpu import print_hardware_report; print_hardware_report()"
```

All GPU modules degrade gracefully: if no NVIDIA GPU is present, `init_backend`
falls back to the NumPy backend and the framework runs the original CPU path
with identical numerics.
