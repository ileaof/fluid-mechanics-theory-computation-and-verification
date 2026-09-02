# CFDPyGPU — a modular Finite-Volume CFD framework in Python

CFDPyGPU is an **educational and professional** Computational Fluid Dynamics
framework written from scratch in Python 3.11+ (no OpenFOAM, FEniCS or FiPy).
It solves the incompressible Navier–Stokes equations together with energy,
scalar transport and Volume-of-Fluid (VOF) free-surface models on a Cartesian
structured mesh, using the **Finite Volume Method** with a **projection
(fractional-step)** pressure-velocity coupling.

CFDPyGPU is the **GPU-accelerated superset** of [CFDPy](../CFDPY/README.md).
It reproduces the CPU framework (`mesh/`, `numerics/`, `physics/` and
`visualization/` are identical) and layers three things on top:

* a dedicated **[`gpu/`](#gpu-acceleration) package** that ports the
  performance-critical kernels to NVIDIA CUDA through Numba, with an automatic,
  numerics-identical fallback to the CPU path when no GPU is present;
* **force integration and immersed-boundary work** (`solver/forces.py`,
  `solver/ibm.py`, `solver/cut_cell.py`) developed for the cylinder benchmark;
* a fifth example, **[flow past a circular cylinder](#example-5--flow-past-a-circular-cylinder)**,
  with a Reynolds-sweep driver and a literature comparison table.

The same JSON case files run on either variant with no change.

The code is organised as a small, decoupled package where every subsystem
(mesh, numerics, physics, solvers, visualisation) can be read and modified in
isolation.  It is designed as a research base: the architecture is ready for
unstructured meshes, RANS/LES/DNS, MPI, AMR, additional multiphase /
compressible / radiation / phase-change / species models.

---

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start — running the examples](#quick-start--running-the-examples)
4. [GPU acceleration](#gpu-acceleration)
5. [How to change parameters](#how-to-change-parameters)
6. [How to create a new case](#how-to-create-a-new-case)
7. [Mathematical formulation](#mathematical-formulation)
8. [Algorithms and solver flowchart](#algorithms-and-solver-flowchart)
9. [Project organization](#project-organization)
10. [Extending the framework](#extending-the-framework)
11. [License](#license)

---

## Features

- **2D and 3D** Cartesian structured, collocated finite-volume mesh.
- **Incompressible flow** — continuity + Navier–Stokes.
- **Heat transfer** — energy equation with Dirichlet / heat-flux / adiabatic walls.
- **Natural convection** — Boussinesq approximation `ρ = ρ₀(1 − β(T−T₀))`
  with automatic gravity-driven buoyancy.
- **Forced convection** — prescribed inlets / outlets.
- **Free surfaces** — VOF transport of the volume fraction `α` with linear
  property blending and interface-normal reconstruction (water/air).
- **Scalar transport** — generic advection–diffusion (the temperature solver
  is a specialisation of this).
- **Gravity** — body force handled as an acceleration, density enters through
  the pressure projection.
- **Transient & steady** — implicit Euler and Crank–Nicolson time schemes,
  selectable from the config; steady cases are obtained by running transient
  to convergence.
- **Convection schemes** — upwind, central differencing, QUICK and TVD
  (vanleer / minmod / superbee / beam-warming / osher).
- **Pressure-velocity coupling** — incremental projection method (Chorin);
  the architecture is the natural extension point for SIMPLE.
- **Linear solvers** — CG, BiCGSTAB, GMRES with optional ILU(0)
  preconditioner (caches the factorisation across steps for fixed matrices).
- **Boundary conditions** — no-slip wall, slip wall, velocity inlet, pressure
  outlet, symmetry, periodic, prescribed temperature, heat flux, adiabatic wall.
- **Immersed obstacles** — internal geometry (steps, baffles, cylinders) declared
  under `"obstacles"` as axis-aligned boxes or shaped primitives (cylinder /
  sphere) and represented by blocked cells with direct forcing (zero velocity
  and zero face flux in solid); no matrix row pinning, so the cached Poisson
  factorisation stays valid.
- **Two independent visualisation systems** —
  - **Matplotlib**: pressure / temperature / velocity contours, vector
    quiver, streamlines, VOF interface contour, and MP4 / GIF animations;
  - **Tecplot**: ASCII `.dat` files for Tecplot 360 exporting
    `X Y Z U V W P T Alpha`.  Uses the modern `ZONETYPE=ORDERED` /
    `DATAPACKING=POINT` dialect (the same as the
    [py2tec](https://github.com/luohancfd/py2tec) tools) and round-trips through
    `py2tec.tec2py`; one ORDERED zone per time step with `STRANDID` +
    `SOLUTIONTIME` for time animation.
- **Outputs produced simultaneously**: CSV, HDF5, Tecplot `.dat`, PNG, MP4/GIF.
- **Numba-accelerated**: the performance-critical element-wise TVD flux
  limiters are JIT-compiled with `@njit` (`numerics/numba_kernels.py`), with a
  pure-NumPy fallback when Numba is absent.  The rest of the heavy numerics are
  already vectorised NumPy, leaving further `@njit`/`@cuda.jit` targets cleanly
  isolated (see [Extending](#extending-the-framework)).
- **NVIDIA CUDA acceleration** — the [`gpu/`](#gpu-acceleration) package:
  hardware detection with a startup report, a NumPy/CUDA array backend with a
  device memory pool, `@cuda.jit` sparse-CSR matvec and BLAS-1 reduction
  kernels, and a GPU-resident preconditioned BiCGSTAB.  Gated by a single
  `use_gpu` case-file flag and **self-disabling**: a machine without a
  CUDA-capable GPU silently runs the original CPU path with identical numerics
  and pays no GPU import cost.
- **Force integration** — `solver/forces.py` integrates pressure + viscous
  traction over an immersed body to give `Cd`, `Cl` and the surface `Cp` / `Cf`
  distributions, plus recirculation length and separation angle
  (`"compute_forces": true`).
- **Rhie–Chow momentum interpolation** (`"rhie_chow": true`) to suppress
  collocated pressure checkerboarding, and a pressure-outlet Dirichlet path.
- **Curved-boundary immersed methods (experimental)** — a mirror-point
  ghost-cell IBM (`solver/ibm.py`, `"immersed_method": "ibm"`) and a cut-cell
  geometry kernel (`solver/cut_cell.py`, `"ibm_cut_cell"`, currently dormant).
  Their validation status is documented honestly in
  [`Handoff_Cylinder.md`](Handoff_Cylinder.md) — see the
  [staircase caveat](#example-5--flow-past-a-circular-cylinder).

---

## Installation

### Requirements

- Python **3.11+** (the code uses `from __future__ import annotations` and
  modern type hints; developed and verified on 3.11, compatible with 3.12+).

### Dependencies

Only the libraries listed in the project specification are used (pinned in
[`requirements.txt`](requirements.txt)):

| Library       | Purpose                                  | Required? |
|---------------|------------------------------------------|-----------|
| numpy         | array numerics                           | yes       |
| scipy         | sparse matrices, Krylov solvers, ILU     | yes       |
| matplotlib    | PNG plots, MP4/GIF animations            | yes       |
| tqdm          | progress bar                             | optional  |
| h5py          | HDF5 output                              | optional* |
| numba         | `@njit` TVD limiter (with fallback) **and** the `@cuda.jit` GPU kernels | optional† |
| meshio        | mesh I/O (future unstructured meshes)    | optional  |
| pyvista       | optional 3D viewer (not used by default) | optional  |
| PyYAML        | YAML case files (JSON always works)      | optional  |

\* HDF5 export is skipped gracefully if `h5py` is missing; the run still
produces CSV / Tecplot / PNG / MP4.  Likewise, the TVD limiter falls back to a
pure-NumPy implementation if `numba` is missing.

† Numba is the **only** additional dependency the GPU path needs — there is no
CuPy, no CUDA Python bindings and no compiled extension module.  See
[GPU acceleration](#gpu-acceleration) for the hardware and driver requirements.

Install the runtime stack (recommended, from the `CFDPYGPU/` directory —
there is no requirements file at the repository root):

```bash
pip install -r requirements.txt          # core + recommended extras
```

or, equivalently:

```bash
pip install numpy scipy matplotlib tqdm h5py numba
# optional extras (not required to run the examples):
pip install pyyaml meshio pyvista
```

A working **ffmpeg** binary on the system `PATH` enables MP4 animation; if
ffmpeg is absent or the codec is unavailable, CFDPyGPU automatically falls back to
a pillow-written **GIF**.

### No build step

CFDPyGPU is pure Python — just clone the repository and run `main.py` from the
`CFDPYGPU/` directory, so that the package imports (`config`, `mesh`, `numerics`, …)
resolve.

---

## Quick start — running the examples

From the `CFDPYGPU/` directory:

```bash
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
python main.py examples/cylinder_flow/config.json
python main.py examples/liquid_drop_splash_3D/config.json   # 3-D splash (GPU)
python -m visualization.render_vof_3d examples/liquid_drop_splash_3D/config.json
```

Each run prints a *CFDPy Hardware Report* naming the execution device, then a
header, a progress bar, and writes all configured outputs to the case's
`output_dir` (`outputs/<name>/` by default).

> ⚠️ **Fresh clone: clear the splash case's `restart` key first.**
> `examples/liquid_drop_splash_2D/config.json` ships with
> `"restart": "outputs/liquid_drop_splash_2D/frame_001242.h5"`, the checkpoint from which the
> case was extended from 1.2 s to 4.0 s.  Runtime frames are not tracked in git, so on a fresh
> clone that file does not exist and the run aborts when it tries to open it.  Set
> `"restart": ""` for a from-scratch run (see
> [Restart / checkpoint resume](#restart--checkpoint-resume)).

### Example 1 — Natural convection in a square cavity

A 1 m × 1 m cavity, **west wall hot (350 K)**, **east wall cold (300 K)**,
**top/bottom adiabatic**, Earth gravity pointing down, fluid initially at
rest at the reference temperature `T_ref = 325 K`.  This is the canonical
differentially heated cavity (`Pr = ν/α = 0.71`,
`Ra = g β ΔT L³/(ν α) = 1 × 10⁵`): a Boussinesq model fluid whose diffusivity
is sized so the thermal boundary layer forms and the buoyancy drives a steady
clockwise convection cell within the 8 s run, with all fields reported in
dimensional SI units (temperature in K, velocity in m/s, pressure in Pa).
(Real-property fluids such as water have a thermal diffusivity so small that,
in a 1 m cavity, the conduction time `L²/α ≈ 7 × 10⁶ s` — the wall heat never
penetrates and no convection develops on a seconds-long run; the model fluid
makes the physics visible while keeping the output dimensional.)

```bash
python main.py examples/natural_convection_2D/config.json
```

Outputs: temperature & pressure PNGs, vector & streamline overlays, the
time history CSV (with the mean Nusselt number on the hot wall), Tecplot
`.dat` frames, HDF5 snapshots, and `*_T.mp4` / `*_p.mp4` / `*_velocity.mp4`
animations.

### Example 2 — Dam break

A 2 m × 1 m tank with a water column occupying the left half up to half the
height, air everywhere else.  The water collapses under gravity; the VOF field
tracks the free surface.

```bash
python main.py examples/dam_break_2D/config.json
```

Outputs: free-surface evolution (α contour on the temperature PNG), pressure
PNGs, Tecplot `.dat` frames, HDF5 snapshots, and `*_alpha.mp4` / `*_p.mp4` /
`*_T.mp4` / `*_velocity.mp4` animations.

### Example 3 — Flow over a backward-facing step

A 17 m × 2 m channel with a sudden expansion: an upstream duct of height
`h = 1 m` opens, at `x = 2 m`, into a downstream channel of height `H = 2 m`
(expansion ratio 2:1).  A uniform inlet `U = 1 m/s` drives the flow; with
`ρ = 1`, `μ = 0.01` the step-height Reynolds number is `Re_h = ρU h/μ = 100`
(laminar).  The step itself is an **immersed obstacle** — an axis-aligned box
`[0, 2] × [0, 1]` declared under the `"obstacles"` key — represented on the
Cartesian grid by blocked cells (direct forcing: zero velocity and zero face
flux in solid cells).  A separation bubble forms at the step corner and
reattaches downstream; the recirculation length and shape are the quantities of
interest.

```bash
python main.py examples/backward_facing_step/config.json
```

Outputs: pressure & temperature PNGs with velocity-vector overlays, Tecplot
`.dat` frames, CSV & HDF5 snapshots, and `*_T.mp4` / `*_p.mp4` /
`*_velocity.mp4` animations.

### Example 4 — The splash of a liquid drop

The classic Harlow & Shannon (1967) free-surface benchmark.  A 1 m × 1.6 m
closed tank holds a shallow **water pool** (the bottom `0.3 m`, `α = 1`) under
air.  A circular **water drop** of radius `0.10 m` is suspended at
`(0.5, 1.1) m` in the air phase.  Released from rest, it free-falls under
gravity, **splashes** on impact (t ≈ 0.37 s) — driving a crater, an upward
splash crown and ejecta — and the simulation is run to `tfinal = 4.0 s` so the
full post-impact dynamics (crown collapse, secondary jets, free-surface
settling) are captured.  The VOF field tracks the deforming free surface
throughout.

```bash
python main.py examples/liquid_drop_splash_2D/config.json
```

The drop + pool initial shape is produced by the `"splash_drop"` `alpha_init`
mode; its geometry is set by `pool_height`, `drop_x`, `drop_y` and `drop_r`
(any value `<= 0` auto-resolves from the domain size).  Density ratio
`ρ/ρ_air ≈ 833`, no surface tension (`sigma = 0`), same VOF numerics as the
dam break (implicit time stepping, adaptive CFL, `use_ilu: false`).

> **Performance note.**  Adaptive CFL makes this case strongly time-dependent
> in cost: cheap (~350 wall-s per sim-s) during the drop's free fall, spiking
> to ~600–800 at impact, and **rising through the active splash to ~2000
> wall-s per sim-s** before the free surface settles.  The full 0 → 4.0 s run
> takes roughly **55–60 min** on a single core at `Nx×Ny = 80×120`, of which
> the post-impact leg dominates.  Mass is conserved to ~1.7 % over the whole
> run (interface smearing only — there is no real mass loss).  See
> [Restart / checkpoint resume](#restart--checkpoint-resume) for how the case
> was built up in legs, and set `"flow_streamlines": false` (as this case
> does) so `finalize()` does not stall rendering the velocity animation.

Outputs: free-surface evolution (α contour on the temperature PNG), pressure
PNGs, Tecplot `.dat` frames, HDF5 snapshots, and `*_alpha.mp4` / `*_p.mp4` /
`*_T.mp4` / `*_velocity.mp4` animations (each ~10 s, 20 fps, spanning 0 → 4 s).

#### The 3-D splash — `liquid_drop_splash_3D` (0 → 2.0 s)

The same benchmark in a **3-D closed tank** (1.0 × 1.5 × 1.0 m, 48×72×48
cells ≈ 166 k): the `"splash_drop"` initial condition becomes a true
**spherical** drop (`drop_r = 0.12 m` at `(0.5, 1.05, 0.5)`, set by the new
`drop_z` key) above the 0.3 m pool.  The run is built up in restart legs
(0 → 1.0 s, then 1.0 → 2.0 s from the `frame_000606.h5` checkpoint, see
[Restart / checkpoint resume](#restart--checkpoint-resume)) and takes
**~48 min + ~55 min on an RTX 4050 laptop GPU**; mass is conserved to 1.3 %
over the 2 s.  The movie below spans the whole sequence: free fall, impact
(t ≈ 0.33 s), crater, annular splash crown, the central Worthington jet
(t ≈ 0.9 s) and the crown jets breaking into fingers.

```bash
python main.py examples/liquid_drop_splash_3D/config.json
python -m visualization.render_vof_3d examples/liquid_drop_splash_3D/config.json
```

The 3-D free surface is rendered by `visualization/render_vof_3d.py` (the 2-D
matplotlib viewer does not extend to volumes): a shaded **free-surface height
map** for the pool-connected water, an interface point cloud coloured by flow
speed for the airborne drop and ejecta, and frames stitched to H.264 with
ffmpeg.  Frames are exported in Tecplot's **Z-up** axis convention
(`Y ← z`, `Z ← y`, `V ↔ W` swapped — see `visualization/tecplot_writer.py`)
and each 3-D `.dat` embeds an `Alpha = 0.5` iso-surface style block, so
Tecplot 360 shows the water (pool + drop + crown) on load.  A GPU-Krylov
option (`"use_gpu_krylov"`) exists for this strong-contrast regime but keeps
the CPU BiCGSTAB default — measured on the interface right-hand side it
stagnates where the CPU path still converges (§4 of
[GPU_PERFORMANCE_REPORT.md](GPU_PERFORMANCE_REPORT.md)).

The movie is rendered **directly from the Tecplot `.dat` frames** (which also
validates the export round-trip):

```bash
python -m visualization.render_vof_3d examples/liquid_drop_splash_3D/config.json \
    --source dat --fps 5 --name Splash_Drop_3D_2s.mp4
```

<video src="outputs/liquid_drop_splash_3D/Splash_Drop_3D_2s.mp4"
       controls muted playsinline width="720"></video>

*(Movie: `outputs/liquid_drop_splash_3D/Splash_Drop_3D_2s.mp4` — 51 Tecplot
frames at 5 fps, 0 → 2.0 s: free fall, impact, crater, splash crown,
Worthington jet, finger instability.)*

Every HDF5 frame written during a run (`frame_XXXXXX.h5`) is a valid restart
snapshot: it stores `u, v, w, p, T, alpha` plus the simulation time.  Point the
`"restart"` config key at a snapshot and the driver resumes from it instead of
calling `initialize()`:

```json
"restart": "outputs/liquid_drop_splash_2D/frame_001242.h5"
```

On restart the framework:

* loads the cell-centred fields and the recorded time, re-applies the velocity
  BCs and the obstacle direct-forcing, and seeds the time loop from that time;
* continues the per-frame output numbering past the highest existing frame so
  no output file is overwritten;
* **pre-loads the previously saved frames and the `history.csv` rows** into the
  viewer / history so the final animations and history table span the *whole*
  run (0 → 4.0 s), not just the resumed leg.

This is how the splash case above was extended from 1.2 s to 4.0 s without
recomputing the first leg.  Leave `"restart"` empty (the default) to start a
case from scratch.

Three things to remember when resuming:

1. **`save_hdf5` must have been on** for the earlier leg (and `h5py` installed),
   or there are no checkpoints to resume from; `output_interval` sets how often
   they are written.
2. **Extend `tfinal` past the checkpoint time.**  The loop runs
   `while time < tfinal`, so resuming a 1.2 s checkpoint with `tfinal` still at
   1.2 s loads the state and exits without stepping.
3. **The path is relative to the directory you launch `main.py` from**
   (`CFDPYGPU/`), not to the case folder — the same convention as `output_dir`.
   A path that does not exist aborts the run; it does not silently start from
   scratch.

Checkpoints are interchangeable between CFDPy and CFDPyGPU for the same mesh:
both write the same HDF5 layout, so a run started on the CPU framework can be
resumed here and vice versa.

> 📄 The complete step-by-step procedure, including how to pick a checkpoint and
> what the framework does on resume, is in the root
> [README — Restarting a Simulation from a Checkpoint](../README.md#restarting-a-simulation-from-a-checkpoint).

#### Velocity animation & the streamline hang

`finalize()` renders a velocity animation with `MatplotlibViewer.animate_flow`.
By default it overlays `ax.streamplot` streamlines, which is informative for
laminar cases (backward-facing step) but **stalls indefinitely** on the chaotic
velocity fields produced by VOF splash / dam-break cases — `finalize()` then
never completes and `history.csv` is not written.  The `"flow_streamlines"`
flag (default `true`) gates the overlay; set it to `false` for free-surface
cases to keep `finalize()` responsive (the speed-magnitude + quiver overlay is
still rendered).  The splash and dam-break cases set it to `false`.

#### Tecplot `.dat` output format

`TecplotExporter.write` emits the modern Tecplot 360 ASCII dialect used by the
[py2tec](https://github.com/luohancfd/py2tec) tools — one structured `ORDERED`
zone per time step with `POINT` data packing:

```
TITLE = "CFDPY snapshot t=..."
VARIABLES = "X","Y","Z","U","V","W","Pressure","Temperature","Alpha"
ZONE T="t=..." ZONETYPE=ORDERED I=Nx J=Ny [K=Nz]
     DATAPACKING=POINT STRANDID=1 SOLUTIONTIME=...
<one line per node, all 9 variables, I (x) varies fastest>
```

This replaces the legacy `F=POINT` token (a deprecated finite-element
specifier that conflicts with `DATAPACKING` and is rejected by current
Tecplot 360).  The files round-trip through `py2tec.tec2py`, and `STRANDID` +
`SOLUTIONTIME` let Tecplot chain the per-step files into a time animation.  All
four example output directories ship their `frame_*.dat` in this format.

#### Immersed obstacles (blocked cells)

Internal geometry that the boundary patches cannot express — a step, a flap, a
cylinder approximated by a staircase — is added through the `"obstacles"`
list.  Each entry is either an axis-aligned box in physical coordinates or a
shaped obstacle (cylinder / sphere) that the mask builder rasterises into the
same cell mask:

```json
"obstacles": [
    {"x0": 0.0, "x1": 2.0, "y0": 0.0, "y1": 1.0},                         // box
    {"shape": "cylinder", "center": [5.0, 2.05], "radius": 0.5, "axis": "z"} // circular (2-D)
]
```

A cell whose centre lies inside any obstacle (box rectangle, or
`(Xc−cx)² + (Yc−cy)² ≤ r²` for a z-axis cylinder) is flagged solid.  The
framework then:

* zeroes the normal **face flux** at every solid/fluid interface (no-penetration),
  threaded through every face-flux routine (`momentum`, `pressure`, `projection`);
* clamps the cell-centred **velocity to zero** inside solid cells after the
  predictor and the pressure correction (direct forcing, no-slip);
* lets the pressure Poisson see zero divergence in the solid region.

No matrix row is pinned and no sparsity pattern is altered, so the cached
variable-coefficient Poisson matrix and its ILU factorisation stay valid — the
obstacle is a pure data mask layered on top of the existing solvers.

### Example 5 — Flow past a circular cylinder

The classic external-aerodynamics benchmark.  A cylinder of diameter `D = 1`
is centred at `(5.0, 2.05)` in a `22 D × 4.1 D` channel, with a uniform inlet
`U∞ = 1` on the west boundary, a pressure outlet on the east boundary, slip
walls top and bottom, and no-slip on the cylinder.  The viscosity is set from
the Reynolds number (`μ = ρ U∞ D / Re`, with `ρ = U∞ = D = 1` so `μ = 1/Re`),
giving a sweep `Re = 20, 40, 100, 200, 300, 1000` that spans the steady
symmetric regime (Re ≤ ~47, closed near-wake) and the von Kármán shedding
regime (Re ≥ ~50).  The case uses TVD (`vanleer`) convection, implicit time
stepping with adaptive CFL, ILU-preconditioned BiCGSTAB for both the momentum
and Poisson linear systems, **Rhie-Chow** momentum interpolation to suppress
collocated pressure checkerboarding, and the **force-integration** path
(`compute_forces: true`) that records `Cd`, `Cl` in the history and renders
vorticity + force-history animations.

```bash
# Single case (Re=40, the default in config.json):
python main.py examples/cylinder_flow/config.json

# Reynolds sweep + mesh study + Markdown report:
python examples/cylinder_flow/run_reynolds.py                 # validation subset (Re=40, 100)
python examples/cylinder_flow/run_reynolds.py --mesh-study    # Re=40 at 200/400/800/1600 cells
python examples/cylinder_flow/run_reynolds.py --full          # full Re sweep (expensive)
```

`run_reynolds.py` post-processes each case (mean Cd, Cl_rms, Strouhal from a
windowed FFT of the lift history, recirculation length, separation angle),
compares against the literature table in `benchmarks.py` (Tritton, Coutanceau
& Bouard, Dennis & Chang, Norberg, Henderson), and writes
`examples/cylinder_flow/cylinder_report.md` with a summary table and a
per-case block (Re, cells, mean dt, iterations, compute time, Cd, Cl, Cl_rms,
St, Lr, % vs benchmark, quality assessment).

> **Staircase caveat (important — this case does NOT validate as-is).**  The
> cylinder is rasterised into the cell mask as a **staircase** (jagged polygon),
> not a smooth immersed boundary.  For a *bluff* body this pins flow separation
> at the staircase's 90° corners, producing a corner-separated polygonal wake
> that does **not** approach the smooth-circle wake at feasible resolutions.
> The Re=40 validation run confirms this: steady `Cd ≈ 3.71` at `200×80` and
> **`Cd ≈ 3.65` at `400×160`** (literature `1.52`) — the error barely moves
> under 2× refinement (it is mesh-independent, not slow convergence), and the
> recirculation length `Lr/D ≈ 0.85` vs the literature `2.20` shows the *wake
> field* itself is qualitatively wrong.  The force formula is correct (the
> transient and the pressure/viscous split are physically sensible); the
> discrepancy is the discrete body, not the force integration.  Treat this
> example as a demonstration of the staircase limitation, not as a literature
> match.  The reported `div` is the collocated face-averaged divergence
> residual, not a mass-conservation diagnostic.

**Where the curved-boundary work stands.**  Two attempts at replacing the
staircase have been made and are documented in full in
[`Handoff_Cylinder.md`](Handoff_Cylinder.md) §3a and §6:

| Mesh | Method | Cd (steady) | Cl | Lr/D | vs literature (Cd 1.52 / Lr 2.20) |
|---|---|---|---|---|---|
| 200×80 | staircase | 3.71 | −0.61 | 0.85 | Cd +144 %, Lr −61 % |
| 200×80 | **ghost-cell IBM** | **3.19** | −0.57 | **0.91** | Cd +110 %, Lr −59 % |

* **Mirror-point ghost-cell IBM has landed** (`solver/ibm.py`, opt-in via
  `"immersed_method": "ibm"`, already set in the cylinder config).  It runs
  stably but moves Cd only 14 %, because it enforces the *tangential* no-slip at
  the true wall while **no-penetration is still imposed on the staircase faces**
  by `BoundaryCondition.mask_solid_faces` — and for a bluff body it is the
  no-penetration geometry, not the wall tangential velocity, that sets
  separation.
* **A cut-cell attempt was stopped** and left behind `"ibm_cut_cell": false`.
  The geometry kernel (`solver/cut_cell.py`) is verified — solid area recovers
  *πr²* to 1.8e-5 / 7.7e-6 / 6.3e-6 at 200×80 / 400×160 / 800×320 with exact
  mirror symmetry — and the aperture-only Poisson plus flux-form face correction
  drives the cut-cell divergence to ~1.6e-14.  The wall is *collocated velocity
  recovery*: no way of turning the corrected face fluxes back into cell-centred
  velocities is simultaneously stable, non-smoothing and consistent with the
  cut-cell divergence.  A stable+accurate cut-cell needs face fluxes carried as
  primary state (staggered-like) — deferred by decision.
* **Centre snapping is kept ON** (`"snap_obstacle_to_grid": true`): it snaps the
  cylinder centre to the nearest grid line, removing the spurious steady lift
  caused by a lopsided staircase.  It is independent of the cut-cell flag and is
  a standalone improvement.

**Do not launch `--full` or the Re=100 case without an explicit reason** — both
reproduce the corner-pinned wake at high cost (Re=40 at 400×160 already takes
~2 h 26 min).  `run_reynolds.py` is kept as the harness ready to drive a future
wall-flux IBM.

Outputs: pressure / velocity / vorticity PNGs, `*_vorticity.mp4` +
`*_velocity.mp4` animations, the `Cd(t)` / `Cl(t)` / Cl-FFT force-history
plots, Tecplot `.dat` frames, HDF5 snapshots, and `history.csv` (with `Cd`,
`Cl`, `div`, `dt`, CFL columns) per case in `outputs/cylinder_Re<N>_<mesh>/`.

---

## GPU acceleration

This is what distinguishes CFDPyGPU from [CFDPy](../CFDPY/README.md).  The
measured profiling and benchmark numbers behind every claim below live in
[`GPU_PERFORMANCE_REPORT.md`](GPU_PERFORMANCE_REPORT.md).

### Opt-in and self-disabling

A single **`use_gpu`** flag in the case file (default `true`) gates the GPU
path.  At startup the framework probes the hardware and prints a *CFDPy
Hardware Report*.  When a CUDA-capable NVIDIA GPU is present the device is
used; when it is not — or when `"use_gpu": false` — the framework falls back to
the original pure-NumPy/SciPy code path with **identical numerics**.  A
CPU-only machine pays no GPU import cost: the CUDA backend and kernels are
imported lazily, only when a GPU is actually in use.

```json
{ "...": "...", "use_gpu": false }      // force the CPU path for validation
```

```bash
python -c "from gpu import print_hardware_report; print_hardware_report()"
```

### The layered `gpu/` package

The CPU/GPU choice is made in one place so the surrounding solver code stays
device-agnostic.  Each layer depends only on the one below it.

| Module | Responsibility |
|---|---|
| `gpu/hardware.py` | The single source of truth for *"is there a usable NVIDIA GPU?"*.  Probes device attributes (name, compute capability, SM count, memory, warp size) and queries CUDA driver/runtime versions through the bundled `cudart64` DLL when locatable.  Side-effect free — it only *reads* attributes, never allocating device memory or creating a context that would disturb a later run. |
| `gpu/backend.py` | An array/device backend with two implementations of one interface: `NumPyBackend` (the CPU path and the fallback) and `CUDABackend` (Numba-CUDA device arrays in Numba's per-context memory pool, so repeated `zeros`/`empty` of the same shape reuse cached allocations instead of round-tripping through `cudaMalloc`/`cudaFree`).  `get_backend()` returns the CUDA backend when `use_gpu` is true *and* a GPU was detected; the result is cached process-wide. |
| `gpu/kernels.py` | Low-level `@cuda.jit` kernels: a sparse CSR matvec (`matvec_csr`, one thread per row, grid-stride, race-free without atomics) and the BLAS-1 primitives a Krylov driver needs (`dot`, `dot2`, `max_abs`, `norm2`, and the in-place `copy` / `axpy` / `scale_add` / `fill` / `div_pointwise`).  The two-level shared-memory tree reduction (`BLOCK = 256`) writes one partial per block — no float atomics, so results are deterministic — and reduces the partials on the device, so only **one float crosses device→host per reduction**.  `dot2` fuses two dot products into a single host sync, which is exactly what the BiCGSTAB *ω*-step needs. |
| `gpu/linear.py` | `GPUBiCGSTAB` — a GPU-resident preconditioned BiCGSTAB (van der Vorst) with an optional **Jacobi (diagonal) preconditioner**, the only preconditioner that is cheap on a GPU (a pointwise divide) and free of the sequential triangular solves that make ILU awkward on a device.  Workspace is allocated once per solver instance and reused; the inverse diagonal is cached per matrix identity.  Convergence is tested on the cheap recurrence residual but **verified with the true residual** `b − A x` on exit, to catch the BiCGSTAB phantom-convergence stop, and the residual norm is only evaluated every 4 iterations (`check_every = 4`) to cut sync overhead. |
| `gpu/multigrid.py` | `GPUGeometricMultigrid` — a cell-centred **geometric multigrid V-cycle** for the pressure-Poisson solve: red-black Gauss-Seidel smoothing (one 3-D kernel launch per colour per sweep), full-weighting restriction, trilinear prolongation, harmonic-rediscretised coarse operators and a cached direct LU on the coarsest level.  The production `PressureSolver` uses it for the pure-Neumann steps inside its convergence envelope (mild density contrast; `pressure_gpu_solver` / `mg_max_density_ratio` config) and falls back to `GPUBiCGSTAB` / SciPy otherwise — strong-contrast VOF (air/water, ratio 833) needs the Krylov path, where rediscretised coarse operators diverge. |

### Validation

Two standalone harnesses check the GPU work against the CPU reference:

```bash
python -m gpu.validate_kernels     # every kernel vs its NumPy reference
python -m gpu.validate_linear      # GPU BiCGSTAB vs CPU on the real operator
python -m gpu.validate_multigrid   # geometric multigrid vs direct + Krylov
```

`validate_kernels` confirms the matvec is **exact** and the reductions agree to
~1e-15 relative.  `validate_linear` solves the **real production
pressure-Poisson operator** built by `PressureSolver._matrix` (pure-Neumann,
mean-projected RHS, mean-subtracted solution) and finds the GPU solution
agreeing with the CPU one to L2 ~1e-6 / rel∞ ~1e-7 — within the solver
tolerance amplified by the condition number, as expected of two iterative
solvers stopping on a relative residual.

### Hardware and driver requirements

- An **NVIDIA CUDA-capable GPU** (any compute capability the installed Numba
  build supports; developed and benchmarked on a GeForce RTX 4050 Laptop GPU,
  compute capability 8.9, 6 GB).
- The **NVIDIA CUDA driver** installed on the system.  The runtime comes either
  from a system CUDA Toolkit or, more conveniently, from the `nvidia-*` pip
  wheels, which ship `cudart64` and are auto-discovered by `gpu/hardware.py`:

  ```bash
  pip install nvidia-cuda-runtime-cu12   # optional; only for the version report
  ```

- **Numba ≥ 0.58 built with CUDA support** (`numba.cuda`) — already in
  [`requirements.txt`](requirements.txt).  There is **no** dependence on CuPy,
  on CUDA Python, or on a hand-written extension module, and **no build step**.

Non-NVIDIA GPUs (AMD ROCm, Intel oneAPI, Apple Metal) are **not** supported by
this path; those machines fall back to the CPU implementation automatically.

### Parallel execution strategy

- **One thread per row / per cell**, with grid-stride loops so a single launch
  covers any problem size.  The sparse matvec assigns each row to exactly one
  thread (race-free accumulation, no atomics); the reductions use a 256-thread
  block so the shared-memory tree reduction is exact.
- **No inter-block coupling**, so the 2-D stencil operators slated for the
  roadmap map to natural 2-D CUDA grids with shared-memory halos and need no
  host↔device transfer during a cycle.
- **Fields stay resident**: the device memory pool and the per-solver workspace
  cache keep allocations alive across steps, eliminating repeated
  `cudaMalloc`/`cudaFree` and the per-solve copies that would otherwise
  dominate at small *N*.
- **Multi-GPU / MPI extension point**: the backend carries a `device_index` and
  activates its context on first use, so a future domain-decomposition path can
  call `init_backend(device_index=local_rank)` once per MPI rank — one rank, one
  GPU — with no change to the rank-local kernels.

### Status — honest benchmark result

> **The GPU kernels are validated but deliberately NOT yet wired into the
> production `PressureSolver`.**

The profiling in [`GPU_PERFORMANCE_REPORT.md`](GPU_PERFORMANCE_REPORT.md) shows
the bottleneck *shifts* between regimes: in single-phase runs the ILU
factorisation is 91.5 % of the step, while in VOF / adaptive-Δt runs the
matrix changes every step, ILU is useless, and ~2000 BiCGSTAB matvecs per step
are 97.7 % of the cost.  Measured on the real Poisson operator:

| Grid | N | CPU + ILU | CPU no-ILU | GPU | vs CPU+ILU | vs CPU no-ILU |
|------|---|-----------|------------|-----|------------|---------------|
| 60×60   | 3 600  | 2.95 ms   | 4.78 ms   | 113.4 ms | 0.03× | 0.04× |
| 200×160 | 32 000 | 48.7 ms   | 254.6 ms  | 272.4 ms | 0.18× | 0.93× |
| 400×160 | 64 000 | 108.7 ms  | 567.3 ms  | 437.8 ms | 0.25× | **1.30×** |

The GPU wins in the VOF regime at production size and the advantage grows with
*N*, but ILU is a far stronger preconditioner than Jacobi, so CPU+ILU still
wins for single-phase.  Per the incremental methodology — *profile → implement →
validate → benchmark → promote only on success* — an increment that would
regress the production solver is not promoted.  The
[GPU geometric-multigrid preconditioner](GPU_PERFORMANCE_REPORT.md#5-incremental-gpu-roadmap)
is the next step and the one that makes both regimes net wins.

### Profiling

```bash
python profile_hotspots.py examples/cylinder_flow/config.json 30
python profile_hotspots.py examples/dam_break_2D/config.json 20
```

`profile_hotspots.py` drives the real `Simulation` under `cProfile` with all
I/O disabled and three warm-up steps, so only numerical work is measured and
JIT compilation / lazy matrix builds are excluded.

---

## How to change parameters

Every parameter lives in the **case file** (`config.json` / `config.yaml`) —
no code changes are needed.  The flat form is the most compact:

```json
{
    "name": "my_case",
    "Nx": 100, "Ny": 100, "Nz": 1,
    "Lx": 1.0, "Ly": 1.0, "Lz": 1.0,

    "dt": 0.001, "tfinal": 2.0,
    "time_scheme": "crank-nicolson",
    "adaptive_dt": true, "cfl_max": 0.3,

    "rho": 1000.0, "mu": 1.0e-3, "cp": 4180.0, "k": 0.6,
    "beta": 2.0e-4, "t_ref": 325.0,
    "gravity": [0.0, -9.81, 0.0],
    "boussinesq": true,
    "use_vof": false,

    "convection": "tvd", "limiter": "vanleer",
    "linear_solver": "bicgstab", "use_ilu": true,
    "linear_tol": 1e-6, "poisson_tol": 1e-7,

    "velocity_bc": { "west": "no-slip", "east": "outlet",
                     "south": "no-slip", "north": "no-slip" },
    "temperature_bc": { "west": {"kind":"fixed","value":350.0},
                        "east": {"kind":"fixed","value":300.0},
                        "south": "adiabatic", "north": "adiabatic" },
    "pressure_bc": { "west":"neumann", "east":"neumann",
                     "south":"neumann", "north":"neumann" },

    "output_dir": "outputs/my_case",
    "output_interval": 0.1,
    "save_csv": true, "save_hdf5": true,
    "save_tecplot": true, "save_png": true, "save_mp4": true,
    "verbose": true
}
```

The most useful knobs:

- **Resolution**: `Nx`, `Ny`, `Nz` (set `Nz = 1` for a 2D case).
- **Domain size**: `Lx`, `Ly`, `Lz`.
- **Time step / horizon**: `dt`, `tfinal`; `adaptive_dt` + `cfl_max` let the
  CFL condition shrink `dt` automatically (`dt_min` / `dt_max` bound it).
- **Time scheme**: `"implicit"` (1st-order backward Euler) or
  `"crank-nicolson"` (2nd-order).
- **Convection**: `"upwind"`, `"central"`, `"quick"`, `"tvd"` (with `limiter`).
- **Linear solver**: `"cg"` (symmetric systems, e.g. pressure), `"bicgstab"`,
  `"gmres"`.  `use_ilu` enables the ILU(0) preconditioner — recommended for
  constant-coefficient (single-phase, fixed-dt) cases; turn it **off** for
  variable-coefficient VOF / adaptive-dt runs (the matrix changes every step
  so the factorisation cannot be reused).
- **Physics toggles**: `boussinesq`, `use_vof`, `gravity`, `beta`, `t_ref`.
- **Boundary conditions** are dicts keyed by patch
  (`west|east|south|north|bottom|top`).  Velocity BC kinds: `no-slip`, `slip`,
  `inlet` (with `value`), `outlet`, `symmetry`, `periodic`.  Temperature BC
  kinds: `fixed` (with `value`), `heatflux` (with `value`), `adiabatic`.
- **Immersed obstacles**: `"obstacles": [{"x0":..,"x1":..,"y0":..,"y1":..,
  "z0":..,"z1":..}]` (axis-aligned boxes) or `{"shape":"cylinder",
  "center":[cx,cy],"radius":r,"axis":"z"}` / `{"shape":"sphere",...}`;
  cells whose centre lies inside an obstacle are blocked (direct forcing).
  `z0`/`z1` default to ±∞.
- **Force integration**: `"compute_forces": true` builds a `ForcesCalculator`
  for an immersed body, records `Cd`/`Cl` in `history.csv` each output frame,
  and renders vorticity + `Cd(t)`/`Cl(t)`/Cl-FFT plots.  Reads `U_inf` from the
  west inlet and `D = 2·radius` from the first cylinder obstacle.
- **Free-surface init shape**: `"alpha_init"` selects `"uniform"`, `"dam_break"`,
  `"block"`, or `"splash_drop"` (drop + pool; geometry via `pool_height`,
  `drop_x`, `drop_y`, `drop_r`, any `<= 0` = auto).
- **Restart**: `"restart": "path/to/frame_XXXXXX.h5"` resumes from a saved
  snapshot (see [Restart / checkpoint resume](#restart--checkpoint-resume));
  empty = start from scratch.
- **Velocity animation**: `"flow_streamlines": false` disables the streamline
  overlay in `finalize()` — set this for VOF splash / dam-break cases, where
  `streamplot` would otherwise stall (see
  [Velocity animation & the streamline hang](#velocity-animation--the-streamline-hang)).
- **Immersed-boundary method**: `"immersed_method"` selects `"staircase"`
  (default, cell-mask direct forcing) or `"ibm"` (mirror-point ghost-cell
  forcing at the true curved wall).  `"snap_obstacle_to_grid": true` snaps an
  obstacle centre to the nearest grid line, removing the spurious lift a
  lopsided staircase produces.  `"ibm_cut_cell"` gates the dormant cut-cell
  path and should be left `false` (see the
  [staircase caveat](#example-5--flow-past-a-circular-cylinder)).
- **GPU**: `"use_gpu"` (default `true`) enables the CUDA path when a
  CUDA-capable NVIDIA GPU is detected; set it to `false` to force the CPU path
  with identical numerics.  On a GPU machine the pressure-Poisson solve is
  computed by the geometric multigrid (`"pressure_gpu_solver": "auto"`, the
  default) whenever the problem is inside its convergence envelope —
  pure-Neumann boundaries, no cut cells, density ratio ≤ `"mg_max_density_ratio"`
  (default 5), and a grid large enough to coarsen (N > `mg_coarse_max_cells`,
  default 4096) — and by the Krylov solver otherwise; `"pressure_gpu_solver":
  "krylov"` forces the pre-multigrid behaviour everywhere.  When the multigrid
  guard declines (strong density contrast, e.g. air/water VOF) the
  `"use_gpu_krylov"` option (default `false`) moves that Krylov solve itself
  onto the GPU (`"gpu_krylov_min_cells"`, default 16384, keeps small systems
  on the CPU) — about 2x the CPU rate on benchmark systems, but note it can
  stagnate on hard two-phase right-hand sides, in which case the run
  automatically falls back to the CPU solver.  See
  [GPU acceleration](#gpu-acceleration).
- **Output**: `output_interval` (in simulation time), and the `save_*`
  booleans.  `output_dir` controls where everything lands.

A **nested** form is also accepted (`{"mesh": {...}, "time": {...}, ...}`);
both forms are merged into the same flat `Config`, with flat keys taking
precedence over nested defaults.

---

## How to create a new case

1. Copy an existing case folder:

   ```bash
   cp -r examples/natural_convection_2D examples/my_case
   ```

2. Edit `examples/my_case/config.json` — set the `name`, mesh, physics, BCs
   and output options as described above.

3. Run it:

   ```bash
   python main.py examples/my_case/config.json
   ```

For a VOF (free-surface) case, set `"use_vof": true` and pick an
`"alpha_init"`:

- `"dam_break"` — water column on the left half up to half the height;
- `"block"` — a centred square block of heavy fluid;
- `"splash_drop"` — a circular drop above a liquid pool at the bottom
  (geometry via `pool_height`, `drop_x`, `drop_y`, `drop_r`; `<= 0` = auto);
- `"uniform"` — `alpha_value` everywhere (default 0 = light phase).

The heavy/light phase densities and viscosities are `rho`/`mu` and
`rho_light`/`mu_light` respectively.

---

## Mathematical formulation

### Governing equations

Continuity (incompressible):

```
∇ · u = 0
```

Momentum (Navier–Stokes, with body force as acceleration):

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν ∇²u + g
```

Energy (temperature), written with thermal diffusivity `α = k/(ρc_p)`:

```
∂T/∂t + (u·∇)T = α ∇²T
```

Volume fraction (VOF), transported by the divergence-free velocity:

```
∂α/∂t + ∇·(α u) = 0,    α ∈ [0,1]
```

Boussinesq buoyancy (natural convection):

```
ρ = ρ₀ (1 - β (T - T₀)),    f_buoy = -β (T - T₀) g
```

### Finite-volume discretisation

- **Collocated** arrangement: all variables live at cell centres; face values
  are interpolated.  Rhie–Chow-style coupling is supplied by the projection
  (the pressure increment corrects the face flux).
- **Spatial order**: 2nd-order central differences for diffusion and gradients;
  the convection scheme is selectable (1st-order upwind, 2nd-order central,
  3rd-order QUICK, or TVD with a flux limiter).
- **Temporal order**: the convective and buoyancy terms are treated
  **explicitly**; diffusion is **implicit** with the θ-scheme,
  θ = 1 → backward Euler, θ = ½ → Crank–Nicolson.

### Projection method (incremental, variable-coefficient)

1. **Predict** — remove the old pressure gradient, advance with convection +
   diffusion + body force to get `u*`:

   ```
   u* = u^n + Δt[ -(u^n·∇)u^n + θ ν ∇²u* + (1-θ)ν∇²u^n + g - ∇p^n/ρ ]
   ```

2. **Pressure increment** — solve the variable-coefficient Poisson equation
   for `δp` (the face coefficient `1/ρ` captures the density jump at a
   water/air interface, which preserves hydrostatic equilibrium):

   ```
   ∇·( (1/ρ) ∇δp ) = (1/Δt) ∇·u*
   ```

   The operator has a one-dimensional null space (the constant pressure
   field, since only Neumann BCs appear).  Rather than pinning a matrix row
   — which would alter the sparsity pattern every time `ρ` changes — CFDPyGPU
   keeps the **pure symmetric operator** (constant sparsity, cacheable) and
   removes the null space analytically: the RHS is mean-projected before the
   solve and the solution is mean-subtracted afterwards.

3. **Correct**:

   ```
   u^{n+1} = u* - (Δt/ρ) ∇δp
   p^{n+1} = p^n + δp
   ```

### Linear solvers

The momentum/energy implicit systems and the Poisson equation are solved with
a Krylov method (CG / BiCGSTAB / GMRES) and an optional ILU(0) preconditioner
(`scipy.sparse.linalg.spilu`).  The preconditioner is cached on the matrix
identity, so for fixed-coefficient runs it is factored **once** and reused on
every step; for variable-coefficient runs (VOF) set `use_ilu: false` because
the matrix changes every step.

---

## Algorithms and solver flowchart

```
                    ┌─────────────────────────────────────────────┐
   load_config ───► │ Simulation.__init__                          │
                    │  Mesh · Fluid · Gravity · Boussinesq          │
                    │  BoundaryCondition · LinearSolver(s)          │
                    │  ProjectionMethod · EnergySolver · VOFSolver  │
                    │  MatplotlibViewer · TecplotExporter           │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                              Simulation.initialize()
                            (allocate fields, ICs, α, BCs)
                                         │
                                         ▼
                 ┌──────────────────────────────────────────┐
                 │  for each step until t >= tfinal:         │   ← Simulation.run()
                 │                                           │
                 │   1. pick dt (fixed or CFL-adaptive)      │
                 │   2. rho = fluid.blend(α)  (VOF)          │
                 │   3. body force = g + Boussinesq accel.    │
                 │   4. ProjectionMethod.step:                │
                 │        a. remove old ∇p from source        │
                 │        b. MomentumSolver.predict → u*      │
                 │           • convective term (explicit)     │
                 │           • implicit diffusion solve       │
                 │        c. PressureSolver.solve → δp, F     │
                 │           • build/cache Poisson matrix     │
                 │           • mean-project RHS, solve        │
                 │        d. velocity & pressure correction   │
                 │   5. re-apply velocity BCs                 │
                 │   6. EnergySolver.step → T^{n+1}           │
                 │   7. VOFSolver.advect → α^{n+1}            │
                 │   8. if output interval reached:           │
                 │        _save_frame → PNG/CSV/DAT/HDF5       │
                 │        add_frame for animation             │
                 └──────────────────────────────────────────┘
                                         │
                                         ▼
                            Simulation.finalize()
                  • MP4/GIF animations (T, p, α)
                  • history.csv
                  • summary printout
```

### Per-step detail (the projection step)

```
ProjectionMethod.step(u, v, w, p, dt, sources, rho):
  ── 1. predictor ─────────────────────────────────────────────
     src ← sources − (1/ρ) ∇p^n
     (u*, v*, w*) ← MomentumSolver.predict(...)
         conv  ← (u·∇)u  via face_interpolate (upwind/central/QUICK/TVD)
         M = I − θ·dt·ν·L   (cached when θ·dt·ν is constant)
         solve M u* = u + dt(−conv + src) + θ dt ν rhs_bc
  ── 2. pressure increment ────────────────────────────────────
     F  ← face fluxes of u*  (interpolated, BC-imposed)
     rhs ← ∇·u* / dt          (then rhs ← rhs − mean(rhs))
     A  ← ∇·((1/ρ) ∇·)         (pattern cached; data refreshed per ρ)
     δp ← solve A δp = rhs      (mean-subtracted)
  ── 3. correction ────────────────────────────────────────────
     u ← u* − (dt/ρ) ∇δp ;  p ← p + δp
     return u, v, w, p, δp, div
```

---

## Project organization

```
CFDPYGPU/
├── main.py                  # CLI entry point + Simulation orchestrator
├── requirements.txt         # pinned Python dependencies
├── profile_hotspots.py      # cProfile per-step hotspot driver (I/O disabled)
├── verify_cut_cell_pressure.py  # algebraic regression checks for the cut-cell Poisson
├── gpu/                     # the GPU acceleration package (layered)
│   ├── hardware.py          #   detection + startup hardware report (no state)
│   ├── backend.py           #   NumPy / CUDA array backend, memory pool, CPU fallback
│   ├── kernels.py           #   @cuda.jit sparse CSR matvec + BLAS-1 reductions
│   ├── linear.py            #   GPUBiCGSTAB: GPU-resident preconditioned Krylov solve
│   ├── validate_kernels.py  #   CPU-vs-GPU kernel validation + microbenchmarks
│   └── validate_linear.py   #   GPU BiCGSTAB vs CPU on the real Poisson operator
├── config/
│   └── config_loader.py     # Config dataclass, BoundarySpec, JSON/YAML loader
│                            #   (+ use_gpu, rhie_chow, compute_forces,
│                            #    immersed_method, snap_obstacle_to_grid, ibm_cut_cell)
├── mesh/
│   └── mesh.py              # Cartesian structured mesh (2D/3D), cell/face geometry
├── numerics/                # pure finite-volume operators (no state)
│   ├── interpolation.py     #   face value reconstruction (upwind/central/QUICK/TVD)
│   ├── numba_kernels.py     #   @njit TVD limiter kernel (NumPy fallback)
│   ├── gradients.py         #   cell & face gradients
│   ├── divergence.py        #   cell-centred divergence from face fluxes
│   ├── laplacian.py         #   constant-coefficient Poisson matrix
│   └── timestep.py          #   CFL / Fourier adaptive dt
├── physics/
│   ├── fluid.py             #   Fluid: property blending for VOF, from_config
│   ├── material.py          #   free-form material property container
│   ├── gravity.py           #   Gravity body force
│   └── buoyancy.py          #   Boussinesq buoyancy acceleration
├── solver/
│   ├── boundary.py          #   BoundaryCondition: patches, laplacian builders,
│   │                        #     solid-face masking, apply_immersion dispatch
│   ├── linear_solver.py     #   CG / BiCGSTAB / GMRES + ILU(0)
│   ├── momentum.py          #   semi-implicit NS predictor
│   ├── pressure.py          #   incremental variable-coeff Poisson,
│   │                        #     Rhie–Chow, pressure-outlet Dirichlet, cut-cell branch
│   ├── projection.py        #   ProjectionMethod (predict → solve → correct)
│   ├── energy.py            #   advection–diffusion for T
│   ├── vof.py               #   conservative VOF transport, properties, interface
│   ├── forces.py            #   ForcesCalculator: Cd / Cl / Cp / Cf, Lr, separation angle
│   ├── ibm.py               #   mirror-point ghost-cell immersed boundary (opt-in)
│   └── cut_cell.py          #   cut-cell volume/aperture geometry (dormant, flag OFF)
├── visualization/
│   ├── matplotlib_view.py   #   MatplotlibViewer: PNG + MP4/GIF
│   ├── render_vof_3d.py     #   3-D VOF frames (HDF5) -> PNG stills + MP4
│   ├── tecplot_writer.py    #   TecplotExporter: .dat (py2tec dialect) / .csv / .h5
│   └── postprocessor.py     #   vorticity, streamfunction, Nusselt
├── examples/
│   ├── natural_convection_2D/config.json
│   ├── dam_break_2D/config.json
│   ├── backward_facing_step/config.json
│   ├── liquid_drop_splash_2D/config.json
│   ├── liquid_drop_splash_3D/config.json   # 3-D splash: spherical drop,
│   │                       #   48x72x48, Tecplot+HDF5 frames, render_vof_3d
│   └── cylinder_flow/       #   Re sweep + mesh study + report
│       ├── config.json
│       ├── run_reynolds.py  #     sweep driver; writes cylinder_report.md
│       ├── benchmarks.py    #     literature comparison table + signed-% diff
│       └── _probe.py        #     ad-hoc Cd/Cl trajectory probe (edit-and-run)
├── outputs/                 # created at runtime, one folder per case;
│                            #   only the rendered .mp4/.wmv animations are tracked
├── README.md                #   this file
├── GPU_PERFORMANCE_REPORT.md #  profiling, CPU-vs-GPU benchmarks, GPU roadmap
├── Handoff.md               #   notes on the splash example, restart & Tecplot work
└── Handoff_Cylinder.md      #   cylinder benchmark: staircase, IBM and cut-cell status
```

### Design principles

- **SOLID / OOP**: each class has one responsibility; the `Simulation` is the
  only *composition root* that knows all subsystems, every other module
  depends only on its neighbours through clean interfaces (dependency
  inversion).  No globals, no duplicated logic, no over-long functions.
- **Numerics are stateless**: the `numerics/` operators are pure functions of
  arrays — they are trivially testable and reusable.
- **Config-driven**: nothing is hard-coded; every parameter comes from the
  case file.
- **Complete & documented**: every public class and function has a docstring
  (with the relevant maths); there are no `TODO`, `pass`-only stubs or
  "implementar depois" placeholders.

---

## Extending the framework

The architecture deliberately leaves clean extension points:

- **SIMPLE coupling**: `solver/projection.py` already follows the
  predict → solve → correct skeleton; a SIMPLE/PISO solver reuses
  `MomentumSolver` and `PressureSolver` with different coefficients and
  under-relaxation.
- **Unstructured meshes**: the `numerics/` operators are written against a
  face-flux / cell-index abstraction; replacing `Mesh` with an unstructured
  face-cell connectivity (via `meshio`) localises the change.
- **RANS / LES / DNS**: subgrid / turbulence closures plug into
  `MomentumSolver` as additional source terms and effective viscosities.
- **GPU / CUDA**: partially delivered — see [GPU acceleration](#gpu-acceleration).
  The device backend, the `@cuda.jit` kernels and `GPUBiCGSTAB` are implemented
  and validated; the remaining work is the multigrid preconditioner, wiring the
  GPU solve into `PressureSolver`, keeping fields device-resident, and porting
  the stencil operators (`interpolation`, `divergence`, `gradients`), which are
  pure NumPy vectorised loops that map to 2-D CUDA grids with no refactor.  The
  roadmap with expected impact per step is in
  [`GPU_PERFORMANCE_REPORT.md`](GPU_PERFORMANCE_REPORT.md) §5.
- **MPI**: domain decomposition enters at the `Mesh` / `LinearSolver` level;
  the field arrays and the projection step already operate on local arrays.
- **AMR**: the `Mesh` is a value object; a hierarchy of meshes can be
  maintained with the existing operators and a conservative refluxing step.
- **More physics**: radiation, phase change, species transport, compressible
  flow and additional multiphase models each add a new `solver/*.py` module
  driven from `Simulation.step`, mirroring `EnergySolver` / `VOFSolver`.

---

## License

Licensed under **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**, together with the rest of the repository.  The full legal
text is in [`LICENSE`](../LICENSE) at the repository root.

You are free to share and adapt this framework for any **non-commercial**
purpose, provided you give appropriate credit and indicate any changes made.
Commercial use requires a separate licence from the maintainer.  If you use
CFDPyGPU in teaching, research or a publication, please cite it as described in
the root [README](../README.md#citation).

---

## See also

- [`../README.md`](../README.md) — the book companion README: chapter
  programs, repository structure, installation, limitations and roadmap.
- [`../CFDPY/README.md`](../CFDPY/README.md) — the CPU framework this package
  is a superset of.
- [`GPU_PERFORMANCE_REPORT.md`](GPU_PERFORMANCE_REPORT.md) — profiling
  methodology, hotspot rankings, CPU-vs-GPU benchmarks and the GPU roadmap.
- [`Handoff.md`](Handoff.md) — the liquid-drop splash example, the
  restart/resume path and the Tecplot dialect migration.
- [`Handoff_Cylinder.md`](Handoff_Cylinder.md) — the cylinder benchmark: force
  integration, Rhie–Chow coupling, the staircase failure analysis, the
  ghost-cell IBM result and the halted cut-cell attempt.
