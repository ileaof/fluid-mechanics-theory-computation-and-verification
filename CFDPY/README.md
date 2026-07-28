# CFDPy — a modular Finite-Volume CFD framework in Python

CFDPy is an **educational and professional** Computational Fluid Dynamics
framework written from scratch in Python 3.11+ (no OpenFOAM, FEniCS or FiPy).
It solves the incompressible Navier–Stokes equations together with energy,
scalar transport and Volume-of-Fluid (VOF) free-surface models on a Cartesian
structured mesh, using the **Finite Volume Method** with a **projection
(fractional-step)** pressure-velocity coupling.

The code is organised as a small, decoupled package where every subsystem
(mesh, numerics, physics, solvers, visualisation) can be read and modified in
isolation.  It is designed as a research base: the architecture is ready for
unstructured meshes, RANS/LES/DNS, GPU/CUDA, MPI, AMR, additional multiphase /
compressible / radiation / phase-change / species models.

---

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start — running the examples](#quick-start--running-the-examples)
4. [How to change parameters](#how-to-change-parameters)
5. [How to create a new case](#how-to-create-a-new-case)
6. [Mathematical formulation](#mathematical-formulation)
7. [Algorithms and solver flowchart](#algorithms-and-solver-flowchart)
8. [Project organization](#project-organization)
9. [Extending the framework](#extending-the-framework)

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
- **Immersed obstacles** — internal geometry (steps, baffles) declared as
  axis-aligned boxes under `"obstacles"` and represented by blocked cells with
  direct forcing (zero velocity and zero face flux in solid); no matrix row
  pinning, so the cached Poisson factorisation stays valid.
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
| numba         | JIT-accelerated TVD limiter (with fallback) | optional |
| meshio        | mesh I/O (future unstructured meshes)    | optional  |
| pyvista       | optional 3D viewer (not used by default) | optional  |
| PyYAML        | YAML case files (JSON always works)      | optional  |

\* HDF5 export is skipped gracefully if `h5py` is missing; the run still
produces CSV / Tecplot / PNG / MP4.  Likewise, the TVD limiter falls back to a
pure-NumPy implementation if `numba` is missing.

Install the runtime stack (recommended, from the `CFDPY/` directory —
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
ffmpeg is absent or the codec is unavailable, CFDPy automatically falls back to
a pillow-written **GIF**.

### No build step

CFDPy is pure Python — just clone the repository and run `main.py` from the
`CFDPY/` directory, so that the package imports (`config`, `mesh`, `numerics`, …)
resolve.

---

## Quick start — running the examples

From the `CFDPY/` directory:

```bash
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
```

Each run prints a header, a progress bar, and writes all configured outputs to
the case's `output_dir` (`outputs/<name>/` by default).

> ⚠️ **Fresh clone: clear the splash case's `restart` key first.**
> `examples/liquid_drop_splash_2D/config.json` ships with
> `"restart": "outputs/liquid_drop_splash_2D/frame_001242.h5"`, the checkpoint from which the
> case was extended from 1.2 s to 4.0 s.  Runtime frames are not tracked in git, so on a fresh
> clone that file does not exist and the run aborts when it tries to open it.  Set
> `"restart": ""` for a from-scratch run (see
> [Restart / checkpoint resume](#restart--checkpoint-resume)).

### Example 1 — Natural convection in a square cavity

A 1 m × 1 m cavity, **west wall hot (350 K)**, **east wall cold (300 K)**,
**top/bottom adiabatic**, gravity pointing down, fluid initially at rest at
the reference temperature `T_ref = 325 K`.  Boussinesq buoyancy drives a
clockwise convection cell.

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

#### Restart / checkpoint resume

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
3. **The path is relative to the directory you launch `main.py` from** (`CFDPY/`),
   not to the case folder — the same convention as `output_dir`.  A path that
   does not exist aborts the run; it does not silently start from scratch.

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
four example cases write their `frame_*.dat` in this format when `save_tecplot`
is on.  The `.dat` frames are regenerable and are therefore not tracked in git —
only the rendered `.wmv` animations under `outputs/` are (see the root
[README](../README.md#rendered-animations-shipped-with-the-repository)).

#### Immersed obstacles (blocked cells)

Internal geometry that the boundary patches cannot express — a step, a flap, a
cylinder approximated by a staircase — is added through the `"obstacles"`
list.  Each entry is an axis-aligned box in physical coordinates:

```json
"obstacles": [
    {"x0": 0.0, "x1": 2.0, "y0": 0.0, "y1": 1.0}
]
```

A cell whose centre lies inside any box is flagged solid.  The framework then:

* zeroes the normal **face flux** at every solid/fluid interface (no-penetration),
  threaded through every face-flux routine (`momentum`, `pressure`, `projection`);
* clamps the cell-centred **velocity to zero** inside solid cells after the
  predictor and the pressure correction (direct forcing, no-slip);
* lets the pressure Poisson see zero divergence in the solid region.

No matrix row is pinned and no sparsity pattern is altered, so the cached
variable-coefficient Poisson matrix and its ILU factorisation stay valid — the
obstacle is a pure data mask layered on top of the existing solvers.

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
  "z0":..,"z1":..}]` — list of axis-aligned boxes; cells whose centre lies
  inside a box are blocked (direct forcing).  `z0`/`z1` default to ±∞.
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
   — which would alter the sparsity pattern every time `ρ` changes — CFDPy
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
CFDPY/
├── main.py                  # CLI entry point + Simulation orchestrator
├── requirements.txt         # pinned Python dependencies
├── config/
│   └── config_loader.py     # Config dataclass, BoundarySpec, JSON/YAML loader
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
│   ├── boundary.py          #   BoundaryCondition: patches, laplacian builders
│   ├── linear_solver.py     #   CG / BiCGSTAB / GMRES + ILU(0)
│   ├── momentum.py          #   semi-implicit NS predictor
│   ├── pressure.py          #   incremental variable-coeff Poisson
│   ├── projection.py        #   ProjectionMethod (predict → solve → correct)
│   ├── energy.py            #   advection–diffusion for T
│   └── vof.py               #   conservative VOF transport, properties, interface
├── visualization/
│   ├── matplotlib_view.py   #   MatplotlibViewer: PNG + MP4/GIF
│   ├── tecplot_writer.py    #   TecplotExporter: .dat (py2tec dialect) / .csv / .h5
│   └── postprocessor.py     #   vorticity, streamfunction, Nusselt
├── examples/
│   ├── natural_convection_2D/config.json
│   ├── dam_break_2D/config.json
│   ├── backward_facing_step/config.json
│   └── liquid_drop_splash_2D/config.json
├── outputs/                 # created at runtime, one folder per case
├── README.md                #   this file
└── Handoff.md               #   notes on the splash example, restart & Tecplot work
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
- **GPU / CUDA**: the heavy numeric kernels (`interpolation`, `divergence`,
  `gradients`, the matrix assembly) are pure NumPy vectorised loops — wrap
  them with `@numba.njit` (or `@cuda.jit`) with no refactor.  The
  `LinearSolver` can be swapped for a GPU Krylov solver behind the same
  `.solve(A, b)` interface.
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
CFDPy in teaching, research or a publication, please cite it as described in
the root [README](../README.md#citation).

---

## See also

- [`../README.md`](../README.md) — the book companion README: chapter
  programs, repository structure, installation, limitations and roadmap.
- [`Handoff.md`](Handoff.md) — developer notes on the liquid-drop splash
  example, the restart/resume path and the Tecplot dialect migration.
- [`../CFDPYGPU/README.md`](../CFDPYGPU/README.md) — the GPU-accelerated
  superset of this framework (Numba-CUDA kernels, force integration, immersed
  boundary work), which runs the same case files.
