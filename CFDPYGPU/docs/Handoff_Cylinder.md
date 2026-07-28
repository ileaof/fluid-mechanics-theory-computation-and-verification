# Handoff — CFDPy flow-past-a-circular-cylinder benchmark

**Date:** 2026-07-26
**Scope:** Added the classic "flow past a circular cylinder" benchmark as a
fifth example, with a Reynolds-sweep driver, a literature comparison table, a
force-integration module, Rhie-Chow collocated coupling, and a pressure-outlet
Dirichlet wiring. This file records *what changed, why, and what to watch out
for*; the authoritative framework description remains `README.md`.

---

## 1. What was done

### 1.1 New example: `examples/cylinder_flow/`
- `config.json` — base case: `22D × 4.1D` channel, cylinder `D=1` at
  `(5.0, 2.05)` (radius `0.5`), west inlet `U∞=1`, east **pressure outlet**,
  top/bottom slip, cylinder no-slip. `Nx×Ny = 400×160`, `μ=0.025` (Re=40
  placeholder; the sweep overrides it), TVD `vanleer` convection, implicit +
  adaptive CFL (`cfl_max=0.3`), ILU-preconditioned BiCGSTAB,
  `rhie_chow: true`, `compute_forces: true`, `flow_streamlines: false`.
- `run_reynolds.py` — sweep driver. Builds a `Config` per `(Re, mesh, tfinal)`,
  sets `μ = ρ U∞ D / Re`, runs, post-processes (mean Cd, Cl_rms, Strouhal via a
  windowed Hanning FFT of Cl, recirculation length, separation angle), compares
  to `benchmarks.py`, and writes `cylinder_report.md`. Flags: `--full`,
  `--mesh-study`, `--cases`, `--mesh`, `--tfinal`, `--render`, `--no-report`.
  The validation subset is `[(40,(400,160),40),(100,(400,160),80)]`; the full
  sweep adds Re=20/200/300/1000 (the last at 800×320).
- `benchmarks.py` — literature table (`Bench` namedtuple: Cd, Cl_rms, St,
  Lr/D, θ_sep) for Re = 20, 40, 100, 200, 300, 1000, plus `pct()` signed-% diff.

### 1.2 Cylinder obstacle primitive (`config/config_loader.py`, `main.py`)
The obstacle schema now accepts shaped entries alongside boxes:
`{"shape":"cylinder","center":[cx,cy],"radius":r,"axis":"z"}` and
`{"shape":"sphere",...}`. `_build_solid_mask` dispatches on `shape` (box →
rectangle test; z-axis cylinder → `(Xc−cx)²+(Yc−cy)² ≤ r²`, reusing the same
arithmetic as the splash-drop `_init_alpha`; sphere → 3-D analogue). All
shapes OR into one boolean mask; the rest of the forcing pipeline (face-flux
masking, velocity clamping, projection correction) is unchanged.

### 1.3 Force integration — `solver/forces.py` (new)
`ForcesCalculator(mesh, bc, fluid, U_inf, D)` integrates pressure + viscous
traction over the fluid/solid staircase interface:
- pressure `F_p = −p_fluid · n_body · A` (`n_body` = outward body normal);
- viscous `F_v = (2μ/h)·u_fluid·A` per component (one-sided wall gradient; the
  wall sits at the fluid/solid face, `h/2` from the fluid centre, `u_wall=0`);
- `Cd = 2Fx/(ρU∞²D)`, `Cl = 2Fy/(ρU∞²D)`.
Also: `recirculation_length` (centreline `u` sign change behind the body),
`separation_angle_deg` (where the streamwise wall shear `Cf` changes sign),
`surface` (per-facet `θ`, `Cp`, `Cf` for the surface plots).

### 1.4 Diagnostics wiring (`main.py`)
New config field `compute_forces` (default off). When on, a `ForcesCalculator`
is built in `__init__` (`U_inf` from the west inlet, `D = 2·cylinder_radius`);
`step()` adds `Cd`/`Cl` to the per-step diagnostic so they land in `history.csv`;
`_save_frame` adds `vort` (from `PostProcessor.vorticity`) to the snapshot;
`finalize()` renders `<name>_vorticity.mp4` and the `Cd(t)`/`Cl(t)`/Cl-FFT PNGs.

### 1.5 Rhie-Chow coupling (`solver/pressure.py`)
`PressureSolver._rhie_chow` replaces the averaged cell-centre pressure gradient
implicit in the face flux with the direct face pressure difference of `p^n`:
`F -= (dt/ρ_f)·[(p_R−p_L)/h − ½(g_L+g_R)]`. For a smooth pressure the bracket is
~0; for a checkerboard pressure the cell-centred gradient vanishes while the
face difference is large, so the correction damps the decoupled mode. Opt-in
(`rhie_chow` config flag); solid faces are re-masked after the correction.
Called from `solve()` when `p_old` is passed through from `ProjectionMethod.step`.

### 1.6 Pressure-outlet Dirichlet (`solver/pressure.py`)
`pressure_bc: {"east":{"kind":"outlet","value":0.0}}` is now honoured: the
outlet-cell rows of the Poisson matrix are pinned to identity and the RHS there
is `p_out − p_old`, giving `p^{n+1}=p_out`; the constant null-space projection is
then skipped. **Note:** the cylinder case actually runs with all-Neumann
pressure BCs + mean-subtraction, because pressure drag is gauge-invariant
(`∮p·n dA` has zero constant component for any closed body, staircase included
— divergence theorem `∮n dA = 0`). The Dirichlet wiring is kept for spec
compliance / generality but is not what makes the forces correct.

---

## 2. Files changed
- **New:** `solver/forces.py`, `examples/cylinder_flow/config.json`,
  `examples/cylinder_flow/run_reynolds.py`, `examples/cylinder_flow/benchmarks.py`,
  `docs/Handoff_Cylinder.md`.
- **Modified:** `config/config_loader.py` (obstacle schema + cylinder/sphere),
  `main.py` (cylinder mask, `compute_forces` wiring, vorticity + force-history
  rendering), `solver/pressure.py` (`_rhie_chow`, pressure-outlet Dirichlet),
  `solver/projection.py` (pass `p_old` to `pressure.solve`), `README.md`
  (Example 5, obstacle docs, knobs, tree).

---

## 3. Validation status — the staircase FAILS for this case (read this)

The cylinder is a **staircase** (cell-mask rasterisation), not a smooth immersed
boundary. The Re=40 validation run at the approved mesh (400×160, tfinal=15,
2 h 26 min) **does not validate**:

| mesh | cells across D | Cd (steady) | Cl (steady) | Lr/D | lit Cd=1.52 / Lr=2.20 / Cl=0 |
|---|---|---|---|---|---|
| 200×80  | ~9 × 20  | 3.71 | −0.61 | —    | Cd +144 % |
| 400×160 | ~18 × 40 | **3.65** | **−0.65** | **0.85** | Cd +140 %, Lr −61 % |

**Cd barely moved under 2× refinement (3.71 → 3.65)** — the error is essentially
**mesh-independent**, NOT slow first-order convergence. The clincher is
`Lr/D = 0.85` vs the literature `2.20`: recirculation length is a *field*
diagnostic (centreline `u` sign change), so the **wake / velocity field is
qualitatively wrong**, not just the force post-processing. Root cause: a
staircase body **pins flow separation at its 90° corners**, producing a
corner-separated polygonal wake — short recirculation, low base pressure, high
drag — that does not approach the smooth-circle wake at feasible resolutions.
This is a known severe limitation of staircase IBM for *bluff* bodies (it is
fine for streamlined bodies / internal steps, where separation is absent or
fixed anyway).

The **force formula is correct** — verified directly: the t=3 transient gives
`Fx=1.72, Fp_x=0.92, Fv_x=0.80, Cd=3.44`, all physically sensible; the mask is
the right size (140 solid cells at 200×80 vs πr²/dxdy=139.3); pressure/viscous
split is sensible. The wrong steady Cd is the wrong *wake*, not a force bug.
The persistent **Cl = −0.65** is the off-grid x-centre (`x=5.0` → cell 90.4 on
400×160) making the staircase lopsided; true Re=40 is symmetric (`Cl→0`).

### 3a. Mirror-point ghost-cell IBM landed — but does NOT fix the wake

A first curved-boundary IBM (`solver/ibm.py`, opt-in via
`"immersed_method":"ibm"`, now set in `examples/cylinder_flow/config.json`) was
added on top of the staircase. For each solid *ghost* cell (first solid layer)
it precomputes the closest true-boundary point `B = c + r·(C−c)/|C−c|`, the
image `I = 2B−C` in the fluid, and a bilinear stencil at `I`; each step it sets
`u_ghost = −u_I` so linear interpolation at the midpoint `B` is zero (no-slip at
the true wall) and zeroes the deep interior. The three `solid→0` clamp sites
(`momentum.predict`, `projection._correct`, `main._apply_obstacle`) now dispatch
through `BoundaryCondition.apply_immersion`, so staircase vs IBM is one flag.

It runs stably (200×80 Re=40 tfinal=20, 694 s, 42 ghost cells, weights sum to 1,
no blow-up), but it **does not validate**:

| mesh | method | Cd (steady) | Cl | Lr/D | vs lit Cd=1.52 / Lr=2.20 |
|---|---|---|---|---|---|
| 200×80 | staircase | 3.71 | −0.61 | 0.85 | Cd +144 %, Lr −61 % |
| 200×80 | **ibm** | **3.19** | **−0.57** | **0.91** | Cd +110 %, Lr −59 % |

Cd moved only 14 % (3.71→3.19) and `Lr/D` barely moved (0.85→0.91) — the wake is
still corner-pinned. **Why:** the IBM as built only enforces the *tangential*
no-slip at the true wall. The *normal* no-penetration is still imposed by
`BoundaryCondition.mask_solid_faces`, which zeroes the face flux at every
staircase solid/fluid face (OR of the two neighbours). That staircase
no-penetration is what the pressure Poisson sees, so the body the flow goes
around is still the jagged polygon — and for a bluff body the
no-penetration geometry, not the wall tangential velocity, sets the separation.
A fully-correct cell-centred ghost-cell IBM (Mittal/Uhlmann) must also move the
**no-penetration** to the true wall — either by interpolating the face flux at
the true boundary in the Poisson divergence (a wall-flux / cut-cell divergence
discretisation) or by the full-domain direct-forcing approach (no face masking,
force every body cell, re-force after projection). That is a substantial change
with real stability risk, not the small patch implemented here.

**Conclusion: the cylinder benchmark still does NOT validate.** Do NOT run
Re=100 (~5 h) or the full sweep — even with the current IBM they reproduce the
corner-pinned wake at high cost. `run_reynolds.py` remains the *harness* ready
to drive a future wall-flux IBM. The staircase + tangential-ghost IBM is a
demonstration of the limitation, not a validation.

The reported `div` in `history.csv` is the **collocated face-averaged divergence
residual** (`projection._divergence_residual`), i.e. the checkerboard content of
`p^{n+1}` driven to zero by the Poisson solve — **not** a mass-conservation
diagnostic. Values O(1)–O(10) early (startup spike) settling to O(1) are normal
for this gauge.

---

## 4. How to reproduce
```bash
# Validation subset (Re=40, 100 @ 400×160) — writes cylinder_report.md:
python examples/cylinder_flow/run_reynolds.py

# Mesh-independence study at Re=40 (the honest "does it converge?" check):
python examples/cylinder_flow/run_reynolds.py --mesh-study

# Single quick case (Re=40, 400×160):
python main.py examples/cylinder_flow/config.json

# Custom: Re=40 @ 400×160, tfinal=40, no report:
python examples/cylinder_flow/run_reynolds.py --cases 40 --mesh 400x160 --tfinal 40 --no-report
```
Cost note: 200×80 Re=40 tfinal=20 ≈ 11.5 min (2562 steps, ~0.27 s/step at 16 k
cells). 400×160 Re=40 tfinal=15 ≈ **2 h 26 min** (5561 steps, ~585 s/sim-s at 64 k
cells; in-memory frame history grows to ~1.1 GB). Do **not** extrapolate cost
linearly — the 400×160 run was ~8× slower than the 200×80 probe, not the ~4× a
cell-count scaling predicts (the per-step Poisson solve dominates and ILU scales
worse than linearly). The full sweep (incl. Re=1000 @ 800×320) is many hours and,
per §3, not worth running until IBM lands. **Estimate-first: do not launch
`--full` or the Re=100 case without an explicit go-ahead** — both reproduce the
staircase's wrong numbers at high cost.

`output_interval` must stay small (the runner sets `0.05`) so the Cl FFT and
the Cd time-mean have enough history rows. A large `output_interval` reproduces
a spurious "Cd=91" artifact: only the t=0.05 startup-spike row is sampled.

---

## 5. Known limitations & next steps
- **Staircase geometry (the blocker).** The cell-mask rasterisation pins
  separation at the staircase corners and gives a mesh-independently wrong wake
  for this bluff body (§3). Replace it with a curved-boundary IBM: impose no-slip
  at the *true* immersed wall distance (linearly extrapolated ghost velocity),
  and integrate forces on the true surface (`ds = r dθ`, not the staircase facet
  area). **This is required to validate the cylinder benchmark** — the runner,
  the force module, the Rhie-Chow coupling and the report writer are all ready to
  drive it; only the boundary forcing changes. **A cut-cell attempt on
  2026-07-27 hit a collocated-projection wall — see §6** for the full analysis
  and the staggered-like rearchitecting a stable+accurate cut-cell requires.
  The cut-cell code is left behind the `ibm_cut_cell` flag (OFF); the geometry
  kernel (`solver/cut_cell.py`) and the vf-free Poisson + flux-form face
  correction are verified and reusable.
- **Cylinder centre off the grid.** `x = 5.0` lands at cell 90.4 on 400×160,
  making the staircase lopsided (the steady `Cl = −0.65`). Either snap the
  centre to a cell centre / face or let IBM use the true centre — both remove
  the spurious lift. (`y = 2.05` lands exactly on a row boundary at 400×160, so
  the vertical asymmetry is already absent.)
- **Uniform mesh only.** "Local refinement" is currently uniform refinement, so
  the mesh study is expensive. A real AMR / nested mesh around the cylinder
  would make the high-Re cases affordable.
- **Re=1000** needs a fine boundary-layer mesh for accurate Cd regardless of IBM.
- **Steady-state stop.** No early-stop criterion; the runner runs to `tfinal`.
  A Cd-rate-based stop would cut the steady-case cost.
- **No SIMPLE.** The projection method is SIMPLE-ready (predict→solve→correct)
  but SIMPLE itself is not implemented; the cylinder case does not need it.

---

## 6. Cut-cell attempt (2026-07-27) — collocated projection wall

**Goal.** Replace the staircase no-penetration with a true-wall cut-cell
treatment so the bluff body separates at the true cylinder surface (the
staircase + tangential-ghost IBM alone only moved Cd 3.71→3.19, still far from
the 1.52 target; see §3). A cut-cell geometry kernel + aperture-weighted
pressure-Poisson was implemented behind the `ibm_cut_cell` flag.

**Status: stopped, flag left OFF.** The geometry and the Poisson operator are
sound, but the *collocated* velocity correction cannot realise the
divergence-free cut-cell face flux without either becoming unstable or
corrupting the physics. A stable + accurate cut-cell needs a staggered-like
rearchitecting (carry face fluxes as the primary state) — a multi-hour change
with risk to the VOF / single-phase paths, deferred by decision. The code
remains in place behind the flag for a future session; with `ibm_cut_cell:
false` the cylinder case runs the stable snap+ghost path unchanged.

### 6.1 What works (verified, keep)
- **`solver/cut_cell.py` — `CutCellGeometry`.** Per-cell fluid volume fraction
  `vf`, per-face aperture `ap`, `is_solid` (vf≤eps), `is_cut`, `has_curve`.
  Sub-cell sampling (sub=64); **band test** for cylinders
  (`|d−r| ≤ 0.5·diag`) catches edge-only crossings the corner-straddle test
  misses; small-cell kill restricted to `candidate & (vf<eps)`. Self-test:
  `solid_area = πr²` to **1.8e-5 / 7.7e-6 / 6.3e-6** at 200×80 / 400×160 /
  800×320, **perfect mirror symmetry** (sym_err=0) about the true centre,
  n_cut = 58 / 116 / 232. This kernel is correct and reusable.
- **Centre snap (`snap_obstacle_to_grid: true`).** Snaps the cylinder centre
  to the nearest grid line (multiple of h/2), written back into the obstacle
  dict so IBM/forces agree. Kills the spurious steady lift (Cl→0). **Kept ON**
  in the cylinder config — it is independent of the cut-cell flag and is a
  standalone improvement to the snap+ghost baseline.
- **`tools/verify_cut_cell_pressure.py`.** Algebraic checks (matrix clean, row-sum
  ~1e-13, conditioning, finite solves, one-step bounded). Keep as the
  cut-cell regression test.

### 6.2 The vf-free Poisson + flux-form face correction (algebraically exact)
The cut-cell Poisson operator and divergence were made **volume-fraction-free**
(aperture-only): `_cc_face_coeffs` returns `c·ap` (not `c·ap/vf`) and
`_cc_divergence` is `(ap·F₊ − ap·F₋)/dx` (not `/vf·dx`). Row-sum stays zero →
constant null space and mean-projection unchanged; conditioning bounded by
aperture ratios, not `1/vf` (the `c·ap/vf` form is ill-conditioned: vf as
small as ~0.02 → BiCGSTAB residual 5.16; the vf-free form → 0.19, diag
min/median 0.02→0.15).

The **flux-form face-flux correction** `F_{j+1/2} -= dt·(1/ρ)_f·(dp_{j+1}−dp_j)/dx`
(bare direct face difference — the aperture enters only through `D'` and the
operator, so no `ap`/`vf` factor in the correction itself) makes the cut-cell
divergence vanish **exactly**: `D'F² = D'F* − dt·A'·dp`, verified to **1.6e-14**
on a 60×30 cylinder mesh. The remaining residual is purely the linear-solve
residual (BiCGSTAB+ILU(0) stalls on the non-symmetric matrix; GMRES(50) is
*worse*, 32s vs 0.46s).

### 6.3 The wall: collocated cell-velocity recovery
The face fluxes are exactly divergence-free, but the framework carries
**cell-centred** velocities (the next step's divergence recomputes face fluxes
via `F = 0.5(u_L+u_R)`). Every way of turning the corrected face fluxes back
into cell velocities fails on this collocated grid:

| Correction variant | Result (80×32 first steps) |
|---|---|
| vf-scaled Poisson (`c·ap/vf`) + standard central-diff | ill-conditioned (residual 5.16), blowup |
| vf-free Poisson + **standard central-diff** everywhere | residual `O(1)` mass source in cut cells → **unstable**: max\|v\| grows 1.6→9.8 over 8 steps, adaptive dt → dt_min grind |
| vf-free Poisson + **cut-cell override** (`u_cut = 2·F_tgt − u_nb`) | velocity spikes (max\|v\|=7.4, div=44) → unstable |
| vf-free Poisson + **full flux-form recovery** (`u = ½(Fₓ[i]+Fₓ[i+1])`) | exact divergence (1e-14) **but** `(1/4,1/2,1/4)` smoother every step ≈ **2.4× physical viscosity** → effective Re 40→~13 → no separation, wrong physics |

The central-difference gradient is the adjoint of the *standard* (staircase)
divergence — consistent for full cells, not for the aperture-weighted cut-cell
divergence `D'`. There is no collocated cell-velocity recovery that is
simultaneously non-smoothing, non-spiking, and `D'`-consistent.

### 6.4 What a fix requires (deferred)
A stable + accurate cut-cell projection needs the **face flux as the primary
mass-conserving quantity** (staggered-like): predict face velocities, solve the
cut-cell Poisson, correct the face fluxes (the exact flux-form correction
above), and derive cell-centred velocities from the face fluxes only for
advection/output/forces. This means reworking `MomentumSolver` to predict on
face velocities and carrying `Fx, Fy` (and `Fz`) as state across steps — a
change to the core projection + face-flux bookkeeping that must not perturb the
VOF and single-phase paths. The exact flux-form correction in §6.2 is the
correct `D'`-adjoint and carries over unchanged; only the cell↔face state
handling changes. Start there.

### 6.5 Files touched (all behind the `ibm_cut_cell` flag, dormant when off)
- `solver/cut_cell.py` (new) — geometry kernel. **Reusable as-is.**
- `solver/pressure.py` — `set_cut_cell`, `cut_cell_active`, `_cc_face_coeffs`
  (vf-free), `_cc_divergence` (vf-free), branched `_matrix` / `solve`.
- `solver/projection.py` — `_correct` branches to `_correct_cut_cell` when
  active (currently the standard central-diff correction — the unstable
  variant; the override and full-recovery variants are documented in the
  docstring and were rejected per §6.3).
- `main.py` — `CutCellGeometry` wiring, `_snap_obstacle_centres`; when
  `ibm_cut_cell` is on, `bc.solid` becomes `cc.is_solid` (deep-solid ring) and
  the IBM mask uses it.
- `config/config_loader.py` — `snap_obstacle_to_grid` (kept ON), `ibm_cut_cell`
  (OFF in the cylinder config).
- `tools/verify_cut_cell_pressure.py` (new) — algebraic regression test.