# SI Units Compliance Report

**Framework:** CFDPy (CPU) and CFDPyGPU (GPU) finite-volume CFD frameworks
**Policy:** every dimensional quantity is represented, stored, computed, documented, visualised and exported exclusively in **coherent SI units**; all internal calculations are performed in SI; any non-dimensional case must be **explicitly documented**.
**Audit scope:** configuration, mesh, geometry, material properties, boundary/initial conditions, governing equations, solvers, post-processing, visualisation, CSV/Tecplot/HDF5 export, console/log output, documentation.
**Verdict:** **Compliant** after the corrections in §6. The solver was already dimensionally consistent in SI; the gaps were in *labelling* (exports, plots, console), *documentation* (some config fields) and *undeclared non-dimensional example cases*. All are now enforced by a single source of truth, [`units.py`](CFDPY/units.py), and an automatic validator that runs at every simulation start.

---

## 1. Executive summary

| Area | Before | After |
|---|---|---|
| Internal calculations | SI, dimensionally consistent | unchanged (SI) |
| Config parameter units | documented for most properties; missing on geometry/time/IC | **all** dimensional fields documented in SI |
| Tecplot export | bare names (`"Pressure"`, `"U"`) | `"Pressure (Pa)"`, `"U (m/s)"`, `"Alpha (-)"` |
| CSV export | `x,y,z,u,v,w,p,T,alpha` | `x (m),…,u (m/s),…,p (Pa),T (K),alpha (-)` |
| HDF5 export | no unit metadata | `units` attribute per dataset + `unit_system="SI (coherent)"` |
| Matplotlib axes / colour bars | `x`, `y`, bare titles | `x (m)`, `y (m)`, `Pressure (Pa)`, … |
| Console banner / summary | mixed | every dimensional value carries its SI unit |
| Automatic validation | none | `units.validate_config` runs at startup; reports non-SI / undocumented / hidden-Celsius |
| Non-dimensional examples | undeclared (policy violation) | declared `"nondimensional": true` + `reference_scales` |

**No numerics changed.** Every edit documents, labels, validates or declares — none converts a value silently.

---

## 2. The enforcement mechanism — `units.py`

A new module, [`CFDPY/units.py`](CFDPY/units.py) (mirrored to [`CFDPYGPU/units.py`](CFDPYGPU/units.py)), is the **single source of truth** for units:

* `QUANTITIES` — every physical quantity → (meaning, coherent SI unit).
* `DIMENSIONLESS` — the groups that must stay pure numbers (Re, Pr, Gr, Ra, Nu, Pe, Fo, Bi, St, Mach, Courant, VOF α, relative humidity, mass/mole fraction, Cd/Cl/Cp/Cf), rendered with an explicit `(-)`.
* `FIELD_UNITS` / `CONFIG_UNITS` — export columns and config parameters → SI unit.
* `label()`, `tecplot_varnames()`, `csv_header()` — the exporters and plots pull every label from here, so a unit is defined in exactly one place.
* `validate_config()` — the automatic consistency checker (§7).

Data labels use **parentheses** (`Velocity (m/s)`); prose docstrings use **brackets** (`[Pa]`).

---

## 3. Dimensional variable register (state fields)

The state fields written to every exporter and plotted by the viewer:

| # | Meaning | SI unit | Source file(s) | Variable | Documented | Correct | Inconsistency | Correction |
|---|---|---|---|---|---|---|---|---|
| 1 | Position x | m | `mesh/mesh.py`, exporters | `Xc` / `X` / `x` | ✅ | ✅ | export label missing | labelled `(m)` |
| 2 | Position y | m | idem | `Yc` / `Y` / `y` | ✅ | ✅ | export label missing | labelled `(m)` |
| 3 | Position z | m | idem | `Zc` / `Z` / `z` | ✅ | ✅ | export label missing | labelled `(m)` |
| 4 | Velocity u | m/s | `main.py`, `solver/*` | `u` / `U` | ✅ | ✅ | export label missing | labelled `(m/s)` |
| 5 | Velocity v | m/s | idem | `v` / `V` | ✅ | ✅ | export label missing | labelled `(m/s)` |
| 6 | Velocity w | m/s | idem | `w` / `W` | ✅ | ✅ | export label missing | labelled `(m/s)` |
| 7 | Pressure | Pa | `solver/pressure.py` | `p` / `Pressure` | ✅ | ✅ | export label missing | labelled `(Pa)` |
| 8 | Temperature | K | `solver/energy.py` | `T` / `Temperature` | ✅ | ✅ | export label missing | labelled `(K)` |
| 9 | VOF fraction | – (dimensionless) | `solver/vof.py` | `alpha` / `Alpha` | ✅ | ✅ | not marked dimensionless | labelled `(-)` |
| 10 | Time | s | `main.py` | `time` / `t` | ✅ | ✅ | HDF5 lacked unit | `time_units="s"` attr |
| 11 | Mesh spacing | m | `mesh/mesh.py` | `dx,dy,dz` | ✅ | ✅ | — | — |

Derived (post-processing, `visualization/postprocessor.py`):

| # | Meaning | SI unit | Variable | Notes |
|---|---|---|---|---|
| 12 | Vorticity ω = ∂v/∂x − ∂u/∂y | 1/s | `vorticity` | labelled in registry |
| 13 | Stream function ψ (2-D) | m²/s | `streamfunction` | labelled in registry |
| 14 | Nusselt number | – | `nusselt_wall` | dimensionless ✅ |
| 15 | Drag / lift / pressure / skin-friction coeff. (GPU) | – | `Cd,Cl,Cp,Cf` | dimensionless ✅ (`solver/forces.py`) |

---

## 4. Configuration-parameter register

Every dimensional `Config` field (`config/config_loader.py`) with its SI unit; all now carry an inline unit comment and are registered in `units.CONFIG_UNITS`:

| Parameter | Meaning | SI unit | Documented | Correct |
|---|---|---|---|---|
| `Lx, Ly, Lz` | domain lengths | m | ✅ (added) | ✅ |
| `Nx, Ny, Nz` | cell counts | – | ✅ | ✅ |
| `dt, tfinal, dt_min, dt_max` | times | s | ✅ (added) | ✅ |
| `output_interval` | output cadence | s | ✅ (fixed — see §6.2) | ✅ |
| `cfl_max` | Courant target | – | ✅ | ✅ |
| `rho, rho_light` | density | kg/m³ | ✅ | ✅ |
| `mu, mu_light` | dynamic viscosity | Pa·s | ✅ | ✅ |
| `cp` | specific heat | J/(kg·K) | ✅ | ✅ |
| `k` | thermal conductivity | W/(m·K) | ✅ | ✅ |
| `beta` | thermal expansion | 1/K | ✅ | ✅ |
| `sigma` | surface tension | N/m | ✅ | ✅ |
| `gravity` | gravitational accel. | m/s² | ✅ (added) | ✅ |
| `t_ref, t0` | temperatures | K | ✅ (added) | ✅ |
| `u0, v0, w0` | initial velocity | m/s | ✅ (added) | ✅ |
| `drop_x, drop_y, drop_r, pool_height` | drop/pool geometry | m | ✅ | ✅ |
| `alpha_value` | background VOF fraction | – | ✅ | ✅ |
| `linear_tol, poisson_tol` | relative residual tol. | – | ✅ | ✅ |

---

## 5. Governing-equation dimensional homogeneity

Checked by hand (the solver operates in acceleration/flux form, all SI):

| Equation | Form | Each term unit | Homogeneous |
|---|---|---|---|
| Continuity | ∇·u = 0 | 1/s | ✅ |
| Momentum | ∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + g | m/s² | ✅ |
| Energy | ∂T/∂t + (u·∇)T = α∇²T | K/s | ✅ |
| VOF | ∂α/∂t + ∇·(αu) = 0 | 1/s | ✅ |
| Boussinesq body force | f = −β(T−T₀)g | m/s² | ✅ |
| Pressure Poisson | ∇·((1/ρ)∇δp) = (1/Δt)∇·u* | 1/s² | ✅ |

The energy/momentum Dirichlet boundary term (`Lrhs`) is scaled by the full `dt·α` / `dt·ν` (fixed earlier this session), preserving dimensional consistency at both time levels.

---

## 6. Findings and corrections applied

**6.1 Missing SI labels on all exports and plots (critical).** Tecplot `VARIABLES`, the CSV header, HDF5 datasets, and every Matplotlib axis/colour bar carried no units. → All now labelled from `units.py`. HDF5 dataset **names are unchanged**, so the restart/checkpoint path is unaffected (verified: a run restarts correctly from an SI-tagged `.h5`).

**6.2 Documentation defect.** `output_interval` was commented "wall-time interval" but is measured in **simulation time**. → Corrected to "simulation-time between output frames [s]".

**6.3 Undocumented dimensional config fields.** `Lx/Ly/Lz`, `dt/tfinal/dt_min/dt_max`, `u0/v0/w0`, `t0/t_ref`, `gravity` had no unit annotation. → All annotated `[m]`, `[s]`, `[m/s]`, `[K]`, `[m/s²]`.

**6.4 Undeclared non-dimensional cases (policy violation).** Three example cases use unit-scaled (non-physical) properties:

| Case | Values | Scaling |
|---|---|---|
| `cylinder_flow` | ρ=1, U=1, D=1, μ=0.025 | Re = 1/μ = 40 |
| `backward_facing_step` | ρ=1, U=1, h=1, μ=0.01 | Re_h = 1/μ = 100 |
| `natural_convection_2D` | ρ=1, cp=1, μ=2.13e-3, k=3e-3 | Pr=0.71, Ra=10⁵ |

These are legitimate CFD non-dimensionalisations, but the strict policy requires them to be declared. → Each config now carries `"nondimensional": true` and a `"reference_scales"` block naming the length/velocity/density/ΔT scales and the resulting dimensionless group. The validator then accepts them as documented exceptions. `dam_break_2D` and `liquid_drop_splash_2D` remain full real-SI (water, ρ=1000).

---

## 7. Automatic validation

`units.validate_config(cfg)` runs at every simulation start (`Simulation._header`) and reports:

* **non-SI / non-dimensional cases** — properties outside any real-fluid SI range (e.g. `cp=1`), an **error** in `strict=True` mode unless the case declares `nondimensional: true`;
* **hidden non-SI values** — a `t0`/`t_ref` below 200 K (likely Celsius mistaken for kelvin);
* **undocumented dimensional parameters** — numeric fields with no unit in `CONFIG_UNITS`.

Result on the bundled examples: all 5 pass (2 real-SI clean, 3 declared non-dimensional). Example startup output:

```
VOF=off, Boussinesq=on, units=SI (coherent)
SI check: OK — all dimensional parameters are documented in SI.
```

Export/visualisation compliance is enforced structurally: because Tecplot, CSV, HDF5 and the plot labels **all** call `units.label()`, an exported or plotted dimensional quantity cannot appear without its SI unit.

---

## 8. Remaining recommendations

1. **Per-variable docstrings in solver internals.** The state fields and config are fully documented; a mechanical follow-up is to add `[unit]` to every intermediate-array docstring in `solver/*.py` and `numerics/*.py` (they are dimensionally correct today, just not all annotated).
2. **`history.csv` column units.** Column *keys* (`Nu`, `div`, `dt`, `umax`, …) are consumed by the restart pre-load as dict keys, so they were left unchanged to avoid breaking resume; their units are catalogued here and in `units.py`. If desired, a companion `history_units.json` sidecar can publish them without touching the keys.
3. **`strict=True` gate.** Consider running the validator in strict mode in CI so an *undeclared* non-dimensional case fails the build.
4. **Temperature offset caveat.** Kelvin is the coherent SI temperature; any future Celsius input must be converted at the config boundary (the validator already flags sub-200 K references).

---

## 9. Compliance checklist (mandatory rule)

For every dimensional variable in the framework:

- [x] a clearly defined SI unit — in `units.py`
- [x] consistent SI use throughout the solver — verified (§5)
- [x] documentation of the unit — config comments + `units.py` + this report
- [x] correct export with SI units — Tecplot / CSV / HDF5 (§3, §6.1)
- [x] correct visualisation with SI units — axes + colour bars (§6.1)
- [x] dimensional consistency with the governing equations — verified (§5)
- [x] non-dimensional cases explicitly documented — `nondimensional` + `reference_scales` (§6.4)

*Generated as part of the framework SI-compliance audit. Enforcement lives in `units.py`; this report is the human-readable record.*
