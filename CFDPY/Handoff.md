# Handoff — CFDPy liquid-drop splash example & restart capability

**Date:** 2026-07-26
**Scope:** Added the "Splash of a Liquid Drop" example, a HDF5-snapshot
restart/resume path, and a guard against the velocity-animation streamline
hang. Extended the splash case from `tfinal = 1.2 s` to `tfinal = 4.0 s` and
produced the full 0 → 4 s MP4 animations.

This document is the handoff for whoever picks up the work next.  The
authoritative description of the framework remains `README.md`; this file
records *what changed, why, and what to watch out for*.

---

## 1. What was done

### 1.1 New example: `examples/liquid_drop_splash_2D/`
The classic Harlow & Shannon (1967) free-surface benchmark.  A 1.0 × 1.6 m
closed no-slip tank (Nx=80, Ny=120) holds a water pool in the bottom 0.3 m and
a circular water drop (radius 0.10 m) at (0.5, 1.1) m.  Released from rest, the
drop free-falls, impacts the pool at t ≈ 0.37 s, and splashes; the run goes to
`tfinal = 4.0 s` to capture crown collapse, secondary jets, and settling.

### 1.2 New `alpha_init` mode: `"splash_drop"` (`main.py`, `_init_alpha`)
Builds a circular heavy-phase drop above a bottom liquid pool.  Geometry comes
from new config fields `pool_height`, `drop_x`, `drop_y`, `drop_r` (any
`<= 0` auto-resolves from the domain size).  Implemented in
`Simulation._init_alpha`; fields declared in `config/config_loader.py`.

### 1.3 Restart / checkpoint resume (`main.py`)
New `Simulation.restart_from(path)` loads `u,v,w,p,T,alpha` + time from any
saved HDF5 frame and resumes the time loop instead of calling `initialize()`.
`run()` now does `restart_from(cfg.restart)` when `cfg.restart` is set, else
`initialize()`.  On restart it also:

* continues per-frame output numbering past the highest existing frame (no
  overwrites);
* pre-loads the previously saved frames into the viewer and the `history.csv`
  rows into `self.history`, so the final animations + history span the whole
  run, not just the resumed leg;
* aligns the next output to the `output_interval` grid via the new
  `_next_output_time()` helper.

New config fields: `restart: str = ""`, `flow_streamlines: bool = True`.

### 1.4 Streamline-hang guard (`main.py` finalize)
`finalize()` now calls `animate_flow(..., with_streamlines=self.cfg.flow_streamlines)`.
matplotlib's `ax.streamplot` stalls indefinitely on the chaotic post-impact
velocity fields of VOF splash/dam-break cases; the flag lets those cases
disable it (speed + quiver overlay still rendered).  `liquid_drop_splash_2D`
and `dam_break_2D` configs set `"flow_streamlines": false`.

### 1.5 Extended run to 4.0 s
The 0 → 1.2 s leg was run first, then `tfinal` was raised to 4.0 s and the case
was resumed from `outputs/liquid_drop_splash_2D/frame_001242.h5` (t = 1.2003 s).
The resumed leg reached t = 4.0004 s cleanly (exit code 0), and `finalize()`
regenerated all four MP4s over the full 0 → 4 s frame set.

---

## 2. Files changed

| File | Change |
|------|--------|
| `main.py` | `splash_drop` branch in `_init_alpha`; new `restart_from`, `_preload_frames`, `_preload_history`, `_next_output_time`; `run()` resume branch; `finalize()` uses `flow_streamlines` |
| `config/config_loader.py` | new `Config` fields: `drop_x`, `drop_y`, `drop_r`, `pool_height`, `restart`, `flow_streamlines` |
| `examples/liquid_drop_splash_2D/config.json` | new example (splash_drop, tfinal=4.0, restart=frame_001242.h5, flow_streamlines=false) |
| `examples/dam_break_2D/config.json` | added `"flow_streamlines": false` |
| `README.md` | Example 4 section; Restart / checkpoint resume section; Velocity animation & streamline-hang section; new knobs in the "useful knobs" list; project tree entry |
| `Handoff.md` | this file |

No changes to the numerics, physics, or solver modules — the restart path and
the splash init are pure additions layered on the existing solvers.

---

## 3. Outputs produced (`outputs/liquid_drop_splash_2D/`)

| Artifact | Count / size |
|----------|--------------|
| `liquid_drop_splash_2D_alpha.mp4` | free-surface (VOF α), 9.90 s, 20 fps, 900×750 |
| `liquid_drop_splash_2D_p.mp4` | pressure, 9.90 s, 20 fps |
| `liquid_drop_splash_2D_T.mp4` | temperature, 9.90 s, 20 fps |
| `liquid_drop_splash_2D_velocity.mp4` | speed + quiver (no streamlines), 9.90 s, 20 fps, 1050×750 |
| `frame_XXXXXX.h5` / `.csv` / `.dat` | 197 field snapshots (60 from leg 1 + 137 from leg 2) |
| `T_*.png`, `p_*.png`, `vel_*.png` | 591 PNGs (3 per frame) |
| `history.csv` | 198 rows; leg-1 rows preloaded (blank `div`/`dt`), leg-2 rows complete |

**Mass conservation:** 0.3253 → 0.3197 over 0 → 4.0 s (1.7 % drift, interface
smearing only — no real mass loss).  Peak velocities `max|u| ≈ 1.48`,
`max|v| ≈ 3.53` m/s.

---

## 4. How to reproduce

```bash
# Fresh full run 0 -> 4.0 s (clear "restart" in the config first, ~55-60 min):
python main.py examples/liquid_drop_splash_2D/config.json

# Or, as it was actually done, in two legs:
#   leg 1: tfinal=1.2, restart=""      -> writes frame_000001..frame_001242.*
#   leg 2: tfinal=4.0, restart=".../frame_001242.h5"  -> resumes to 4.0 s
```

To start a case from scratch, set `"restart": ""` (or remove the key).  To
resume any case from any point, point `"restart"` at the desired `frame_*.h5`.

---

## 5. Known limitations & things to watch

1. **No surface tension.** `sigma` is declared in `Config` but never used by the
   solver (grep `sigma` → only the config default + dam_break).  The splash is
   the inviscid-capillary version; crown/jet shape is less crisp than a
   CSF-corrected solver would give.  Implementing a continuum-surface-force
   source in `solver/vof.py` + `momentum.py` is the natural extension point.

2. **`streamplot` is fragile on chaotic fields.** Even with
   `flow_streamlines: false` in `finalize()`, note that `Simulation._save_frame`
   still calls `plot_field(..., streamlines=True)` for every per-frame velocity
   PNG.  In both the 1.2 s and 4.0 s runs this never hung (single static frame,
   wrapped in `try/except`), but if a future case produces a truly pathological
   frame it could — consider gating `_save_frame`'s velocity-PNG streamlines on
   `flow_streamlines` too for full safety.

3. **Restart step-number gap in `history.csv`.** The preloaded leg-1 rows carry
   the artificial `step = 1..60` from the rebuilt history (the real per-frame
   step numbers were 1, 11, 21, … , 1242).  Leg-2 rows continue from the real
   `step_count = 1243…`.  So the `step` column has a cosmetic discontinuity at
   the restart boundary; the `t` column is continuous and correct.  Harmless,
   but if you want a clean `step` column, rebuild `history.csv` from the HDF5
   frames by filename index instead of preloading.

4. **Cost is strongly time-dependent (adaptive CFL).** Measured marginal cost
   for the splash case: ~350 wall-s/sim-s during free fall, ~600–800 at impact,
   **rising to ~2000 wall-s/sim-s** through the active splash before settling.
   The full 0 → 4.0 s run is ~55–60 min single-core.  A 0 → 1.2 s run is ~11
   min.  Budget accordingly; the cost is dominated by the post-impact leg where
   `dt` is smallest.

5. **`tfinal` is now 4.0 s in the committed config**, and `"restart"` points at
   `frame_001242.h5`.  Re-running as-is will resume from 1.2 s again (fine — the
   snapshot still exists).  For a clean from-scratch run, clear `"restart"`.

---

## 6. Suggested next steps

- Implement surface tension (CSF) so the splash crown/jet is physically crisp.
- Gate `_save_frame` velocity-PNG streamlines on `flow_streamlines` (see §5.2).
- Add a `--restart PATH` CLI flag in `main()` as a convenience over editing JSON.
- Consider writing a dedicated restart file (not reusing output frames) if
  snapshot frequency diverges from output frequency.
- Rebuild `history.csv` with a continuous `step` column if needed (§5.3).