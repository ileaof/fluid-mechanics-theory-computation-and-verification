# Fluid Mechanics: Theory, Computation, and Verification

## A Finite-Volume Approach with Python

> The official companion repository for the graduate-level textbook
> *Fluid Mechanics: Theory, Computation, and Verification — A Finite-Volume Approach with Python*.
> Every theoretical result in the book is paired with an executable Python program, an
> analytical solution, a finite-volume implementation, and a verification study — culminating
> in a complete educational CFD simulator, **CFDPy**, developed from scratch for this text.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-≥1.25-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-≥1.11-8CAAE6.svg)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-≥3.7-3776AB.svg)](https://matplotlib.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange.svg)](LICENSE)
[![Verification](https://img.shields.io/badge/Method-Verification--driven-success.svg)](#computational-philosophy)
[![CFDPy](https://img.shields.io/badge/CFD-CFDPy-orange.svg)](CFDPY/README.md)
[![CFDPyGPU](https://img.shields.io/badge/GPU-CFDPyGPU-76B900.svg)](CFDPYGPU/README.md)
[![CUDA](https://img.shields.io/badge/CUDA-Numba-76B900.svg)](CFDPYGPU/README.md)

---

## Table of Contents

- [Introduction](#introduction)
  - [Scientific motivation](#scientific-motivation)
- [Quick Start](#quick-start)
- [Main Capabilities](#main-capabilities)
- [Book Organization](#book-organization)
  - [Chapter 1 — Foundations of the Finite Volume Method: Plane Couette–Poiseuille Flow](#chapter-1--foundations-of-the-finite-volume-method-plane-couettepoiseuille-flow)
  - [Chapter 2 — Fluid Statics and the Hydrostatic Balance](#chapter-2--fluid-statics-and-the-hydrostatic-balance)
  - [Chapter 3 — Kinematics of Flow: Streamlines, Pathlines, and Vorticity](#chapter-3--kinematics-of-flow-streamlines-pathlines-and-vorticity)
  - [Chapter 4 — Integral Control-Volume Analysis](#chapter-4--integral-control-volume-analysis)
  - [Chapter 5 — Differential Analysis and the Navier–Stokes Equations](#chapter-5--differential-analysis-and-the-navierstokes-equations)
  - [Chapter 6 — Dimensional Analysis and Dynamic Similarity](#chapter-6--dimensional-analysis-and-dynamic-similarity)
  - [Chapter 7 — Viscous Flow in Pipes and Pipe Networks](#chapter-7--viscous-flow-in-pipes-and-pipe-networks)
  - [Chapter 8 — Boundary Layers and External Flow](#chapter-8--boundary-layers-and-external-flow)
  - [Chapter 9 — Compressible Flow](#chapter-9--compressible-flow)
  - [Chapter 10 — Turbulent Flow and Reynolds Averaging](#chapter-10--turbulent-flow-and-reynolds-averaging)
  - [Chapter 11 — Convection–Diffusion and Discretization Schemes](#chapter-11--convectiondiffusion-and-discretization-schemes)
  - [Chapter 12 — Code and Solution Verification](#chapter-12--code-and-solution-verification)
  - [Chapter 13 — Buoyant Flows and Free-Surface Problems](#chapter-13--buoyant-flows-and-free-surface-problems)
- [Python Programs](#python-programs)
- [CFD Simulator — CFDPy](#cfd-simulator--cfdpy)
- [GPU-Accelerated Simulator — CFDPyGPU](#gpu-accelerated-simulator--cfdpygpu)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Restarting a Simulation from a Checkpoint](#restarting-a-simulation-from-a-checkpoint)
- [Output Formats](#output-formats)
- [Rendered Animations Shipped with the Repository](#rendered-animations-shipped-with-the-repository)
- [Current Capabilities and Limitations](#current-capabilities-and-limitations)
- [Roadmap](#roadmap)
- [Computational Philosophy](#computational-philosophy)
- [Intended Audience](#intended-audience)
- [References](#references)
  - [Citation](#citation)
  - [BibTeX](#bibtex)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)
  - [Author and Contact](#author-and-contact)

---

## Introduction

This repository is the computational companion to the textbook
*Fluid Mechanics: Theory, Computation, and Verification — A Finite-Volume Approach with Python*.
It contains the complete, runnable body of code on which the book's exposition rests, and it is
designed so that no result in the text has to be taken on faith: every analytical solution,
every numerical method, and every verification claim can be reproduced by the reader with a
single command.

The repository contains:

- **all Python programs presented in the textbook**, organised chapter by chapter;
- **analytical solutions** — exact, closed-form results that serve as reference data;
- **numerical implementations** — finite-volume discretisations built from first principles;
- **verification studies** — systematic grid- and time-step-refinement campaigns reporting
  observed orders of accuracy, Richardson extrapolation, and the Grid Convergence Index;
- **computational projects** — benchmark problems (Blasius, lid-driven cavity, de Vahl Davis,
  Sod shock tube, dam break) solved to publication-grade standards; and
- **the complete educational CFD simulator**, **CFDPy**, a modular finite-volume framework that
  solves the incompressible Navier–Stokes equations with energy transport, Boussinesq buoyancy,
  and Volume-of-Fluid free surfaces on structured Cartesian meshes; and
- **a GPU-accelerated variant**, **CFDPyGPU**, that ports the finite-volume kernels and the
  pressure-Poisson Krylov solve to NVIDIA CUDA through Numba, with an automatic, numerics-identical
  fallback to the CPU path when no CUDA GPU is present.

The guiding principle of the book, and of this repository, is **verification-driven scientific
computing**. A numerical result is not accepted until it has been shown to converge to a known
answer at the design rate of accuracy. The Method of Manufactured Solutions, Richardson
extrapolation, and the Grid Convergence Index are used throughout — not as optional extras, but
as the ordinary working tools of the computational fluid dynamicist. Reproducibility is enforced
by construction: every program is deterministic (no random numbers), depends only on the standard
scientific Python stack, prints its convergence tables to the console, and writes its figure to
disk.

### Scientific motivation

Fluid mechanics is usually taught as two disconnected subjects: an analytical course in which
closed-form solutions are derived for a handful of idealised flows, and a computational course in
which a commercial or open-source solver is driven as a black box. The gap between them is where
most practical errors live. A student who can derive the Blasius profile but cannot tell whether a
solver has converged to it, and a practitioner who can mesh a geometry but cannot distinguish a
discretisation error from a modelling error, are both missing the essential skill of the field:
**knowing when a number can be trusted**.

This repository is built on three convictions.

1. **Theory and computation must be developed together.** Every governing equation in the book is
   discretised, implemented, and run in the same chapter in which it is derived, so the reader
   sees the continuous equation, the discrete operator, and the numerical answer as one object
   rather than three.

2. **A numerical result without a verification study is an opinion.** Grid convergence, observed
   order of accuracy, Richardson extrapolation, the Grid Convergence Index, and the Method of
   Manufactured Solutions are applied to every scheme in the text — including a deliberately
   planted coding bug in Chapter 12 that MMS catches and ordinary testing does not. This is the
   practice codified by ASME V&V 20, and it is treated here as elementary rather than advanced.

3. **A solver you cannot read is a solver you cannot verify.** The capstone simulator, CFDPy, is
   written from scratch in readable Python rather than wrapped around an existing framework, so
   every operator the book derives can be traced to the lines of code that implement it — and so
   that the reader can extend it. The GPU variant, CFDPyGPU, applies the same principle to
   hardware acceleration: each CUDA kernel is validated against the NumPy reference it replaces
   and benchmarked honestly before it is promoted into the production path.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification.git
cd fluid-mechanics-theory-computation-and-verification

# 2. Install the scientific Python stack (Python 3.11+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r CFDPY/requirements.txt

# 3. Run a chapter program — prints a verification table, writes a PNG
python chapter01/ex1_2_fvm.py

# 4. Run the CFD simulator on a bundled case (from inside CFDPY/)
cd CFDPY
python main.py examples/natural_convection_2D/config.json

# 5. Or run the GPU-accelerated variant — it falls back to the CPU path with
#    identical numerics when no NVIDIA GPU is present
cd ../CFDPYGPU
python main.py examples/natural_convection_2D/config.json
```

Chapter programs write their PNG figure into the **current working directory** and print their
convergence tables to the console. Simulator runs write every configured output into the case's
`output_dir` (`outputs/<name>/` by default). Full details in [Installation](#installation) and
[Output Formats](#output-formats).

---

## Main Capabilities

| Capability | Where | Notes |
|---|---|---|
| 39 standalone chapter programs (analytical / FVM / advanced) | `chapter01/`–`chapter13/` | NumPy + SciPy + Matplotlib only; deterministic; no build step |
| Verification machinery — observed order, Richardson extrapolation, GCI, MMS | throughout, formalised in `chapter12/` | The methodological backbone of the book |
| Classical benchmarks — Blasius, Ghia lid-driven cavity, de Vahl Davis, Sod shock tube, Ritter/Stoker dam break | `chapter08/`–`chapter13/` | Each verified against published reference data |
| Incompressible Navier–Stokes on 2D/3D structured Cartesian meshes | [`CFDPY/`](CFDPY/README.md) | Collocated finite volumes, incremental (Chorin) projection |
| Energy transport + Boussinesq natural convection | `CFDPY/solver/energy.py`, `CFDPY/physics/buoyancy.py` | Dirichlet / heat-flux / adiabatic walls |
| Volume-of-Fluid free surfaces | `CFDPY/solver/vof.py` | Linear property blending, interface-normal reconstruction |
| Selectable convection — upwind, central, QUICK, TVD (5 limiters) | `CFDPY/numerics/interpolation.py` | Numba-JIT limiter kernel with a pure-NumPy fallback |
| Krylov linear solvers — CG / BiCGSTAB / GMRES + ILU(0) | `CFDPY/solver/linear_solver.py` | Factorisation cached across steps for fixed matrices |
| Immersed obstacles (blocked cells, direct forcing) | `CFDPY/solver/boundary.py` | Boxes; cylinder/sphere primitives added in `CFDPYGPU/` |
| Force integration (Cd / Cl / Cp / Cf) and Rhie–Chow coupling | `CFDPYGPU/solver/forces.py`, `CFDPYGPU/solver/pressure.py` | Cylinder benchmark harness — see [Limitations](#current-capabilities-and-limitations) |
| Curved-boundary IBM and cut-cell geometry (experimental) | `CFDPYGPU/solver/ibm.py`, `CFDPYGPU/solver/cut_cell.py` | Opt-in flags; validation status documented honestly |
| NVIDIA CUDA acceleration via Numba (`@cuda.jit`) with automatic CPU fallback | [`CFDPYGPU/gpu/`](CFDPYGPU/README.md) | Validated kernels + GPU-resident BiCGSTAB; not yet in the production solve |
| Dual visualisation — Matplotlib (PNG, MP4/GIF) and Tecplot 360 ASCII | `CFDPY/visualization/` | Plus CSV history and HDF5 snapshots that double as restart checkpoints |

---

## Book Organization

The book is organised in thirteen chapters that move from the simplest one-dimensional viscous
flow to multiphysics, free-surface, and verification problems. Each chapter introduces the
theory, derives the governing equations, develops the finite-volume discretisation, verifies it
against an analytical or benchmark solution, and closes with an advanced engineering application.
The chapters are summarised below.

### Chapter 1 — Foundations of the Finite Volume Method: Plane Couette–Poiseuille Flow

**Objectives.** Establish the finite-volume method on the simplest problem in viscous fluid
mechanics — steady, fully developed flow between parallel plates — and introduce verification
through the Method of Manufactured Solutions.

**Theory.** The chapter opens with the continuum hypothesis, the Newtonian constitutive law, and
the reduction of the incompressible momentum equation to a one-dimensional balance when the flow
is fully developed. The plane Couette–Poiseuille problem, with a moving upper plate and an imposed
pressure gradient, admits the closed-form quadratic profile
*u(y) = U(y/H) + (1/2μ)(dp/dx)(y² − Hy)*, which becomes the canonical reference solution for the
remainder of the text.

**Governing equations.** The steady one-dimensional momentum balance
*d/dy(μ du/dy) = dp/dx* with Dirichlet wall conditions.

**Numerical techniques.** The cell-centred finite-volume method is built from scratch: control
volumes, face fluxes, central differencing of the diffusive term, half-cell treatment of
Dirichlet boundaries, and assembly of the tridiagonal coefficient matrix. The system is solved by
point Gauss–Seidel with under-relaxation, with the Thomas algorithm as an independent check.

**Python examples.** Example 1.1 evaluates the analytical profile and verifies the wall shear
stress and volume flow rate. Example 1.2 discretises the same problem by the finite-volume
method and shows second-order (here, round-off) recovery of the quadratic exact solution.
Example 1.3 verifies a *variable*-coefficient solver against a manufactured solution using the
harmonic mean of cell viscosities at faces.

**Engineering applications.** Journal bearings, lubrication flows, and the calibration of
finite-volume solvers for heat conduction and diffusion problems.

**Learning outcomes.** The reader can derive and implement a finite-volume discretisation,
assemble and solve the resulting linear system, and quantify its order of accuracy.

**Connection with later chapters.** The finite-volume machinery and the harmonic-mean practice
recur in Chapters 5, 7, 10, and 11, and the Method of Manufactured Solutions becomes the
verification backbone of Chapters 12 and 13.

### Chapter 2 — Fluid Statics and the Hydrostatic Balance

**Objectives.** Treat fluids at rest — pressure distribution, forces on submerged surfaces, and
the role of compressibility and density stratification — and verify the results numerically.

**Theory.** The chapter develops the hydrostatic equation *dp/dz = −ρg* from the stress tensor at
rest and applies it to incompressible fluids, to the compressible International Standard
Atmosphere, and to density-stratified fluids. Resultant hydrostatic forces and centres of
pressure are obtained for plane surfaces by integration of the pressure distribution.

**Governing equations.** The hydrostatic balance *dp/dz = −ρg*, closed by an equation of state
(*ρ = p/RT* for the atmosphere; a prescribed *ρ(z)* for stratified fluids).

**Numerical techniques.** Composite quadrature for the resultant force and centre of pressure;
conservative, second-order finite-volume marching of the hydrostatic balance with the
trapezoidal face rule; grid-refinement studies against the exact ISA solution.

**Python examples.** Example 2.1 computes the force on an inclined gate in closed form and
verifies it against direct numerical integration. Example 2.2 integrates the compressible
hydrostatic balance to recover the ISA pressure profile at second order. Example 2.3 treats a
linearly stratified fluid, where the pressure is quadratic in depth, and verifies the force and
centre of pressure on a vertical gate.

**Engineering applications.** Dams, gates, retaining structures, atmospheric and oceanographic
pressure profiles, and stratified reservoirs.

**Learning outcomes.** The reader can compute hydrostatic forces and centres of pressure, handle
variable density, and verify quadrature by convergence testing.

**Connection with later chapters.** Hydrostatics underpins the Boussinesq buoyancy of Chapter 13
and the gravity body force implemented in CFDPy.

### Chapter 3 — Kinematics of Flow: Streamlines, Pathlines, and Vorticity

**Objectives.** Develop the kinematic description of flow — the fields and curves used to
visualise motion — and quantify vorticity and circulation.

**Theory.** The chapter introduces the velocity field, the stream function, the velocity
potential, and the distinction between streamlines, pathlines, and streaklines in unsteady flow.
Vorticity is defined as the curl of velocity, and circulation is linked to vorticity through
Stokes' theorem. Plane stagnation-point flow illustrates irrotational straining, and the
Taylor–Green vortex provides a non-trivial rotational benchmark.

**Governing equations.** The kinematic identities *u = ∂ψ/∂y, v = −∂ψ/∂x*, the vorticity
*ω = ∂v/∂x − ∂u/∂y*, and Stokes' theorem relating circulation to the area integral of vorticity.

**Numerical techniques.** Second-order central differences for gradients and vorticity;
fourth-order Runge–Kutta integration of particle paths; verification of the discrete vorticity
field and circulation against exact closed forms.

**Python examples.** Example 3.1 analyses stagnation-point flow and verifies every kinematic
quantity by finite differences. Example 3.2 contrasts streamlines, pathlines, and streaklines of
an unsteady field and confirms fourth-order convergence of the pathline integrator. Example 3.3
verifies the vorticity and circulation of the Taylor–Green vortex two independent ways.

**Engineering applications.** Flow visualisation, particle tracking, and the interpretation of
experimental and computational flow fields.

**Learning outcomes.** The reader can distinguish the three flow-line families, compute
vorticity and circulation, and verify kinematic quantities against analytical benchmarks.

**Connection with later chapters.** Vorticity reappears in the vorticity-transport view of
Chapter 5, the lid-driven cavity of Chapter 11, and the cavity flow of Chapter 13.

### Chapter 4 — Integral Control-Volume Analysis

**Objectives.** Apply the integral mass, momentum, and energy conservation laws to
finite control volumes, and verify the closure of each balance.

**Theory.** The chapter develops the Reynolds transport theorem and the integral conservation
laws for mass, linear momentum, and mechanical energy. Bernoulli's equation is recovered for
inviscid, steady, incompressible flow along a streamline. The Torricelli draining tank, the
reducing bend, and the momentum and kinetic-energy correction factors for non-uniform profiles
are treated in detail.

**Governing equations.** The integral mass, momentum, and energy balances; Torricelli's law
*v_jet = √(2gh)*; the momentum-flux correction factor *β = (1/AV²)∫u² dA*.

**Numerical techniques.** Fourth-order Runge–Kutta integration of the draining-tank ODE;
numerical quadrature of the momentum and kinetic-energy correction factors for laminar pipe
flow, with second-order convergence to the exact values *β = 4/3* and *α = 2*.

**Python examples.** Example 4.1 solves the Torricelli problem in closed form and verifies it
against RK4. Example 4.2 computes the anchoring force on a reducing bend and confirms
machine-precision closure of the momentum balance. Example 4.3 verifies an unsteady tank
balance by the Method of Manufactured Solutions with both fixed-step RK4 and adaptive
embedded Runge–Kutta, reporting Richardson extrapolation and a Grid Convergence Index.

**Engineering applications.** Pipe fittings, jet engines, thrust on nozzles and bends, and
reservoir and tank dynamics.

**Learning outcomes.** The reader can select and apply integral control volumes, compute
anchoring forces, and verify that conservation balances close.

**Connection with later chapters.** The control-volume viewpoint motivates the finite-volume
discretisations of Chapters 5, 7, and 11 and the conservative flux form implemented in CFDPy.

### Chapter 5 — Differential Analysis and the Navier–Stokes Equations

**Objectives.** Derive the differential conservation laws and solve the Navier–Stokes equations
for unsteady, one- and two-dimensional viscous flows.

**Theory.** The chapter derives the incompressible Navier–Stokes equations from the differential
momentum balance and the constitutive law for a Newtonian fluid. Stokes' first problem (the
impulsively started plate) is solved by the similarity variable *η = y/(2√(νt))* and the
complementary-error-function profile. The Taylor–Green vortex is presented as an exact,
time-decaying two-dimensional Navier–Stokes solution and used to expose the vorticity-transport
equation.

**Governing equations.** The incompressible Navier–Stokes equations
*∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u*; the diffusion equation *∂u/∂t = ν∂²u/∂y²*; the vorticity-transport
equation *∂ω/∂t + (u·∇)ω = ν∇²ω*.

**Numerical techniques.** Cell-centred finite-volume discretisation in space with the
Crank–Nicolson θ-scheme in time; tridiagonal solves by the Thomas algorithm; simultaneous
space–time refinement to confirm second-order convergence; verification against the erfc and
Taylor–Green exact solutions.

**Python examples.** Example 5.1 verifies the erfc similarity solution and the self-similar
collapse of the profiles. Example 5.2 solves Stokes' first problem by an implicit
finite-volume/Crank–Nicolson scheme and refines to second order. Example 5.3 verifies the
Taylor–Green vortex as an exact Navier–Stokes solution and checks its vorticity-transport
equation.

**Engineering applications.** Startup of Couette flow, transient wall shear, and the
viscous-diffusion timescale of boundary layers.

**Learning outcomes.** The reader can discretise parabolic PDEs implicitly, handle
stability through Crank–Nicolson, and verify against similarity and manufactured solutions.

**Connection with later chapters.** This is the numerical template for the momentum solver in
CFDPy and for the boundary-layer and convection-diffusion chapters that follow.

### Chapter 6 — Dimensional Analysis and Dynamic Similarity

**Objectives.** Use dimensional analysis to reduce physical problems to their essential
dimensionless groups, and demonstrate dynamic similarity numerically.

**Theory.** The chapter presents the Buckingham Pi theorem and applies it to drag on a sphere,
recovering the drag coefficient and the Reynolds number. Dynamic similarity is then exhibited
through the Darcy friction factor *f(Re, ε/D)*, which collapses the pressure drop in rough pipes
of every size onto the Moody chart. The Colebrook equation is solved for the turbulent regime,
with the Hagen–Poiseuille result *f = 64/Re* from Chapter 1 recovered in the laminar limit.

**Governing equations.** The Buckingham Pi theorem; the Colebrook equation
*1/√f = −2 log₁₀[(ε/D)/3.7 + 2.51/(Re√f)]*; the plane-Poiseuille invariant *f·Re = 96*.

**Numerical techniques.** Construction of the dimensional matrix and computation of its null
space to obtain the Pi groups; bracketed root finding (Brent's method) for Colebrook; finite-volume
solution of plane Poiseuille flow to verify *f·Re = 96* across wildly different physical scales.

**Python examples.** Example 6.1 computes the Pi groups of sphere drag from the dimensional
matrix. Example 6.2 solves Colebrook and builds the Moody chart, comparing with Haaland's
explicit approximation. Example 6.3 verifies dynamic similarity: three flows at the same
Reynolds number give an identical *f·Re*, and mesh refinement drives it to 96 with Richardson
extrapolation and a GCI.

**Engineering applications.** Scale models, pipe sizing, and the rational design of experiments.

**Learning outcomes.** The reader can apply dimensional analysis, interpret similarity, and
verify dimensionless invariants computationally.

**Connection with later chapters.** Reynolds and Peclet numbers govern the convection schemes
of Chapter 11, the turbulence modelling of Chapter 10, and the benchmarks of Chapters 11 and 13.

### Chapter 7 — Viscous Flow in Pipes and Pipe Networks

**Objectives.** Treat fully developed laminar pipe flow and extend it to pipe networks and
pump–system interaction.

**Theory.** The chapter solves the Hagen–Poiseuille problem in cylindrical coordinates, deriving
the parabolic profile, the fourth-power flow-rate law, and the laminar friction relation
*f·Re = 64*. The finite-volume method is generalised to cylindrical coordinates with the
*r-weighted* face flux. The chapter then assembles individual pipes into a network: three
reservoirs joined at a common junction, with the junction head fixed by continuity and the
friction factor supplied by Colebrook. Pump–system operating points are found by intersecting
the pump curve with the system curve.

**Governing equations.** The cylindrical momentum balance
*(1/r)d/dr(rμ du/dr) = dp/dx*; the head-loss relation
*Δh = (fL/D + ΣK)Q²/(2gA²)*; junction continuity *ΣQ_k = 0*.

**Numerical techniques.** Cell-centred finite-volume discretisation in cylindrical coordinates
with axis and wall boundary fluxes; the Thomas algorithm; bracketed root finding for the
junction head with self-consistent friction factors.

**Python examples.** Example 7.1 verifies the Hagen–Poiseuille closed forms against numerical
quadrature. Example 7.2 solves the cylindrical finite-volume system and refines to second order.
Example 7.3 solves the three-reservoir network, checks a symmetric limiting case, and finds a
pump operating point.

**Engineering applications.** Water distribution, pump selection, and hydraulic network design.

**Learning outcomes.** The reader can discretise in cylindrical coordinates, solve nonlinear
networks, and verify continuity closure.

**Connection with later chapters.** Cylindrical discretisation informs the immersed-obstacle
and duct flows of CFDPy, and the friction-factor logic feeds the turbulence chapter.

### Chapter 8 — Boundary Layers and External Flow

**Objectives.** Analyse thin viscous layers near solid boundaries and inviscid external flow
about bodies.

**Theory.** The chapter derives Prandtl's boundary-layer equations and the Blasius similarity
solution for flow over a flat plate, obtaining the celebrated wall curvature
*f″(0) = 0.33206* and the integral parameters *δ/x*, *C_f*. It then turns to inviscid potential
flow: the source-panel method is developed for flow over a body, the exact surface pressure on an
ellipse is recovered, and d'Alembert's paradox — zero drag on a smooth body in ideal flow — is
demonstrated, with the circular cylinder obtained as the limiting case.

**Governing equations.** The Blasius equation *f‴ + ½ f f″ = 0*; the panel-method linear system
for tangential flow at surface control points; the ellipse pressure coefficient
*C_p = 1 − (q/U)²*.

**Numerical techniques.** Shooting with fourth-order Runge–Kutta and secant correction;
an independent iterative integral (Picard) method for Blasius that provides a second verification
path; the source-panel method with refinement and Richardson extrapolation.

**Python examples.** Example 8.1 solves Blasius by shooting. Example 8.2 solves it by the
iterative integral method and confirms *f″(0) → 0.33206* independently. Example 8.3 computes
potential flow over an ellipse by the source-panel method, verifies *C_p* against the exact
solution, and shows the net drag vanishes.

**Engineering applications.** Skin-friction drag estimation, aerofoil panel methods, and
inviscid-flow pre-design.

**Learning outcomes.** The reader can solve boundary-layer ODEs by shooting and by iteration,
and apply panel methods for external flow.

**Connection with later chapters.** Boundary layers motivate the turbulence modelling of
Chapter 10, and panel methods connect to the convection schemes and CFDPy applications.

### Chapter 9 — Compressible Flow

**Objectives.** Treat compressible inviscid flow — isentropic relations, normal shocks, and
unsteady shock-capturing.

**Theory.** The chapter develops the stagnation and area–Mach relations for isentropic flow and
uses them to analyse the converging–diverging nozzle, identifying the subsonic and supersonic
branches. Normal shocks are treated through the Rankine–Hugoniot relations, with the
stagnation-pressure loss and entropy rise quantified. The chapter closes with the unsteady
one-dimensional Euler equations and the Sod shock tube, whose exact Riemann solution — a left
rarefaction, a contact, and a right shock — is the canonical verification case for
shock-capturing schemes.

**Governing equations.** The isentropic and area–Mach relations; the Rankine–Hugoniot jump
conditions; the one-dimensional Euler equations
*∂ₜ[ρ, ρu, E] + ∂ₓ[ρu, ρu²+p, (E+p)u] = 0*.

**Numerical techniques.** Bracketed root finding for the area–Mach relation and the in-nozzle
shock position; Toro's exact Riemann solver for the Sod problem; a finite-volume
shock-capturing scheme with the Rusanov (local Lax–Friedrichs) flux, second-order SSP Runge–Kutta
time stepping, and MinMod slope limiting.

**Python examples.** Example 9.1 solves the converging–diverging nozzle on both branches.
Example 9.2 verifies the normal-shock relations against independent mass, momentum, and energy
balances and locates a shock inside a nozzle. Example 9.3 verifies a finite-volume
shock-capturing scheme against the exact Sod solution and measures its (sub-first-order)
L1 convergence at the discontinuities.

**Engineering applications.** Nozzles, diffusers, supersonic wind tunnels, and shock dynamics.

**Learning outcomes.** The reader can analyse compressible duct flow, apply jump conditions,
and verify shock-capturing schemes against exact Riemann solutions.

**Connection with later chapters.** The finite-volume flux formulation and limiters reappear
in the convection schemes of Chapter 11 and in the VOF and scalar transport of CFDPy; the
shock-tube methodology is revisited for the shallow-water dam break of Chapter 13.

### Chapter 10 — Turbulent Flow and Reynolds Averaging

**Objectives.** Introduce the Reynolds-averaged view of turbulence and close it with eddy-viscosity
models, verifying the discretisation by the Method of Manufactured Solutions.

**Theory.** The chapter derives the Reynolds-averaged Navier–Stokes equations and the Reynolds
stress, and closes them with Prandtl's mixing-length hypothesis. The universal law of the wall
is obtained — viscous sublayer, log-law, and buffer-layer blend — with the von Kármán constant
*κ = 0.41* and *B = 5.0*. The two-equation *k–ε* model is then presented, with the eddy viscosity
*νₜ = C_μ k²/ε* and its production, dissipation, and convective–diffusive transport terms.

**Governing equations.** The mixing-length eddy viscosity *νₜ = ℓ²|dU/dy|* with van Driest
damping; the *k–ε* transport equations and the mean-momentum equation, coupled through *νₜ*.

**Numerical techniques.** Cell-centred finite-volume discretisation with Picard linearisation
of the nonlinear eddy viscosity; truncation-error analysis of the coupled system by MMS; grid
refinement with Richardson extrapolation and a GCI.

**Python examples.** Example 10.1 builds the composite law-of-the-wall profile from Spalding's
implicit formula. Example 10.2 solves turbulent channel flow with a mixing-length closure and
verifies the computed profile against the law of the wall. Example 10.3 performs an
order-of-accuracy verification of the coupled *k–ε* discretisation by MMS, the gold standard for
turbulence-model code verification.

**Engineering applications.** Wall-bounded turbulent flows, pipe and channel friction, and the
verification of industrial RANS codes.

**Learning outcomes.** The reader can apply eddy-viscosity closures, linearise nonlinear
discretisations, and verify coupled systems by MMS.

**Connection with later chapters.** The RANS closure is the natural extension point of CFDPy's
`MomentumSolver`, and MMS verification is formalised in Chapter 12.

### Chapter 11 — Convection–Diffusion and Discretization Schemes

**Objectives.** Isolate the convection–diffusion mechanism — the heart of the Navier–Stokes
equations — and study the boundedness, accuracy, and stability of finite-volume schemes.

**Theory.** The chapter takes the steady one-dimensional convection–diffusion equation as the
model problem, derives its exact exponential solution, and introduces the cell Peclet number
*Pe_cell = ρu Δx/Γ* as the governing parameter of boundedness. Central differencing is shown to
lose boundedness for *Pe_cell > 2*, while upwind differencing is unconditionally bounded but only
first-order accurate, introducing false diffusion. The chapter closes with the lid-driven
cavity, the standard benchmark of incompressible CFD, solved in the vorticity–streamfunction
formulation to eliminate the pressure-velocity coupling difficulty.

**Governing equations.** The convection–diffusion equation *ρu dφ/dx = Γ d²φ/dx²*; the
vorticity–streamfunction system *∇²ψ = −ω*, *u∂ω/∂x + v∂ω/∂y = ν∇²ω*.

**Numerical techniques.** Central and upwind finite-volume schemes; measurement of observed
order of accuracy by grid refinement; successive over-relaxation for the streamfunction Poisson
equation; Thom's wall-vorticity formula; verification against the Ghia, Ghia & Shin (1982)
benchmark at *Re = 100*.

**Python examples.** Example 11.1 verifies the exact exponential profile and exposes the
boundary-layer structure. Example 11.2 contrasts central and upwind differencing and measures
their orders of accuracy. Example 11.3 solves the lid-driven cavity and verifies the centreline
velocity against the Ghia benchmark.

**Engineering applications.** The choice of convection scheme governs the robustness and
accuracy of every CFD simulation.

**Learning outcomes.** The reader can select convection schemes rationally, diagnose
boundedness failures, and verify against a standard CFD benchmark.

**Connection with later chapters.** The convection schemes and limiters (upwind, central,
QUICK, TVD) are exactly those implemented in CFDPy's `numerics/interpolation.py`; the cavity
benchmark is revisited with CFDPy's projection solver.

### Chapter 12 — Code and Solution Verification

**Objectives.** Make verification rigorous: measure observed order of accuracy, estimate
grid-converged values, and detect coding errors by the Method of Manufactured Solutions.

**Theory.** The chapter formalises solution verification. From three systematically refined
grids, the observed order of accuracy is measured without knowing the exact answer, the
leading error term is eliminated by Richardson extrapolation, and the numerical uncertainty is
bounded by Roache's Grid Convergence Index with a factor of safety. The Method of Manufactured
Solutions is presented as the gold standard of code verification: a smooth target field is
manufactured, substituted into the operator to obtain the source that makes it exact, and the
solver is required to recover it at its design order.

**Governing equations.** The error expansion *f_h = f_exact + C hᵖ*; the observed-order,
Richardson, and GCI formulae; the steady viscous Burgers equation
*u du/dx − ν d²u/dx² = S(x)* as the nonlinear Navier–Stokes surrogate.

**Numerical techniques.** Systematic grid refinement; Richardson extrapolation and GCI
computation; MMS truncation-error analysis; deliberate insertion of a subtle coding bug (a
first-order one-sided convective derivative) to show that MMS detects what ordinary testing
misses.

**Python examples.** Example 12.1 demonstrates the observed order and Richardson extrapolation
with the trapezoidal rule. Example 12.2 computes the GCI on the Chapter 11 model problem —
which has a known exact solution — and checks that it brackets the true error. Example 12.3
verifies a Burgers solver by MMS and shows that a single plausible line of code collapses the
order to one.

**Engineering applications.** Verification underpins every trustworthy CFD result and is
increasingly required by standards (ASME V&V 20).

**Learning outcomes.** The reader can verify code and solutions rigorously and quantify
numerical uncertainty.

**Connection with later chapters.** The verification tools developed here are applied
throughout CFDPy's example suite and in the free-surface and buoyancy benchmarks of Chapter 13.

### Chapter 13 — Buoyant Flows and Free-Surface Problems

**Objectives.** Couple the momentum and energy equations through buoyancy, and treat
free-surface flows by exact Riemann solutions and finite-volume shock-capturing.

**Theory.** The chapter opens with natural convection in a differentially heated vertical
channel, solved exactly under the Boussinesq approximation to obtain the cubic buoyant profile.
It then turns to the differentially heated square cavity — the de Vahl Davis benchmark — solved
in vorticity–streamfunction form and verified against the reference Nusselt number. Finally,
the shallow-water equations are introduced as a hyperbolic system analogous to the compressible
Euler equations of Chapter 9, and the dam-break Riemann problem is solved exactly
(Ritter/Stoker) and used to verify a finite-volume shock-capturing scheme.

**Governing equations.** The Boussinesq momentum balance
*ν d²u/dy² + gβ(T − T_m) = 0*; the Boussinesq vorticity–streamfunction–temperature system with
Rayleigh and Prandtl numbers; the shallow-water equations
*∂ₜh + ∂ₓ(hu) = 0*, *∂ₜ(hu) + ∂ₓ(hu² + gh²/2) = 0*.

**Numerical techniques.** Finite-volume marching of the Boussinesq system to steady state;
successive over-relaxation for the streamfunction; verification of the average Nusselt number
against de Vahl Davis (1983); bracketed root finding for the dam-break intermediate depth;
Rusanov-flux, second-order limited shock-capturing.

**Python examples.** Example 13.1 verifies the cubic buoyant channel profile. Example 13.2
solves the de Vahl Davis cavity and verifies *Nu = 2.238* at *Ra = 10⁴, Pr = 0.71*. Example 13.3
solves the shallow-water dam break exactly and verifies a shock-capturing scheme against it.

**Engineering applications.** Natural convection in enclosures, electronic cooling, and
free-surface / flood-wave hydraulics.

**Learning outcomes.** The reader can couple buoyancy to the flow field, run natural-convection
benchmarks, and verify shallow-water shock-capturing.

**Connection with later chapters.** This chapter is the direct theoretical preparation for
CFDPy's natural-convection, dam-break, and liquid-drop-splash examples, which solve the same
physics with the projection method on structured meshes.

---

## Python Programs

Every chapter of the book is accompanied by **three Python programs**, written to be read in
sequence. They follow a fixed pedagogical pattern that takes the reader from the closed-form
result, through its finite-volume discretisation, to a verification or engineering study.

| Example | Role | Educational purpose |
|---|---|---|
| `exN_1_analytical.py` | **Analytical solution** | Derive and evaluate the exact, closed-form result. Provides the reference data against which the numerical scheme is verified and the asymptotic structure (boundary layers, similarity) the scheme must reproduce. |
| `exN_2_fvm.py` | **Finite-volume implementation** | Discretise the same governing equation by the cell-centred finite-volume method, assemble and solve the linear system, and quantify the discretisation error by grid or time-step refinement. |
| `exN_3_advanced.py` | **Advanced engineering application / verification** | Tackle a harder, often nonlinear problem — a benchmark, a Method-of-Manufactured-Solutions verification, or a real engineering system — and report observed orders, Richardson extrapolation, and the GCI. |

The three examples are deliberately paired: **Example 2 always solves the same problem as
Example 1**, so the numerical and analytical answers can be compared directly; **Example 3**
extends the methodology to a setting where no closed-form solution is available, forcing the
reader to rely on verification alone. Together, the thirty-nine programs form a complete,
self-contained finite-volume course in Python.

All programs are deterministic (no random numbers) and depend only on the standard scientific
Python stack — **NumPy, SciPy and Matplotlib**. None of them requires Numba, h5py, tqdm or a GPU.
Each program prints its verification and convergence tables to the console and saves a single PNG
figure (`figN_M_<name>.png`) into the **current working directory**, using the non-interactive
Matplotlib `Agg` backend so the scripts run headless. They can be run individually:

```bash
python chapter05/ex5_2_fvm.py
python chapter13/ex13_2_fvm.py
```

> **Tip.** Because the figure is written to the working directory, `cd` into a scratch folder (or
> into the chapter directory) first if you would rather not scatter PNGs at the repository root.
> Generated `fig*.png` files are covered by [`.gitignore`](.gitignore).

---

## CFD Simulator — CFDPy

The repository ships with a complete educational and professional CFD framework, **CFDPy**,
written from scratch in pure Python (no OpenFOAM, FEniCS, or FiPy). CFDPy is the capstone of the
textbook: it gathers the finite-volume operators, convection schemes, time integrators, and
verification practices developed chapter by chapter into a single modular code that solves
realistic multiphysics problems.

**Simulator objectives.** To provide a readable, extensible finite-volume framework that
students can study line by line and researchers can extend — the architecture is explicitly
prepared for unstructured meshes, RANS/LES/DNS, GPU/CUDA, MPI, and AMR.

**Physical models and governing equations.** Incompressible flow (continuity + Navier–Stokes),
energy transport with Dirichlet / heat-flux / adiabatic walls, Boussinesq natural convection
*ρ = ρ₀(1 − β(T − T₀))*, forced convection, Volume-of-Fluid free-surface transport of the volume
fraction *α* with linear property blending and interface-normal reconstruction, generic scalar
advection–diffusion, and gravity as a body force.

**Numerical methods.** Collocated 2D/3D Cartesian finite volumes; incremental projection
(Chorin) pressure–velocity coupling with a variable-coefficient Poisson equation whose null space
is removed analytically (no row pinning, so the cached ILU factorisation stays valid);
second-order central diffusion; selectable convection — upwind, central, QUICK, and TVD
(vanleer / minmod / superbee / beam-warming / osher) with a Numba-JIT flux-limiter kernel and a
pure-NumPy fallback; implicit Euler and Crank–Nicolson time schemes with adaptive CFL; Krylov
solvers (CG, BiCGSTAB, GMRES) with optional ILU(0) preconditioning; immersed obstacles as
blocked cells with direct forcing.

**Software architecture.** A small, decoupled, SOLID/OOP package — `config`, `mesh`,
`numerics` (stateless operators), `physics`, `solver` (boundary, linear, momentum, pressure,
projection, energy, vof), and `visualization` — orchestrated by a single `Simulation` composition
root. Every parameter lives in a JSON/YAML case file; nothing is hard-coded.

**Supported physics & verification strategy.** Natural convection (Boussinesq cavity), dam
break, backward-facing step, and the Harlow & Shannon liquid-drop splash are provided as
ready-to-run examples and double as verification cases against the benchmarks developed in
Chapters 11 and 13. Verification relies on the grid-convergence, Richardson, and GCI machinery
of Chapter 12, applied to the same finite-volume operators CFDPy uses.

**Post-processing & output files.** Two independent visualisation systems — Matplotlib
(pressure/temperature/velocity contours, quiver, streamlines, VOF interface, MP4/GIF
animations) and Tecplot 360 ASCII `.dat` export — plus CSV, HDF5 snapshots (which also serve as
restart checkpoints), and PNG figures. The postprocessor computes vorticity, streamfunction, and
Nusselt numbers.

**Educational purpose & research applications.** CFDPy is both a teaching vehicle — every public
class and function carries a docstring with the relevant mathematics — and a research base whose
extension points (SIMPLE/PISO, unstructured meshes, RANS/LES/DNS, GPU, MPI, AMR, additional
multiphase/compressible/radiation/species models) are deliberately isolated.

> 📄 **The complete, authoritative documentation of the simulator — installation, case-file
> reference, mathematical formulation, the full solver flowchart, the project organisation, and
> the extension guide — is maintained in the simulator's own README:**
>
> **➡️ [`CFDPY/README.md`](CFDPY/README.md)** — please refer there for full details.

A quick way to run the four bundled examples — note the `cd`, which is required so that the
package imports (`config`, `mesh`, `numerics`, …) resolve:

```bash
cd CFDPY
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
```

> ⚠️ **Before running the liquid-drop splash case on a fresh clone.** Its shipped `config.json`
> sets `"restart": "outputs/liquid_drop_splash_2D/frame_001242.h5"` — the HDF5 checkpoint from
> which the case was extended from 1.2 s to 4.0 s. Runtime output frames are *not* tracked in git,
> so that file does not exist in a fresh clone and the run aborts when it tries to open it. Set
> `"restart": ""` in the case file for a from-scratch run. The same applies to
> [`CFDPYGPU/examples/liquid_drop_splash_2D/config.json`](CFDPYGPU/examples/liquid_drop_splash_2D/config.json).
> Once you have generated checkpoints of your own, see
> [Restarting a Simulation from a Checkpoint](#restarting-a-simulation-from-a-checkpoint) for the
> full resume procedure.

> 🖥️ **GPU variant.** A CUDA-accelerated port of CFDPy — **CFDPyGPU** — lives in
> [`CFDPYGPU/`](CFDPYGPU/README.md) and is documented in the next section. It shares the same
> finite-volume numerics and case files, ports the pressure-Poisson Krylov solve and the
> finite-volume kernels to NVIDIA CUDA via Numba, and falls back to this CPU implementation
> unchanged when no GPU is available.

---

## GPU-Accelerated Simulator — CFDPyGPU

**CFDPyGPU** is the GPU-accelerated variant of CFDPy. It mirrors the CPU simulator's package
layout (`config`, `mesh`, `numerics`, `physics`, `solver`, `visualization`, `examples`) and adds
a dedicated **`gpu/`** package that ports the performance-critical numerical kernels to NVIDIA CUDA.
The two variants share the same finite-volume discretisation, the same projection method, and the
same JSON case files, so a case that runs on CPU runs on GPU with no change to its `config.json`.

> 📄 **The complete GPU documentation — the layered `gpu/` package, the kernel design, the
> validation harnesses, and the honest CPU-vs-GPU benchmark tables — is maintained in the
> simulator's own README:**
>
> **➡️ [`CFDPYGPU/README.md`](CFDPYGPU/README.md)** — please refer there for full details.
> The measured profiling and benchmark numbers live in
> [`CFDPYGPU/GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md).

### GPU acceleration support

GPU acceleration is **opt-in and self-disabling**: a single `use_gpu` flag in the case file
(defaults to `True`) gates the GPU path, and the framework probes the hardware at startup and
prints a *CFDPy Hardware Report*. When a CUDA-capable NVIDIA GPU is available the simulation runs
on the device; when it is not — or when `use_gpu: false` — the framework silently falls back to the
original pure-NumPy/SciPy CPU code path with **identical numerics**. A CPU-only machine never pays
any GPU import cost: the CUDA backend and kernels are imported lazily, only when a GPU is actually
in use.

### CUDA implementation

The GPU work is layered so that the CPU/GPU choice is made in one place and the surrounding solver
code is device-agnostic:

- **`gpu/hardware.py`** — the single source of truth for "is there a usable NVIDIA GPU?". Probes
  device attributes (name, compute capability, SM count, memory, warp size), and queries the
  CUDA driver/runtime versions through the bundled `cudart64` DLL when locatable. Conservative and
  side-effect free: it only *reads* attributes and never allocates device memory or creates a
  context that would disturb a later run.
- **`gpu/backend.py`** — a small array/device backend with two implementations of the same
  interface: `NumPyBackend` (the original CPU path and the fallback) and `CUDABackend`
  (Numba-CUDA device arrays living in Numba's per-context memory pool, so repeated
  `zeros`/`empty` of the same shape reuse cached allocations instead of round-tripping through
  `cudaMalloc`/`cudaFree`). `get_backend()` returns the CUDA backend when `use_gpu` is true *and* a
  GPU was detected, else the NumPy backend; the result is cached process-wide.
- **`gpu/kernels.py`** — low-level `@cuda.jit` kernels: a sparse CSR matrix-vector product
  (`matvec_csr`, one thread per row, grid-stride, race-free without atomics) and the BLAS-1
  reductions a Krylov driver is built from (`dot`, `dot2`, `max_abs`, `norm2`, and the in-place
  `copy`/`axpy`/`scale_add`/`fill`/`div_pointwise`). The two-level shared-memory tree reduction
  writes one partial sum per block (no float atomics, deterministic) and reduces the partials on the
  device, so only **one float crosses device→host per reduction**; `dot2` fuses two dot products
  into a single host sync (the BiCGSTAB *ω*-step needs `(t,s)` and `(t,t)` together).
- **`gpu/linear.py`** — `GPUBiCGSTAB`, a GPU-resident preconditioned BiCGSTAB (van der Vorst) with an
  optional **Jacobi (diagonal) preconditioner** — the only preconditioner cheap on a GPU (a
  pointwise divide) and free of the sequential triangular solves that make ILU awkward on a device.
  Workspace is allocated once per solver instance and reused across solves; the inverse diagonal is
  cached per matrix identity and rebuilt only when the matrix changes. The convergence test uses the
  cheap recurrence residual but verifies with the *true* residual `b − A x` on exit, to catch the
  BiCGSTAB phantom-convergence stop, and the residual norm is only evaluated every 4 iterations
  (`check_every = 4`) to cut sync overhead.

Validation harnesses (`gpu/validate_kernels.py`, `gpu/validate_linear.py`) confirm that every GPU
kernel matches its NumPy reference to round-off (matvec exact, reductions ~1e-15) and that the GPU
BiCGSTAB solution agrees with the CPU solution to L2 ~1e-6 / rel∞ ~1e-7 on the real production
pressure-Poisson operator.

### NVIDIA GPU requirements and supported hardware

- An **NVIDIA CUDA-capable GPU** (any compute capability supported by the installed Numba build;
  developed and benchmarked on a GeForce RTX 4050 Laptop GPU, compute capability 8.9, 6 GB).
- The **NVIDIA CUDA driver** installed on the system (the runtime is provided either by a
  system CUDA Toolkit or, conveniently, by the `nvidia-*` pip wheels — see below).
- **Numba ≥ 0.58 built with CUDA support** (`numba.cuda`). Numba is the *only* additional
  dependency the GPU path requires — there is **no** dependence on CuPy, on CUDA Python, or on a
  hand-written extension module. The `@cuda.jit` kernels and the device backend are pure Numba.

Non-NVIDIA GPUs (AMD ROCm, Intel oneAPI, Apple Metal) are **not** supported by this path; they fall
back to the CPU implementation automatically.

### Parallel execution strategy

The kernels follow a deliberately simple, structured-stencil-friendly model:

- **One thread per row / per cell**, with grid-stride loops so a single launch covers any problem
  size. The sparse matvec assigns each row to exactly one thread (race-free accumulation, no
  atomics); the reductions use a 256-thread block (a power of two, so the shared-memory tree
  reduction is exact) with a grid-stride first pass and a single-block final reduce.
- **No inter-block coupling**, so the 2-D stencil operators (gradient, divergence,
  face-interpolate) slated for the roadmap map to natural 2-D CUDA grids with shared-memory halos
  and need no host↔device transfer during a cycle.
- **Fields stay resident**: the device memory pool and the per-solver workspace cache keep
  allocations alive across steps, eliminating repeated `cudaMalloc`/`cudaFree` and the per-solve
  host↔device copies that would otherwise dominate at small *N*.
- **Multi-GPU / MPI extension point**: the backend carries a `device_index` and activates its
  context on first use, so the future domain-decomposition path can call
  `init_backend(device_index = local_rank)` once per MPI rank — one rank, one GPU — with no change
  to the rank-local kernels.

### GPU dependencies

| Component | Purpose | Required for GPU? |
|---|---|---|
| `numba` ≥ 0.58 (with CUDA) | `@cuda.jit` kernels + device-array backend | yes (GPU path) |
| NVIDIA CUDA driver | device context + runtime | yes (GPU path) |
| CUDA runtime (`cudart64`) | queried for the version report only | optional |
| `nvidia-cuda-runtime-cu12` pip wheel | ships `cudart64` without a system CUDA Toolkit | optional (convenient) |

No CuPy, no CUDA Python (`nvidia-*` bindings), and no compiled extension are needed.

### Installation (GPU path)

The GPU path adds only Numba (already an optional CPU dependency) on top of the standard stack:

```bash
cd CFDPYGPU
pip install -r requirements.txt          # numpy / scipy / matplotlib / tqdm / h5py / numba
```

To obtain the CUDA runtime without installing a system-wide CUDA Toolkit, the `nvidia-*` pip
wheels bundle `cudart64` and are auto-discovered by `gpu/hardware.py`:

```bash
pip install nvidia-cuda-runtime-cu12     # optional; only needed for the version report
```

There is **no build step** and **no separate GPU install** — clone the repository, install the
Python dependencies, and run. If no NVIDIA GPU is detected, the run simply prints
`Execution Device : CPU` and proceeds on the CPU path.

### Example usage

```bash
cd CFDPYGPU
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
python main.py examples/cylinder_flow/config.json         # Re-sweep + mesh study + report
```

Force the CPU path on a GPU-equipped machine (for validation / debugging):

```json
{ "...": "...", "use_gpu": false }
```

Print only the hardware report:

```bash
python -c "from gpu import print_hardware_report; print_hardware_report()"
```

### Additional solver modules in the GPU variant

Beyond the `gpu/` package, `CFDPYGPU/` also carries three solver modules that the CPU variant does
not have. They were developed for the cylinder benchmark and are documented in full in
[`CFDPYGPU/Handoff_Cylinder.md`](CFDPYGPU/Handoff_Cylinder.md):

- **`solver/forces.py`** — `ForcesCalculator`: integrates pressure and viscous traction over the
  fluid/solid interface of an immersed body to give `Cd`, `Cl`, and the per-facet `Cp` / `Cf`
  surface distributions, plus the recirculation length and separation angle. Enabled with
  `"compute_forces": true`.
- **`solver/ibm.py`** — `IBMForcing`: a mirror-point ghost-cell immersed-boundary treatment that
  places the tangential no-slip at the *true* curved wall instead of at the staircase cell face
  (Mittal et al. 2008; Uhlmann 2005). Opt-in via `"immersed_method": "ibm"`; staircase direct
  forcing remains the default.
- **`solver/cut_cell.py`** — `CutCellGeometry`: per-cell fluid volume fractions and per-face
  aperture fractions for a curved body, computed by deterministic sub-cell sampling with
  small-cell stabilisation. The geometry kernel is verified (solid area recovers *πr²* to ~1e-5
  with exact mirror symmetry), but the aperture-weighted projection built on top of it is **not
  enabled** — it sits behind `"ibm_cut_cell": false` pending a staggered-like rearchitecting of
  the face-flux state.

The GPU variant also adds **Rhie–Chow** momentum interpolation (`"rhie_chow": true`), a
pressure-outlet Dirichlet path, obstacle centre snapping (`"snap_obstacle_to_grid"`), and
cylinder/sphere obstacle primitives alongside the axis-aligned boxes of the CPU version.

### Relationship between the CPU and GPU versions

CFDPyGPU is a **superset fork** of CFDPy: `mesh/`, `numerics/`, `physics/` and `visualization/`
are identical to the CPU package; `config/`, `main.py` and four `solver/` modules carry the
GPU-variant additions above; and the new `gpu/` package is layered on top. The two therefore share
the same governing equations, finite-volume operators, projection method, and JSON case format.
The GPU path is being promoted incrementally
— profile, port one hotspot, validate against the CPU, benchmark, and only then wire it into the
production solver. The GPU kernels and `GPUBiCGSTAB` are validated but, per this methodology, are
**not yet wired into the production `PressureSolver`** until the planned geometric-multigrid
preconditioner makes the GPU solve a net win in *both* the single-phase and VOF regimes; until then
the production solver runs on the CPU. See
[`CFDPYGPU/GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md) for the honest
CPU-vs-GPU benchmark tables and the incremental GPU roadmap.

---

## Repository Structure

The repository is organised into **per-chapter program directories** (`chapter01`–`chapter13`),
each holding the three Python examples of that chapter, and two simulator packages — **`CFDPY/`**
(CPU) and **`CFDPYGPU/`** (GPU-accelerated superset) — each decomposing into `config`, `mesh`,
`numerics`, `physics`, `solver`, `visualization`, `examples`, and `outputs`. The canonical
top-level layout and the role of each part are described below.

| Directory / file | Purpose |
|---|---|
| `chapter01/` … `chapter13/` | The **chapters** of the book, each with three runnable Python programs (`exN_1_analytical.py`, `exN_2_fvm.py`, `exN_3_advanced.py`). |
| `CFDPY/` | The **complete CFD simulator** (the `cfd` core), with its own README, requirements, and example cases. |
| `CFDPY/Handoff.md` | Developer handoff notes: the liquid-drop splash example, the HDF5 restart/resume path, and the Tecplot dialect migration. |
| `CFDPY/examples/` | **Ready-to-run example cases** (natural convection, dam break, backward-facing step, liquid-drop splash), each with a `config.json`. |
| `CFDPY/numerics/` | The **stateless finite-volume operators** (`src` of the simulator): interpolation, gradients, divergence, Laplacian, adaptive time step, Numba kernels. |
| `CFDPY/solver/` | The **solver core**: boundary conditions, linear solvers, momentum, pressure, projection, energy, and VOF. |
| `CFDPY/physics/` | Fluid properties, materials, gravity, and Boussinesq buoyancy. |
| `CFDPY/mesh/` | The Cartesian structured collocated mesh (2D/3D). |
| `CFDPY/config/` | The case-file loader and `Config`/`BoundarySpec` dataclasses. |
| `CFDPY/visualization/` | Matplotlib viewer, Tecplot exporter, and postprocessor. |
| `CFDPY/outputs/` | **Verification / output data** written at runtime: PNG, CSV, Tecplot `.dat`, HDF5 snapshots, MP4/GIF, and run logs. |
| `CFDPYGPU/` | The **GPU-accelerated variant of CFDPy** (Numba-CUDA kernels + a GPU-resident BiCGSTAB, with automatic CPU fallback), mirroring the CPU package layout and adding a `gpu/` package. Has its own README, requirements, and example cases. |
| `CFDPYGPU/gpu/` | The **GPU backend**: hardware detection, NumPy/CUDA array backend, `@cuda.jit` BLAS-1 + sparse-matvec kernels (`kernels.py`), and the GPU-resident preconditioned BiCGSTAB (`linear.py`). |
| `CFDPYGPU/solver/` | The solver core **plus** the GPU-variant additions: `forces.py` (Cd/Cl integration), `ibm.py` (mirror-point ghost-cell IBM), and `cut_cell.py` (cut-cell geometry, currently dormant). |
| `CFDPYGPU/examples/` | Ready-to-run GPU cases — natural convection, dam break, backward-facing step, liquid-drop splash, plus **cylinder flow** with a Reynolds-sweep / mesh-study driver and literature benchmark table. |
| `CFDPYGPU/GPU_PERFORMANCE_REPORT.md` | The profiling, CPU-vs-GPU benchmark tables, and the incremental GPU roadmap. |
| `CFDPYGPU/Handoff.md` | Same splash / restart / Tecplot handoff notes as the CPU variant. |
| `CFDPYGPU/Handoff_Cylinder.md` | Cylinder benchmark handoff: the force-integration and Rhie–Chow work, the staircase failure analysis, the ghost-cell IBM result, and the halted cut-cell attempt. |
| `CFDPYGPU/profile_hotspots.py` | cProfile per-step hotspot driver (I/O disabled) — produces the rankings in the performance report. |
| `CFDPYGPU/verify_cut_cell_pressure.py` | Algebraic regression checks for the cut-cell Poisson operator. |
| `LICENSE` | Creative Commons Attribution-NonCommercial 4.0 International, full legal text. |
| `.gitattributes` / `.gitignore` | Line-ending normalisation and binary-file marking; ignore rules for `__pycache__`, virtual environments, chapter figures, and regenerable simulator output. |

The repository has **no** `docs/`, `notebooks/`, `tests/`, `tools/` or `verification/` directory
at present; the documentation lives in the READMEs and handoff notes listed above. Adding a test
suite and consolidated verification campaigns is tracked in the [Roadmap](#roadmap).

### Directory tree

```
fluid-mechanics-theory-computation-and-verification/
├── README.md                          # This file — the book companion README
├── LICENSE                            # CC BY-NC 4.0 full legal text
├── .gitattributes                     # LF normalisation + binary markings
├── .gitignore                         # __pycache__, venvs, figures, runtime output
├── chapter01/                         # Foundations of the FVM — Couette–Poiseuille
│   ├── ex1_1_analytical.py
│   ├── ex1_2_fvm.py
│   └── ex1_3_advanced.py
├── chapter02/                         # Fluid statics and the hydrostatic balance
│   ├── ex2_1_analytical.py
│   ├── ex2_2_fvm.py
│   └── ex2_3_advanced.py
├── chapter03/                         # Kinematics — streamlines, pathlines, vorticity
│   ├── ex3_1_analytical.py
│   ├── ex3_2_fvm.py
│   └── ex3_3_advanced.py
├── chapter04/                         # Integral control-volume analysis
│   ├── ex4_1_analytical.py
│   ├── ex4_2_fvm.py
│   └── ex4_3_advanced.py
├── chapter05/                         # Differential analysis — Navier–Stokes
│   ├── ex5_1_analytical.py
│   ├── ex5_2_fvm.py
│   └── ex5_3_advanced.py
├── chapter06/                         # Dimensional analysis and dynamic similarity
│   ├── ex6_1_analytical.py
│   ├── ex6_2_fvm.py
│   └── ex6_3_advanced.py
├── chapter07/                         # Viscous flow in pipes and pipe networks
│   ├── ex7_1_analytical.py
│   ├── ex7_2_fvm.py
│   └── ex7_3_advanced.py
├── chapter08/                         # Boundary layers and external flow
│   ├── ex8_1_analytical.py
│   ├── ex8_2_fvm.py
│   └── ex8_3_advanced.py
├── chapter09/                         # Compressible flow
│   ├── ex9_1_analytical.py
│   ├── ex9_2_fvm.py
│   └── ex9_3_advanced.py
├── chapter10/                         # Turbulent flow and Reynolds averaging
│   ├── ex10_1_analytical.py
│   ├── ex10_2_fvm.py
│   └── ex10_3_advanced.py
├── chapter11/                         # Convection–diffusion and discretization schemes
│   ├── ex11_1_analytical.py
│   ├── ex11_2_fvm.py
│   └── ex11_3_advanced.py
├── chapter12/                         # Code and solution verification
│   ├── ex12_1_analytical.py
│   ├── ex12_2_fvm.py
│   └── ex12_3_advanced.py
├── chapter13/                         # Buoyant flows and free-surface problems
│   ├── ex13_1_analytical.py
│   ├── ex13_2_fvm.py
│   └── ex13_3_advanced.py
├── CFDPY/                             # The complete educational CFD simulator
│   ├── README.md                      #   authoritative simulator documentation
│   ├── Handoff.md                     #   splash example, restart & Tecplot handoff notes
│   ├── main.py                        #   CLI entry point + Simulation orchestrator
│   ├── requirements.txt               #   pinned Python dependencies
│   ├── config/                        #   case-file loader (JSON/YAML)
│   │   └── config_loader.py
│   ├── mesh/                          #   Cartesian structured collocated mesh
│   │   └── mesh.py
│   ├── numerics/                      #   stateless finite-volume operators
│   │   ├── interpolation.py           #     upwind / central / QUICK / TVD face values
│   │   ├── numba_kernels.py           #     @njit TVD limiter (NumPy fallback)
│   │   ├── gradients.py
│   │   ├── divergence.py
│   │   ├── laplacian.py
│   │   └── timestep.py                #     CFL / Fourier adaptive dt
│   ├── physics/                       #   fluid, materials, gravity, buoyancy
│   │   ├── fluid.py
│   │   ├── material.py
│   │   ├── gravity.py
│   │   └── buoyancy.py
│   ├── solver/                        #   boundary, linear, momentum, pressure,
│   │   ├── boundary.py                #     projection, energy, vof
│   │   ├── linear_solver.py
│   │   ├── momentum.py
│   │   ├── pressure.py
│   │   ├── projection.py
│   │   ├── energy.py
│   │   └── vof.py
│   ├── visualization/                 #   matplotlib, tecplot, postprocessor
│   │   ├── matplotlib_view.py
│   │   ├── tecplot_writer.py
│   │   └── postprocessor.py
│   ├── examples/                      #   ready-to-run cases
│   │   ├── natural_convection_2D/config.json
│   │   ├── dam_break_2D/config.json
│   │   ├── backward_facing_step/config.json
│   │   └── liquid_drop_splash_2D/config.json
│   └── outputs/                       #   runtime output: PNG / CSV / .dat / HDF5 / MP4
│       ├── backward_facing_step/Backward_Facing_Step_Tecplot.wmv   (tracked)
│       ├── dam_break_2D/dam_TecPlot.wmv                            (tracked)
│       └── liquid_drop_splash_2D/liquid_drop_splash_TecPlot.wmv    (tracked)
└── CFDPYGPU/                          # GPU-accelerated variant (Numba-CUDA, CPU fallback)
    ├── README.md                      #   authoritative GPU-simulator documentation
    ├── GPU_PERFORMANCE_REPORT.md      #   profiling + CPU-vs-GPU benchmarks + roadmap
    ├── Handoff.md                     #   splash example, restart & Tecplot handoff notes
    ├── Handoff_Cylinder.md            #   cylinder benchmark, staircase / IBM / cut-cell status
    ├── main.py                        #   CLI entry point + Simulation orchestrator
    ├── requirements.txt               #   pinned Python dependencies (numba = CUDA path)
    ├── profile_hotspots.py            #   cProfile per-step hotspot driver
    ├── verify_cut_cell_pressure.py    #   algebraic checks for the cut-cell Poisson
    ├── config/                        #   case-file loader (JSON/YAML) + use_gpu flag
    │   └── config_loader.py
    ├── gpu/                           #   the GPU acceleration package (layered)
    │   ├── hardware.py                #     detection + startup hardware report
    │   ├── backend.py                 #     NumPy / CUDA array backend (auto-detect)
    │   ├── kernels.py                 #     @cuda.jit BLAS-1 + sparse CSR matvec
    │   ├── linear.py                  #     GPU-resident preconditioned BiCGSTAB
    │   ├── validate_kernels.py        #     CPU-vs-GPU kernel validation
    │   └── validate_linear.py         #     GPU BiCGSTAB vs CPU on the real operator
    ├── mesh/                          #   identical to the CPU package
    ├── numerics/                      #   identical to the CPU package
    ├── physics/                       #   identical to the CPU package
    ├── visualization/                 #   identical to the CPU package
    ├── solver/                        #   CPU solver core + GPU-variant additions:
    │   ├── forces.py                  #     Cd / Cl / Cp / Cf surface integration
    │   ├── ibm.py                     #     mirror-point ghost-cell IBM (opt-in)
    │   └── cut_cell.py                #     cut-cell geometry (dormant, flag OFF)
    ├── examples/                      #   ready-to-run cases
    │   ├── natural_convection_2D/config.json
    │   ├── dam_break_2D/config.json
    │   ├── backward_facing_step/config.json
    │   ├── liquid_drop_splash_2D/config.json
    │   └── cylinder_flow/             #   Re sweep + mesh study + report
    │       ├── config.json
    │       ├── run_reynolds.py        #     sweep driver + cylinder_report.md writer
    │       ├── benchmarks.py          #     literature comparison table
    │       └── _probe.py              #     ad-hoc Cd/Cl trajectory probe
    └── outputs/                       #   runtime output: PNG / CSV / .dat / HDF5 / MP4
        ├── natural_convection_2D/     #     *_T.mp4, *_p.mp4                 (tracked)
        ├── dam_break_2D/              #     *_T.mp4, *_p.mp4, *_alpha.mp4    (tracked)
        ├── backward_facing_step/      #     *_T.mp4, *_p.mp4, *.wmv          (tracked)
        └── liquid_drop_splash_2D/     #     *_T/_p/_alpha/_velocity.mp4      (tracked)
```

Only the **rendered animation deliverables** (`.mp4`, `.wmv`) are tracked under `outputs/`; every
other runtime artefact (PNG frames, CSV history, Tecplot `.dat`, HDF5 snapshots, run logs) is
regenerable and is ignored by [`.gitignore`](.gitignore). See
[Rendered Animations Shipped with the Repository](#rendered-animations-shipped-with-the-repository).

---

## Installation

### Python requirements

The chapter programs require **Python 3.11+** and the standard scientific Python stack. Both
simulators (CFDPy and CFDPyGPU) are developed and verified on Python 3.11 and are compatible with
3.12+; they use `from __future__ import annotations` and modern type-hint syntax throughout.

### Dependencies

| Library | Purpose | Required? |
|---|---|---|
| `numpy` | Array numerics | yes |
| `scipy` | Sparse matrices, Krylov solvers, ILU, special functions | yes |
| `matplotlib` | PNG plots and MP4/GIF animations | yes |
| `tqdm` | Progress bar | optional |
| `h5py` | HDF5 snapshot output (skipped gracefully if missing) | optional |
| `numba` | JIT-accelerated TVD flux limiter (pure-NumPy fallback); **CUDA backend for CFDPyGPU** (`@cuda.jit` kernels) | optional |
| NVIDIA CUDA GPU + driver | GPU acceleration in `CFDPYGPU/` (auto-falls back to CPU when absent; no CuPy / CUDA Python needed) | optional |
| `pyyaml` | YAML case files (JSON always works) | optional |
| `meshio` | Unstructured-mesh I/O (future extension) | optional |
| `pyvista` | Optional interactive 3D viewer | optional |

A working **ffmpeg** binary on the system `PATH` enables MP4 animations; if ffmpeg is absent,
CFDPy automatically falls back to a pillow-written GIF.

### Installing the dependencies

Clone the repository and install the runtime stack. The simulator's pinned dependencies live in
[`CFDPY/requirements.txt`](CFDPY/requirements.txt);
[`CFDPYGPU/requirements.txt`](CFDPYGPU/requirements.txt) declares the same set — Numba is already
in it, and on the GPU variant that same Numba install provides the `@cuda.jit` backend, so no
extra package is needed for GPU support:

```bash
git clone https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification.git
cd fluid-mechanics-theory-computation-and-verification

# core + recommended extras
pip install -r CFDPY/requirements.txt
```

or, equivalently:

```bash
pip install numpy scipy matplotlib tqdm h5py numba
# optional extras (not required to run the examples)
pip install pyyaml meshio pyvista
```

There is **no build step** — the repository is pure Python.

### Virtual environment (recommended)

A dedicated virtual environment keeps the scientific stack isolated:

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (Git Bash / PowerShell)
python -m venv .venv
.venv\Scripts\activate

pip install -r CFDPY/requirements.txt
```

### GPU acceleration (optional)

The **CFDPyGPU** variant (see [GPU-Accelerated Simulator — CFDPyGPU](#gpu-accelerated-simulator--cfdpygpu))
adds GPU acceleration on top of the same stack — only **Numba with CUDA support** plus an
**NVIDIA CUDA GPU and driver** are required (no CuPy, no CUDA Python, no compiled extension):

```bash
pip install -r CFDPYGPU/requirements.txt        # same core stack; numba provides @cuda.jit
# optional: CUDA runtime via pip wheel (no system CUDA Toolkit needed)
pip install nvidia-cuda-runtime-cu12
```

If no NVIDIA GPU is detected, CFDPyGPU prints `Execution Device : CPU` and runs the original CPU
path with identical numerics. See [`CFDPYGPU/README.md`](CFDPYGPU/README.md) and
[`CFDPYGPU/GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md) for the kernel design,
validation, and benchmark tables.

### Running the chapter examples

Each chapter program is a standalone script. Run any of them directly:

```bash
python chapter01/ex1_2_fvm.py
python chapter08/ex8_1_analytical.py
python chapter12/ex12_2_fvm.py
```

Each program prints its convergence tables to the console and writes one PNG figure into the
**current working directory**.

### Running the CFD simulator

From the `CFDPY/` directory so that the package imports (`config`, `mesh`, `numerics`, …)
resolve:

```bash
cd CFDPY
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
```

Each run prints a header and a progress bar and writes all configured outputs to the case's
`output_dir` (`outputs/<name>/` by default). See [`CFDPY/README.md`](CFDPY/README.md) for the
full case-file reference and the complete set of options.

The GPU variant takes exactly the same command line from its own directory, and additionally
prints a *CFDPy Hardware Report* at startup naming the execution device:

```bash
cd CFDPYGPU
python main.py examples/cylinder_flow/config.json

# hardware report only, no simulation
python -c "from gpu import print_hardware_report; print_hardware_report()"
```

---

## Restarting a Simulation from a Checkpoint

Long runs do not have to be completed in one sitting. **Every HDF5 frame the simulator writes is
a valid restart checkpoint** — there is no separate checkpoint file format and no extra
configuration to enable one. This is how the liquid-drop splash case was extended from 1.2 s to
4.0 s without recomputing the first leg.

### What a checkpoint contains

Each `frame_%06d.h5` in the case's `output_dir` stores the complete cell-centred state — `u`, `v`,
`w`, `p`, `T` and `alpha` — the mesh coordinates `X`, `Y`, `Z`, and the simulation time as an HDF5
attribute. That is everything `Simulation.restart_from()` needs to continue the time loop.

### Step-by-step procedure

**1. Make sure checkpoints are being written.** Checkpoints are the HDF5 output, so the case must
have `save_hdf5` enabled and `h5py` installed. The write cadence is `output_interval`, measured in
simulation time:

```json
{
    "save_hdf5": true,
    "output_interval": 0.05,
    "output_dir": "outputs/my_case"
}
```

A smaller `output_interval` gives more restart points at the cost of more disk. If `h5py` is
missing the run still completes, but it writes no checkpoints and cannot be resumed.

**2. Run the case as usual, and let it stop** — whether it finishes, you interrupt it, or the
machine goes down. The frames written up to that point remain on disk:

```bash
cd CFDPY
python main.py examples/my_case/config.json
```

**3. Pick the checkpoint to resume from.** List the frames in the case's output directory and take
the last one (highest number = latest simulation time):

```bash
ls outputs/my_case/frame_*.h5 | tail -5
```

To confirm the simulation time a given frame holds before committing to it:

```bash
python -c "import h5py; print(h5py.File('outputs/my_case/frame_001242.h5').attrs['time'])"
```

**4. Point the case file at that checkpoint.** Set the `"restart"` key in `config.json`. **The path
is resolved relative to the directory you launch `main.py` from** — that is `CFDPY/` (or
`CFDPYGPU/`), not the case folder — so the usual form mirrors `output_dir`:

```json
{
    "restart": "outputs/my_case/frame_001242.h5"
}
```

**5. Extend `tfinal` past the checkpoint time.** The time loop runs `while time < tfinal`, so if
`tfinal` is still the value the checkpoint already reached, the run will load the state and exit
immediately without stepping. To continue a 1.2 s checkpoint to 4.0 s:

```json
{
    "restart": "outputs/my_case/frame_001242.h5",
    "tfinal": 4.0
}
```

**6. Re-run the same command.** No CLI flag is involved — the `"restart"` key alone selects the
resume path over a fresh `initialize()`:

```bash
python main.py examples/my_case/config.json
```

**7. Clear `"restart"` when you want a clean run from *t* = 0.** An empty string (the default)
starts from the initial condition:

```json
{ "restart": "" }
```

### What the framework does on resume

| Step | Behaviour |
|---|---|
| **Load state** | Reads `u, v, w, p, T, alpha` and the recorded time; seeds the time loop from that time rather than from zero. |
| **Re-apply boundaries** | Velocity BCs and immersed-obstacle direct forcing are re-applied to the loaded field, exactly as `initialize()` does for a fresh one. |
| **Continue numbering** | Scans `frame_*.h5` in `output_dir` and continues output numbering past the highest existing index, so **no previously written file is overwritten**. |
| **Align the output grid** | The next output lands on the next multiple of `output_interval` at or after the loaded time, keeping the frame cadence consistent across the join. |
| **Pre-load history** | Previously saved frames are loaded into the viewer and the existing `history.csv` rows into the run history, so the **final animations and history table span the whole run** (0 → `tfinal`), not just the resumed leg. |

### Requirements and caveats

- **`h5py` must be installed** — it is in [`CFDPY/requirements.txt`](CFDPY/requirements.txt) under
  the recommended extras. Without it there are no checkpoints to resume from.
- **The mesh must match.** The fields are loaded into the new run's arrays without a shape check,
  so `Nx`, `Ny`, `Nz` (and the domain size) must be identical to the run that wrote the
  checkpoint. Resuming onto a different mesh will fail or produce meaningless results — it is not
  an interpolating restart.
- **A missing path is fatal, not a fallback.** If `"restart"` names a file that does not exist the
  run aborts when it tries to open it; it does *not* silently start from scratch. This is why a
  fresh clone must clear the `"restart"` key shipped in the liquid-drop splash case (see
  [CFD Simulator — CFDPy](#cfd-simulator--cfdpy)).
- **All frames in `output_dir` are pre-loaded, including any later than your restart point.** If
  you are deliberately rewinding to redo a leg, delete the frames after the chosen checkpoint
  first, or the discarded ones will still appear in the final animations and history.
- **Physics and numerics settings may be changed on resume** (`tfinal`, `output_interval`,
  `cfl_max`, the convection scheme, and so on) — only the mesh is fixed. Changing the physics
  mid-run is legitimate for staged setups but makes the joined history non-uniform, so record what
  you changed.
- **The GPU variant behaves identically**: checkpoints are interchangeable between `CFDPY/` and
  `CFDPYGPU/` for the same mesh, since both write the same HDF5 layout.

---

## Output Formats

Both simulators write every enabled output on the same cadence, controlled by `output_interval`
(measured in simulation time) and the `save_*` booleans in the case file. Everything lands in the
case's `output_dir`.

| Format | File pattern | Enabled by | Contents |
|---|---|---|---|
| **PNG frames** | `T_%06d.png`, `p_%06d.png`, `vel_%06d.png` | `save_png` | Temperature / pressure / speed contours with velocity quiver, streamline and VOF-interface overlays |
| **MP4 / GIF animation** | `<name>_T.mp4`, `<name>_p.mp4`, `<name>_alpha.mp4`, `<name>_velocity.mp4`, `<name>_vorticity.mp4` | `save_mp4` | Rendered in `finalize()`; falls back to a pillow-written GIF when ffmpeg is unavailable |
| **CSV** | `history.csv`, `frame_%06d.csv` | `save_csv` | `history.csv` carries the per-frame time, `dt`, CFL, divergence residual, mean Nusselt number, and `Cd`/`Cl` when `compute_forces` is on; the numbered files are per-frame field dumps |
| **Tecplot 360 ASCII** | `frame_%06d.dat` | `save_tecplot` | One `ZONETYPE=ORDERED` / `DATAPACKING=POINT` zone per step with `STRANDID` + `SOLUTIONTIME`; variables `X Y Z U V W Pressure Temperature Alpha` |
| **HDF5 snapshot** | `frame_%06d.h5` | `save_hdf5` | `u, v, w, p, T, alpha` + simulation time — **also a valid restart checkpoint** (`"restart": "<path>"`) |
| **Force diagnostics** | `<name>_forces.png`, `<name>_cl_fft.png` | `compute_forces` | `Cd(t)` / `Cl(t)` histories and the windowed FFT of the lift signal (GPU variant) |
| **Cylinder report** | `examples/cylinder_flow/cylinder_report.md` | `run_reynolds.py` | Per-case Cd, Cl_rms, Strouhal, recirculation length and separation angle against the literature table (GPU variant) |

A working **ffmpeg** binary on `PATH` enables MP4; otherwise the animation is written as a GIF.
HDF5 output is skipped gracefully when `h5py` is not installed.

Chapter programs are simpler: they print their tables to the console and write one PNG into the
current working directory.

---

## Rendered Animations Shipped with the Repository

Runtime output is regenerable and therefore ignored by git, with one deliberate exception: the
**rendered animations** are tracked so the results can be viewed without running a simulation
first. Fifteen animation files ship with the repository (≈32 MB in total).

| Case | Tracked animations |
|---|---|
| Natural convection (GPU variant) | [`natural_convection_2D_T.mp4`](CFDPYGPU/outputs/natural_convection_2D/natural_convection_2D_T.mp4), [`natural_convection_2D_p.mp4`](CFDPYGPU/outputs/natural_convection_2D/natural_convection_2D_p.mp4) |
| Dam break (GPU variant) | [`dam_break_2D_alpha.mp4`](CFDPYGPU/outputs/dam_break_2D/dam_break_2D_alpha.mp4), [`dam_break_2D_T.mp4`](CFDPYGPU/outputs/dam_break_2D/dam_break_2D_T.mp4), [`dam_break_2D_p.mp4`](CFDPYGPU/outputs/dam_break_2D/dam_break_2D_p.mp4) |
| Liquid-drop splash (GPU variant) | [`liquid_drop_splash_2D_alpha.mp4`](CFDPYGPU/outputs/liquid_drop_splash_2D/liquid_drop_splash_2D_alpha.mp4), [`…_velocity.mp4`](CFDPYGPU/outputs/liquid_drop_splash_2D/liquid_drop_splash_2D_velocity.mp4), [`…_T.mp4`](CFDPYGPU/outputs/liquid_drop_splash_2D/liquid_drop_splash_2D_T.mp4), [`…_p.mp4`](CFDPYGPU/outputs/liquid_drop_splash_2D/liquid_drop_splash_2D_p.mp4) |
| Backward-facing step (GPU variant) | [`backward_facing_step_T.mp4`](CFDPYGPU/outputs/backward_facing_step/backward_facing_step_T.mp4), [`backward_facing_step_p.mp4`](CFDPYGPU/outputs/backward_facing_step/backward_facing_step_p.mp4) |
| Tecplot 360 renderings (`.wmv`) | [dam break](CFDPY/outputs/dam_break_2D/dam_TecPlot.wmv), [liquid-drop splash](CFDPY/outputs/liquid_drop_splash_2D/liquid_drop_splash_TecPlot.wmv), [backward-facing step](CFDPY/outputs/backward_facing_step/Backward_Facing_Step_Tecplot.wmv) (and a copy under `CFDPYGPU/`) |

The `.wmv` files are Tecplot 360 renderings of the exported `frame_*.dat` series; the `.mp4`
files are produced directly by the Matplotlib viewer in `finalize()`.

---

## Current Capabilities and Limitations

This repository documents what it does *and* what it does not do. The following limitations are
current and deliberate; each is recorded in more detail in the linked document.

**Solver capabilities (verified).** Incompressible Navier–Stokes with energy transport, Boussinesq
buoyancy and VOF free surfaces on 2D/3D structured collocated Cartesian meshes; incremental
projection with a variable-coefficient Poisson solve whose null space is removed analytically;
upwind / central / QUICK / TVD convection; implicit Euler and Crank–Nicolson time integration with
adaptive CFL; CG / BiCGSTAB / GMRES with cached ILU(0); immersed obstacles by blocked cells; HDF5
checkpoint/restart; Matplotlib and Tecplot output. The natural-convection, dam-break,
backward-facing-step and liquid-drop-splash examples run end to end.

**Limitations.**

- **The cylinder benchmark does not validate.** Rasterising a circle into the cell mask pins flow
  separation at the staircase's 90° corners. At Re = 40 the steady drag is `Cd ≈ 3.65` against the
  literature value `1.52`, and the error is *mesh-independent* (3.71 at 200×80 → 3.65 at 400×160),
  so it is not slow convergence. The mirror-point ghost-cell IBM (`"immersed_method": "ibm"`)
  moves it only to `Cd ≈ 3.19`, because it corrects the tangential no-slip while no-penetration is
  still imposed on the staircase faces. Treat this example as a demonstration of the staircase
  limitation, not as a literature match. Full analysis in
  [`CFDPYGPU/Handoff_Cylinder.md`](CFDPYGPU/Handoff_Cylinder.md) §3.
- **The cut-cell path is dormant.** The geometry kernel is verified, and the aperture-weighted
  Poisson plus flux-form face correction makes the cut-cell divergence vanish to ~1e-14 — but no
  *collocated* cell-velocity recovery from those face fluxes is simultaneously stable,
  non-smoothing and consistent with the cut-cell divergence. The work is left behind
  `"ibm_cut_cell": false` pending a staggered-like rearchitecting that carries face fluxes as
  primary state ([`Handoff_Cylinder.md`](CFDPYGPU/Handoff_Cylinder.md) §6).
- **The GPU kernels are not yet in the production solve.** `gpu/kernels.py` and the
  `GPUBiCGSTAB` in `gpu/linear.py` are validated against the CPU reference (matvec exact,
  reductions ~1e-15, full solve agreeing to L2 ~1e-6), but they are deliberately **not** wired
  into `PressureSolver`: with Jacobi preconditioning the GPU is 1.30× faster than CPU-without-ILU
  at N = 64 000 yet only 0.25× as fast as CPU-with-ILU. Promotion waits on the multigrid
  preconditioner ([`GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md) §4.3).
- **No automated test suite.** Verification is performed by the chapter programs and by the
  standalone harnesses (`gpu/validate_kernels.py`, `gpu/validate_linear.py`,
  `verify_cut_cell_pressure.py`); there is no `pytest` suite and no CI workflow yet.
- **No surface tension.** VOF runs with `sigma = 0`; the splash crown and jet are therefore
  qualitatively correct but not physically crisp. A CSF model is the natural addition.
- **No turbulence model in the simulator.** RANS closures are developed and verified in Chapter 10
  as standalone programs, but `CFDPy`'s `MomentumSolver` is laminar; the closure is an identified
  extension point, not an implemented feature.
- **Structured uniform Cartesian meshes only.** No local refinement, AMR, unstructured meshes,
  MPI domain decomposition, or SIMPLE/PISO coupling — all are documented extension points.
- **Mass conservation in VOF is approximate.** The liquid-drop splash conserves mass to ~1.7 %
  over a 4 s run; this is interface smearing, not a leak.

---

## Roadmap

Ordered roughly by expected impact. The GPU items follow the incremental
profile → implement → validate → benchmark → promote-only-on-success methodology set out in
[`CFDPYGPU/GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md) §5.

| # | Item | Status |
|---|---|---|
| 1 | **GPU geometric-multigrid preconditioner** — V-cycle with red-black Gauss–Seidel smoothing; converges in O(1) iterations and is the single change that makes the GPU solve a net win in *both* the single-phase and VOF regimes | next |
| 2 | **Wire the GPU Poisson solve into the production `PressureSolver`**, keeping the CPU path as the automatic fallback | blocked on 1 |
| 3 | **Keep fields GPU-resident across the whole step**, removing the per-solve host↔device copies | blocked on 2 |
| 4 | **GPU stencil operators** (gradient, divergence, face interpolation) so a step runs without a single host transfer | blocked on 3 |
| 5 | **GPU momentum / energy diffusion solves**, reusing the multigrid preconditioner | blocked on 1, 3 |
| 6 | **Wall-flux / staggered cut-cell immersed boundary** — the change required to make the cylinder benchmark validate | designed, not started |
| 7 | **Surface tension (CSF)** for the VOF free-surface cases | not started |
| 8 | **Automated test suite + CI workflow** — unit tests for the finite-volume operators, regression tests for the example cases | not started |
| 9 | **RANS closure in `MomentumSolver`**, promoting the Chapter 10 mixing-length and *k–ε* models into the simulator | not started |
| 10 | **SIMPLE / PISO coupling**, unstructured meshes (via `meshio`), AMR, and MPI domain decomposition | extension points |

---

## Computational Philosophy

This repository follows a **verification-driven methodology**. A numerical result is not
reported until it has been shown to converge to a known answer at the design rate of accuracy.
The book treats the following tools not as advanced topics but as the ordinary working practice
of computational fluid dynamics.

- **Analytical validation.** Every finite-volume implementation in the book is paired with an
  exact closed-form solution (Couette–Poiseuille, Hagen–Poiseuille, Stokes' first problem,
  Taylor–Green, isentropic nozzle, ISA atmosphere) and checked against it directly.

- **Benchmark validation.** Where no exact solution exists, the schemes are verified against
  accepted benchmark data — the Ghia, Ghia & Shin (1982) lid-driven cavity, the de Vahl Davis
  (1983) natural-convection cavity, the Blasius *f″(0) = 0.33206*, the Sod shock tube, and the
  Ritter/Stoker dam break.

- **Grid convergence.** Every numerical result is obtained on a sequence of systematically
  refined grids (or time steps), and the discretisation error is shown to decrease at the
  design rate.

- **Richardson extrapolation.** The leading error term *C hᵖ* is eliminated by combining
  solutions on two grids, yielding an estimate of the grid-converged value far more accurate
  than either grid alone.

- **Grid Convergence Index (GCI).** Roache's GCI provides a conservative error bar on a
  numerical result when no exact answer is available, with a factor of safety
  *F_s = 1.25* for three or more grids — the foundation of solution verification under
  ASME V&V 20.

- **Method of Manufactured Solutions (MMS).** For solvers with no analytical test case —
  variable-viscosity diffusion, the unsteady tank balance, the coupled *k–ε* model, the
  nonlinear Burgers equation — a smooth target field is manufactured, substituted into the
  operator to obtain the source that makes it exact, and the solver is required to recover it
  at its design order. MMS is the gold standard of code verification and is used to *detect a
  deliberate coding bug* in Chapter 12.

- **Reproducibility.** Every program is deterministic (no random numbers), depends only on the
  standard scientific Python stack, and writes its figures and convergence tables to disk. Any
  figure or number in the book can be regenerated with a single command.

- **Scientific computing.** Sparse linear algebra (SciPy), Krylov methods with ILU
  preconditioning, implicit time integration, and conservative finite-volume flux form are used
  throughout, so that the methods the reader studies are the same methods used in production CFD
  codes.

---

## Intended Audience

The book and this repository are addressed to graduate students, researchers, educators, and
practitioners who want a rigorous, hands-on path from fluid-mechanics theory to verified
computational practice.

- **Mechanical Engineering** — internal flows, piping, heat transfer, and turbomachinery.
- **Chemical Engineering** — transport phenomena, mixing, and scalar transport.
- **Civil Engineering** — open-channel and free-surface flows, pipe networks, and hydraulics.
- **Aerospace Engineering** — boundary layers, external flow, and compressible flow.
- **Energy Engineering** — natural convection, thermal systems, and cooling.
- **Materials Engineering** — diffusion and transport in process metallurgy and processing.
- **Graduate students** in engineering and applied mathematics seeking a verification-first CFD
  course.
- **Researchers** needing an extensible, readable finite-volume base for new models.
- **Educators** building a reproducible, code-integrated fluid-mechanics curriculum.
- **CFD practitioners** who want to understand, verify, and trust their numerical results.

---

## References

### Citation

If you use this repository or the CFDPy simulator in your teaching, research, or publications,
please cite the textbook:

> *Fluid Mechanics: Theory, Computation, and Verification — A Finite-Volume Approach with
> Python.* Companion repository, 2026.

and refer to the simulators' own documentation for the methods implemented:

> CFDPy — a modular finite-volume CFD framework in Python. Companion code, 2026.
> See [`CFDPY/README.md`](CFDPY/README.md).
>
> CFDPyGPU — a Numba-CUDA-accelerated variant of CFDPy. Companion code, 2026.
> See [`CFDPYGPU/README.md`](CFDPYGPU/README.md) and
> [`CFDPYGPU/GPU_PERFORMANCE_REPORT.md`](CFDPYGPU/GPU_PERFORMANCE_REPORT.md).

### BibTeX

```bibtex
@book{FluidMechanicsFVM2026,
  author    = {ileaof},
  title     = {Fluid Mechanics: Theory, Computation, and Verification ---
               A Finite-Volume Approach with Python},
  year      = {2026},
  note      = {Companion Python repository, CFDPy and CFDPyGPU simulators},
  url       = {https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification}
}

@software{CFDPy2026,
  author       = {ileaof},
  title        = {{CFDPy} -- a modular finite-volume CFD framework in Python},
  year         = {2026},
  howpublished = {Companion code to \emph{Fluid Mechanics: Theory,
                  Computation, and Verification}},
  url          = {https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification/tree/main/CFDPY}
}

@software{CFDPyGPU2026,
  author       = {ileaof},
  title        = {{CFDPyGPU} -- a Numba-CUDA-accelerated finite-volume CFD
                  framework in Python},
  year         = {2026},
  howpublished = {Companion code to \emph{Fluid Mechanics: Theory,
                  Computation, and Verification}},
  url          = {https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification/tree/main/CFDPYGPU}
}
```

> **Note on the `author` field.** Replace `ileaof` with the full author name when the book is
> published; the repository currently records only the maintainer's GitHub handle.

### License

This repository — the chapter programs, the CFDPy and CFDPyGPU simulators, and the companion
documentation — is licensed under a
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
See the [`LICENSE`](LICENSE) file for the full legal text.

In short, you are free to **share** and **adapt** the material for any non-commercial purpose,
provided you give appropriate credit to the textbook and indicate any changes made. Commercial
use requires a separate license from the maintainer. This single repository-level `LICENSE`
governs every file in the tree, including both simulator packages.

### Contributing

Contributions that improve the clarity, correctness, or verification coverage of the code are
welcome. Please:

1. Open an issue describing the change before starting work on a non-trivial pull request.
2. Keep programs deterministic and dependent only on the standard scientific Python stack.
3. For any new numerical result, include a grid- or time-step-refinement study reporting the
   observed order of accuracy and, where applicable, Richardson extrapolation and a GCI.
4. Follow the existing style: a module docstring stating the problem, the closed-form or
   benchmark reference, and the verification performed.

### Acknowledgments

The benchmark data referenced throughout the book — Ghia, Ghia & Shin (1982) for the lid-driven
cavity, de Vahl Davis (1983) for the natural-convection cavity, Blasius (1908) for the
flat-plate boundary layer, Harlow & Shannon (1967) for the liquid-drop splash, and Sod (1978)
for the shock tube — are the foundation of computational fluid dynamics and are used here as the
standards against which every scheme is verified. The finite-volume, projection, TVD, and
Riemann-solver methods implemented in CFDPy draw on the classical literature of Patankar, Chorin,
Toro, Roache, and LeVeque.

### Author and Contact

This repository is written and maintained by **[@ileaof](https://github.com/ileaof)**, author of
*Fluid Mechanics: Theory, Computation, and Verification — A Finite-Volume Approach with Python*.
The book, the thirty-nine chapter programs, and both simulator packages (CFDPy and CFDPyGPU) are
the work of a single author; the benchmark data and classical methods they are verified against
are credited in [Acknowledgments](#acknowledgments).

- **Repository:** <https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification>
- **Issues and corrections:** please open an issue on the repository — this is the preferred
  channel for errata, questions about a derivation or a program, and contribution proposals.
- **Commercial licensing:** enquiries about use outside the CC BY-NC 4.0 terms should be
  addressed to the maintainer through the contact details on the GitHub profile.

---

<p align="center">
  <em>Theory, computation, and verification — one chapter, one program, one convergence test at a time.</em>
</p>
