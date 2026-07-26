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
[![License](https://img.shields.io/badge/License-Educational%2FResearch-lightgrey.svg)](#license)
[![Verification](https://img.shields.io/badge/Method-Verification--driven-success.svg)](#computational-philosophy)
[![CFDPy](https://img.shields.io/badge/CFD-CFDPy-orange.svg)](CFDPY/README.md)

---

## Table of Contents

- [Introduction](#introduction)
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
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Computational Philosophy](#computational-philosophy)
- [Intended Audience](#intended-audience)
- [References](#references)

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
  and Volume-of-Fluid free surfaces on structured Cartesian meshes.

The guiding principle of the book, and of this repository, is **verification-driven scientific
computing**. A numerical result is not accepted until it has been shown to converge to a known
answer at the design rate of accuracy. The Method of Manufactured Solutions, Richardson
extrapolation, and the Grid Convergence Index are used throughout — not as optional extras, but
as the ordinary working tools of the computational fluid dynamicist. Reproducibility is enforced
by construction: every program is deterministic (no random numbers), depends only on the standard
scientific Python stack, and writes its figures and convergence tables to disk.

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

All programs are deterministic (no random numbers), depend only on the standard scientific
Python stack (NumPy, SciPy, Matplotlib), and write their figures and convergence tables to disk.
They can be run individually:

```bash
python chapter05/ex5_2_fvm.py
python chapter13/ex13_2_fvm.py
```

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

A quick way to run the four bundled examples:

```bash
python main.py examples/natural_convection_2D/config.json
python main.py examples/dam_break_2D/config.json
python main.py examples/backward_facing_step/config.json
python main.py examples/liquid_drop_splash_2D/config.json
```

---

## Repository Structure

The repository is organised into **per-chapter program directories** (`chapter01`–`chapter13`),
each holding the three Python examples of that chapter, and the **`CFDPY/`** simulator package,
which itself decomposes into `config`, `mesh`, `numerics`, `physics`, `solver`, `visualization`,
`examples`, and `outputs`. The canonical top-level layout and the role of each part are
described below.

| Directory / file | Purpose |
|---|---|
| `chapter01/` … `chapter13/` | The **chapters** of the book, each with three runnable Python programs (`exN_1_analytical.py`, `exN_2_fvm.py`, `exN_3_advanced.py`). |
| `CFDPY/` | The **complete CFD simulator** (the `cfd` core), with its own README, requirements, and example cases. |
| `CFDPY/examples/` | **Ready-to-run example cases** (natural convection, dam break, backward-facing step, liquid-drop splash), each with a `config.json`. |
| `CFDPY/numerics/` | The **stateless finite-volume operators** (`src` of the simulator): interpolation, gradients, divergence, Laplacian, adaptive time step, Numba kernels. |
| `CFDPY/solver/` | The **solver core**: boundary conditions, linear solvers, momentum, pressure, projection, energy, and VOF. |
| `CFDPY/physics/` | Fluid properties, materials, gravity, and Boussinesq buoyancy. |
| `CFDPY/mesh/` | The Cartesian structured collocated mesh (2D/3D). |
| `CFDPY/config/` | The case-file loader and `Config`/`BoundarySpec` dataclasses. |
| `CFDPY/visualization/` | Matplotlib viewer, Tecplot exporter, and postprocessor. |
| `CFDPY/outputs/` | **Verification / output data** written at runtime: PNG, CSV, Tecplot `.dat`, HDF5 snapshots, MP4/GIF, and run logs. |
| `docs/` | (Reserved) lecture notes, slides, and supplementary documentation for the book. |
| `notebooks/` | (Reserved) Jupyter notebooks for interactive exploration of the analytical solutions and convergence studies. |
| `tests/` | (Reserved) unit and regression tests for the finite-volume operators and the simulator. |
| `tools/` | (Reserved) helper scripts for figure generation, post-processing, and batch verification runs. |
| `verification/` | (Reserved) the consolidated grid-convergence, Richardson, and GCI campaigns collected from the chapter programs. |

### Directory tree

```
Fluid-Mechanics-ebook/
├── README.md                          # This file — the book companion README
├── chapter01/                         # Foundations of the FVM — Couette–Poiseuille
│   ├── ex1_1_analytical.py
│   ├── ex1_2_fvm.py
│   └── ex1_3_advanced.py
├── chapter02/                        # Fluid statics and the hydrostatic balance
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
└── CFDPY/                             # The complete educational CFD simulator
    ├── README.md                      #   authoritative simulator documentation
    ├── main.py                        #   CLI entry point + Simulation orchestrator
    ├── requirements.txt               #   pinned Python dependencies
    ├── config/                        #   case-file loader (JSON/YAML)
    │   └── config_loader.py
    ├── mesh/                          #   Cartesian structured collocated mesh
    │   └── mesh.py
    ├── numerics/                      #   stateless finite-volume operators
    │   ├── interpolation.py           #   upwind / central / QUICK / TVD face values
    │   ├── numba_kernels.py           #   @njit TVD limiter (NumPy fallback)
    │   ├── gradients.py
    │   ├── divergence.py
    │   ├── laplacian.py
    │   └── timestep.py                #   CFL / Fourier adaptive dt
    ├── physics/                       #   fluid, materials, gravity, buoyancy
    │   ├── fluid.py
    │   ├── material.py
    │   ├── gravity.py
    │   └── buoyancy.py
    ├── solver/                       #   boundary, linear, momentum, pressure, projection, energy, vof
    │   ├── boundary.py
    │   ├── linear_solver.py
    │   ├── momentum.py
    │   ├── pressure.py
    │   ├── projection.py
    │   ├── energy.py
    │   └── vof.py
    ├── visualization/                  #   matplotlib, tecplot, postprocessor
    │   ├── matplotlib_view.py
    │   ├── tecplot_writer.py
    │   └── postprocessor.py
    ├── examples/                      #   ready-to-run cases
    │   ├── natural_convection_2D/config.json
    │   ├── dam_break_2D/config.json
    │   ├── backward_facing_step/config.json
    │   └── liquid_drop_splash_2D/config.json
    └── outputs/                       #   runtime output: PNG / CSV / .dat / HDF5 / MP4
```

---

## Installation

### Python requirements

The chapter programs require **Python 3.11+** and the standard scientific Python stack. The
CFDPy simulator is developed and verified on Python 3.11 and is compatible with 3.12+.

### Dependencies

| Library | Purpose | Required? |
|---|---|---|
| `numpy` | Array numerics | yes |
| `scipy` | Sparse matrices, Krylov solvers, ILU, special functions | yes |
| `matplotlib` | PNG plots and MP4/GIF animations | yes |
| `tqdm` | Progress bar | optional |
| `h5py` | HDF5 snapshot output (skipped gracefully if missing) | optional |
| `numba` | JIT-accelerated TVD flux limiter (pure-NumPy fallback) | optional |
| `pyyaml` | YAML case files (JSON always works) | optional |
| `meshio` | Unstructured-mesh I/O (future extension) | optional |
| `pyvista` | Optional interactive 3D viewer | optional |

A working **ffmpeg** binary on the system `PATH` enables MP4 animations; if ffmpeg is absent,
CFDPy automatically falls back to a pillow-written GIF.

### Installation

Clone the repository and install the runtime stack. The simulator's pinned dependencies live in
[`CFDPY/requirements.txt`](CFDPY/requirements.txt):

```bash
git clone https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification.git
cd Fluid-Mechanics-ebook

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

### Running the chapter examples

Each chapter program is a standalone script. Run any of them directly:

```bash
python chapter01/ex1_2_fvm.py
python chapter08/ex8_1_analytical.py
python chapter12/ex12_2_fvm.py
```

Programs write their figures and convergence tables next to the script or to a local `figures/`
folder.

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

and refer to the simulator's own documentation for the methods implemented:

> CFDPy — a modular finite-volume CFD framework in Python. Companion code, 2026.
> See [`CFDPY/README.md`](CFDPY/README.md).

### BibTeX

```bibtex
@book{FluidMechanicsFVM2026,
  title     = {Fluid Mechanics: Theory, Computation, and Verification ---
               A Finite-Volume Approach with Python},
  year      = {2026},
  note      = {Companion Python repository and CFDPy simulator},
  url       = {https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification}
}

@software{CFDPy2026,
  title       = {{CFDPy} -- a modular finite-volume CFD framework in Python},
  year        = {2026},
  howpublished = {Companion code to \emph{Fluid Mechanics: Theory,
                  Computation, and Verification}},
  url         = {https://github.com/ileaof/fluid-mechanics-theory-computation-and-verification/tree/main/CFDPY}
}
```

### License

Educational and research use. See the source headers in `CFDPY/` for attribution. The chapter
programs are released for study, teaching, and non-commercial research; please credit the
textbook when redistributing or deriving from them.

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

### Contact

For corrections, questions, or contributions, please open an issue on the repository or contact
the maintainer through the channels listed in the repository profile.

---

<p align="center">
  <em>Theory, computation, and verification — one chapter, one program, one convergence test at a time.</em>
</p>