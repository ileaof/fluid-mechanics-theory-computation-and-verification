"""Coherent-SI unit registry, labelling helpers and consistency validator.

This module is the **single source of truth** for the physical units used
throughout the CFD framework.  The framework enforces a strict global policy:

    Every dimensional quantity is represented, stored, computed, documented,
    visualised and exported exclusively in **coherent SI units**.  All internal
    calculations are performed in SI.  Any non-SI or non-dimensional case must
    be *explicitly documented* (see :func:`validate_config`).

The registry below records, for every dimensional quantity in the framework,
its physical meaning and its coherent SI unit.  The exporters
(:mod:`visualization.tecplot_writer`), the plotting layer
(:mod:`visualization.matplotlib_view`) and the console/log output all label
their output from this one table, so a unit is defined in exactly one place.

Dimensionless groups (Reynolds, Prandtl, Rayleigh, Nusselt, Courant, the VOF
volume fraction, force coefficients, …) are tracked separately in
:data:`DIMENSIONLESS` and are rendered with the explicit ``[-]`` marker so a
reader never has to guess whether a blank unit means "dimensionless" or
"unlabelled".

Nothing here changes the numerics: the solver already works in SI.  This module
only *documents, labels and validates* — it never converts silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# --------------------------------------------------------------------------- #
# 1. The quantity registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quantity:
    """One physical quantity and its coherent-SI unit.

    Attributes
    ----------
    meaning:
        Human-readable physical meaning.
    unit:
        Coherent SI unit string (``"-"`` for a dimensionless group).
    """
    meaning: str
    unit: str


# Canonical dimensional quantities (coherent SI).  Keyed by a short canonical
# name; the exporters map their field/column names onto these.
QUANTITIES: dict[str, Quantity] = {
    # -- geometry / kinematics ------------------------------------------------
    "length":            Quantity("Length",                      "m"),
    "area":              Quantity("Area",                        "m^2"),
    "volume":            Quantity("Volume",                      "m^3"),
    "time":              Quantity("Time",                        "s"),
    "velocity":          Quantity("Velocity",                    "m/s"),
    "acceleration":      Quantity("Acceleration",                "m/s^2"),
    "angular_velocity":  Quantity("Angular velocity",            "rad/s"),
    "vorticity":         Quantity("Vorticity",                   "1/s"),
    "streamfunction":    Quantity("Stream function (2-D)",       "m^2/s"),
    # -- fluid / material properties -----------------------------------------
    "density":           Quantity("Density",                     "kg/m^3"),
    "dynamic_viscosity": Quantity("Dynamic viscosity",           "Pa*s"),
    "kinematic_viscosity": Quantity("Kinematic viscosity",       "m^2/s"),
    "pressure":          Quantity("Pressure",                    "Pa"),
    "temperature":       Quantity("Temperature",                 "K"),
    "heat_flux":         Quantity("Heat flux",                   "W/m^2"),
    "thermal_conductivity": Quantity("Thermal conductivity",     "W/(m*K)"),
    "specific_heat":     Quantity("Specific heat capacity",      "J/(kg*K)"),
    "thermal_diffusivity": Quantity("Thermal diffusivity",       "m^2/s"),
    "thermal_expansion": Quantity("Volumetric thermal expansion", "1/K"),
    "surface_tension":   Quantity("Surface tension",             "N/m"),
    # -- energy / power / forces ---------------------------------------------
    "energy":            Quantity("Energy",                      "J"),
    "enthalpy":          Quantity("Specific enthalpy",           "J/kg"),
    "entropy":           Quantity("Specific entropy",            "J/(kg*K)"),
    "power":             Quantity("Power",                       "W"),
    "force":             Quantity("Force",                       "N"),
    "torque":            Quantity("Torque",                      "N*m"),
    # -- mass / flow ----------------------------------------------------------
    "mass":              Quantity("Mass",                        "kg"),
    "mass_flow_rate":    Quantity("Mass flow rate",              "kg/s"),
    "volume_flow_rate":  Quantity("Volume flow rate",            "m^3/s"),
    "diffusivity":       Quantity("Mass diffusivity",            "m^2/s"),
    "permeability":      Quantity("Permeability",                "m^2"),
    "concentration":     Quantity("Species concentration",       "kg/m^3"),
    "gravity":           Quantity("Gravitational acceleration",  "m/s^2"),
}


# --------------------------------------------------------------------------- #
# 2. Dimensionless groups (must stay dimensionless)
# --------------------------------------------------------------------------- #
# Rendered with the explicit "[-]" marker.  Anything here is asserted to be a
# pure number by policy; the validator flags any attempt to give it a unit.
DIMENSIONLESS: dict[str, str] = {
    "Re":   "Reynolds number",
    "Pr":   "Prandtl number",
    "Gr":   "Grashof number",
    "Ra":   "Rayleigh number",
    "Nu":   "Nusselt number",
    "Pe":   "Peclet number",
    "Fo":   "Fourier number",
    "Bi":   "Biot number",
    "St":   "Strouhal number",
    "Mach": "Mach number",
    "CFL":  "Courant number",
    "Courant": "Courant number",
    "alpha":  "VOF volume fraction",
    "RH":     "Relative humidity",
    "Y":      "Mass fraction",
    "X_mol":  "Mole fraction",
    "Cd":     "Drag coefficient",
    "Cl":     "Lift coefficient",
    "Cp":     "Pressure coefficient",
    "Cf":     "Skin-friction coefficient",
    "theta":  "Time-scheme implicitness weight",
}


# --------------------------------------------------------------------------- #
# 3. Export field units  (state fields written to Tecplot / CSV / HDF5)
# --------------------------------------------------------------------------- #
# Maps the *export column / dataset name* to its SI unit string.  ``alpha`` is
# the dimensionless VOF fraction.
FIELD_UNITS: dict[str, str] = {
    "X": "m", "Y": "m", "Z": "m",
    "x": "m", "y": "m", "z": "m",
    "U": "m/s", "V": "m/s", "W": "m/s",
    "u": "m/s", "v": "m/s", "w": "m/s",
    "Pressure": "Pa", "p": "Pa",
    "Temperature": "K", "T": "K",
    "Alpha": "-", "alpha": "-",
    "Speed": "m/s", "speed": "m/s",
    "Vorticity": "1/s", "vort": "1/s",
    "StreamFunction": "m^2/s", "psi": "m^2/s",
    # plot display titles used by the Matplotlib viewer
    "Speed |u|": "m/s", "Velocity Magnitude": "m/s",
    "Density": "kg/m^3", "VOF": "-", "VOF fraction": "-",
}


# --------------------------------------------------------------------------- #
# 4. Configuration-parameter units
# --------------------------------------------------------------------------- #
# Maps every *dimensional* Config field to its coherent SI unit.  Dimensionless
# knobs map to "-"; purely structural / string / boolean fields are absent
# (the validator ignores those).
CONFIG_UNITS: dict[str, str] = {
    # mesh / geometry
    "Lx": "m", "Ly": "m", "Lz": "m",
    # time
    "dt": "s", "tfinal": "s", "dt_min": "s", "dt_max": "s",
    "output_interval": "s",
    "cfl_max": "-",
    # physical properties
    "rho": "kg/m^3", "mu": "Pa*s", "cp": "J/(kg*K)", "k": "W/(m*K)",
    "beta": "1/K",
    "rho_light": "kg/m^3", "mu_light": "Pa*s", "sigma": "N/m",
    "gravity": "m/s^2",
    # thermodynamic references
    "t_ref": "K", "t0": "K",
    # initial conditions
    "u0": "m/s", "v0": "m/s", "w0": "m/s",
    # obstacle / drop geometry
    "drop_x": "m", "drop_y": "m", "drop_z": "m", "drop_r": "m", "pool_height": "m",
    # dimensionless / tolerances
    "alpha_value": "-",
    "linear_tol": "-", "poisson_tol": "-",
    # GPU pressure-solver knobs (dimensionless by nature)
    "mg_max_density_ratio": "-",
}


# --------------------------------------------------------------------------- #
# 5. Labelling helpers  (used by every exporter / plot / log line)
# --------------------------------------------------------------------------- #
def unit_of(name: str) -> str:
    """Return the SI unit string for an export field or config parameter.

    Looks the name up in :data:`FIELD_UNITS`, then :data:`CONFIG_UNITS`, then
    the dimensionless registry.  Returns ``""`` (unknown) if the name is not a
    recognised dimensional quantity.
    """
    if name in FIELD_UNITS:
        return FIELD_UNITS[name]
    if name in CONFIG_UNITS:
        return CONFIG_UNITS[name]
    if name in DIMENSIONLESS:
        return "-"
    return ""


def label(name: str, pretty: str | None = None) -> str:
    """Return ``"<pretty> (<unit>)"`` for axis / colour-bar / column labels.

    Data labels (plot axes, colour bars, export columns) use **parentheses** —
    ``"Velocity (m/s)"``, ``"x (m)"``, ``"Pressure (Pa)"`` — matching the
    framework's output/visualisation convention; prose docstrings use brackets
    (``[Pa]``).  ``pretty`` overrides the displayed name (defaults to ``name``).
    An unknown quantity is returned unlabelled so callers never fabricate a
    wrong unit.
    """
    disp = pretty if pretty is not None else name
    u = unit_of(name)
    return f"{disp} ({u})" if u else disp


def tecplot_varnames(names: Iterable[str]) -> list[str]:
    """Return Tecplot ``VARIABLES`` names annotated with SI units.

    e.g. ``["X", "U", "Pressure", "Alpha"]`` ->
    ``["X [m]", "U [m/s]", "Pressure [Pa]", "Alpha [-]"]``.  Tecplot 360 and the
    py2tec round-trip both accept quoted variable names containing brackets.
    """
    return [label(n) for n in names]


def csv_header(columns: Iterable[str]) -> str:
    """Return a CSV header row with each column annotated with its SI unit."""
    return ",".join(label(c) for c in columns)


# --------------------------------------------------------------------------- #
# 6. Automatic consistency validator
# --------------------------------------------------------------------------- #
# Representative coherent-SI property ranges for *real* fluids, used to flag a
# config that is very likely non-dimensional (e.g. rho = 1 kg/m^3) so the user
# is forced to document it as an intentional exception.
_PHYSICAL_RANGES: dict[str, tuple[float, float, str]] = {
    #  field : (low, high, note)                      typical real-fluid span
    "rho": (0.05, 2.0e4, "gases ~0.1-2, liquids ~500-14000 kg/m^3"),
    "mu":  (1.0e-6, 2.0, "gases ~1e-5, liquids ~1e-4-1 Pa*s"),
    "cp":  (1.0e2, 1.5e4, "~120 (Hg) to ~14000 (H2) J/(kg*K)"),
    "k":   (1.0e-3, 5.0e2, "~0.02 (air) to ~400 (Cu) W/(m*K)"),
}


@dataclass
class Issue:
    """One SI-compliance finding for a config or field."""
    severity: str          # "error" | "warning" | "info"
    where: str             # config key / field name
    message: str


def validate_config(cfg, *, strict: bool = False) -> list[Issue]:
    """Check a :class:`config.Config` against the coherent-SI policy.

    Detects, per the framework policy:

    * **undocumented dimensional parameters** — a numeric field with no SI unit
      registered in :data:`CONFIG_UNITS`;
    * **non-dimensional / non-SI cases** — physical properties far outside any
      real-fluid range (e.g. ``rho == 1``), which signals a non-dimensionalised
      benchmark that must be *explicitly documented* to satisfy the policy;
    * **hidden conversion factors** — a ``t_ref`` / ``t0`` that looks like a
      Celsius value (< 200 K) rather than an absolute temperature in kelvin.

    Returns a list of :class:`Issue`.  With ``strict=True`` the non-dimensional
    finding is raised to ``"error"`` (the policy's "no exceptions unless
    documented" clause); otherwise it is a ``"warning"``.

    The check is *diagnostic only* — it never mutates the config.
    """
    issues: list[Issue] = []
    sev_nondim = "error" if strict else "warning"

    # (a) non-physical property values => likely non-dimensional case
    nondim_hits: list[str] = []
    for key, (lo, hi, note) in _PHYSICAL_RANGES.items():
        val = getattr(cfg, key, None)
        if val is None or not isinstance(val, (int, float)):
            continue
        # k == 0 is a legitimate "isothermal / no conduction" switch, not a unit
        # problem; skip a zero conductivity.
        if key == "k" and float(val) == 0.0:
            continue
        if not (lo <= float(val) <= hi):
            nondim_hits.append(f"{key}={val} (real range: {note})")
    if nondim_hits:
        issues.append(Issue(
            sev_nondim, "physical_properties",
            "Property value(s) outside any real-fluid SI range, so this case "
            "appears NON-DIMENSIONAL: " + "; ".join(nondim_hits) + ". Per the "
            "SI policy a non-dimensional case must be explicitly documented "
            '(add a top-level "nondimensional": true and a "reference_scales" '
            "block naming the length/velocity/density/ΔT used to scale it)."))

    # (b) absolute-temperature sanity (kelvin, not Celsius)
    for key in ("t0", "t_ref"):
        val = getattr(cfg, key, None)
        if isinstance(val, (int, float)) and 0.0 < float(val) < 200.0:
            issues.append(Issue(
                "warning", key,
                f"{key}={val} K is below 200 K; if this is meant to be degrees "
                "Celsius it is a hidden non-SI value — convert to kelvin."))

    # (c) undocumented dimensional parameters (numeric, non-flag fields with no
    #     registered unit and not a known dimensionless/tolerance/count knob).
    _ignore = {"Nx", "Ny", "Nz", "plot_interval", "linear_maxiter",
               "poisson_maxiter", "alpha_value", "mg_coarse_max_cells",
               "gpu_krylov_min_cells"}
    for key, val in vars(cfg).items():
        if key in _ignore or key in CONFIG_UNITS or key in DIMENSIONLESS:
            continue
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val != 0:
            issues.append(Issue(
                "info", key,
                f"Numeric parameter '{key}={val}' has no SI unit registered in "
                "units.CONFIG_UNITS; document it or confirm it is dimensionless."))

    # (d) an explicitly documented non-dimensional case clears the (a) finding
    if getattr(cfg, "nondimensional", False):
        issues = [i for i in issues if i.where != "physical_properties"]
        issues.append(Issue(
            "info", "physical_properties",
            "Case is explicitly flagged nondimensional:true — accepted as a "
            "documented exception to the coherent-SI policy."))

    return issues


def format_issues(issues: list[Issue]) -> str:
    """Render validator issues as a human-readable multi-line report block."""
    if not issues:
        return "SI check: OK — all dimensional parameters are documented in SI."
    order = {"error": 0, "warning": 1, "info": 2}
    lines = ["SI compliance check:"]
    for it in sorted(issues, key=lambda i: order.get(i.severity, 3)):
        lines.append(f"  [{it.severity.upper():7}] {it.where}: {it.message}")
    return "\n".join(lines)
