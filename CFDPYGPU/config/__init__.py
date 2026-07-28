"""Configuration subsystem for the CFD framework.

This package centralises the parsing of JSON/YAML case files into a fully
validated :class:`~config.config_loader.Config` dataclass.  The configuration
is the *single source of truth* for a simulation: every module reads from it
and never hard-codes physical or numerical parameters.

The loader is intentionally tolerant: it accepts both the flat ``{Nx, Ny, ...}``
syntax shown in the project specification *and* a richer nested layout.  Any
field not provided falls back to a documented default so minimal case files
remain valid.
"""

from .config_loader import Config, BoundarySpec, load_config, default_config

__all__ = ["Config", "BoundarySpec", "load_config", "default_config"]