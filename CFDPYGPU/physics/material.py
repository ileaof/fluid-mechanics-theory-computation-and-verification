"""Generic material record.

: class:`Material` is a thin, schema-less container for *any* material
(rheological fluid, porous medium, solid wall) so that future models --
phase change, radiation, species transport -- can attach properties without
touching :class:`physics.fluid.Fluid`.  It is deliberately decoupled from the
solver: a model reads whatever attributes it needs from the supplied
material and ignores the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Material:
    """Free-form material descriptor.

    Examples
    --------
    >>> water = Material(name="water", rho=1000.0, mu=1e-3, cp=4180.0, k=0.6)
    """

    name: str = "material"
    properties: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str = "material", **props: Any) -> None:
        self.name = name
        self.properties = dict(props)

    def __getattr__(self, item: str) -> Any:
        # Only called when normal attribute lookup fails -- look in props.
        try:
            return self.__dict__["properties"][item]
        except KeyError:
            raise AttributeError(f"Material {self.name!r} has no property {item!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in ("name", "properties"):
            super().__setattr__(key, value)
        else:
            self.properties[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.properties

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)