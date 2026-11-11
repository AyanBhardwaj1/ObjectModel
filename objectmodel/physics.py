"""Physical properties: geometry boxes, materials, mass and inertia."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import oriented_bounding_box

# kg/m^3 — typical engineering values
DENSITY = {
    "wood": 600.0,
    "mdf": 750.0,
    "plastic": 1050.0,
    "aluminum": 2700.0,
    "steel": 7850.0,
    "glass": 2500.0,
    "rubber": 1100.0,
    "fabric": 300.0,
    "cardboard": 690.0,
    "ceramic": 2300.0,
}
# fraction of the bounding box that is solid material (shells, frames, hollow bodies)
DEFAULT_FILL = {
    "wood": 0.25, "mdf": 0.25, "plastic": 0.3, "aluminum": 0.15, "steel": 0.1,
    "glass": 0.5, "rubber": 0.8, "fabric": 0.5, "cardboard": 0.05, "ceramic": 0.4,
}
MIN_THICKNESS = 0.004  # metres; keeps inertia positive-definite for flat parts
MIN_MASS = 0.01  # kg


@dataclass(frozen=True)
class PartGeometry:
    center: np.ndarray  # world frame-0
    rotation: np.ndarray  # columns = box axes
    size: np.ndarray  # full extents (m)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))


@dataclass(frozen=True)
class PhysicalProps:
    material: str
    fill: float
    mass: float
    inertia_diag: np.ndarray  # in box frame (kg m^2)


def part_geometry(points: np.ndarray) -> PartGeometry:
    c, R, size = oriented_bounding_box(points)
    return PartGeometry(c, R, np.maximum(size, MIN_THICKNESS))


def box_inertia(mass: float, size: np.ndarray) -> np.ndarray:
    a, b, c = size
    return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def physical_props(geom: PartGeometry, material: str = "plastic", fill: float | None = None) -> PhysicalProps:
    material = material if material in DENSITY else "plastic"
    fill = DEFAULT_FILL[material] if fill is None else float(np.clip(fill, 0.01, 1.0))
    mass = max(MIN_MASS, DENSITY[material] * geom.volume * fill)
    return PhysicalProps(material, fill, mass, box_inertia(mass, geom.size))
