"""
Spatial reasoning: named zones (drawn as polygons in pixel coordinates)
and whether a tracked object's position falls inside them.

WHY THIS IS PLAIN GEOMETRY, NOT MACHINE LEARNING
------------------------------------------------------
Unlike every other perception module so far, there's no trained model
here. A "zone" is just a shape a human defines once (e.g. "this
rectangle of the camera view is the forklift lane"), and checking "is
this point inside this shape" is a classic, exactly-solvable geometry
problem — the ray casting algorithm below. No training data, no
uncertainty, no confidence score: either the point is inside the
polygon or it isn't.

RAY CASTING, THE ALGORITHM ITSELF
--------------------------------------
To test whether a point is inside a polygon: draw an imaginary ray from
that point off to infinity in one direction (we use "straight right").
Count how many times that ray crosses an edge of the polygon.
  - Odd number of crossings -> the point is INSIDE.
  - Even number of crossings -> the point is OUTSIDE.
Intuition: every time your ray crosses a boundary, you flip from
outside to inside or vice versa. Starting from definitely-outside
(infinitely far away) and crossing an odd number of boundaries means
you end up inside.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]   # ordered list of (x, y) corner points
    zone_type: str = "generic"            # e.g. "restricted", "pedestrian", "forklift_lane"


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon test. `polygon` is a list of (x, y) vertices, in order."""
    x, y = point
    inside = False
    n = len(polygon)

    j = n - 1  # start comparing against the LAST vertex, wrapping around
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Does the edge (xj,yj)-(xi,yi) straddle this point's y-coordinate,
        # and if so, does the ray (going right) cross it?
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


class ZoneManager:
    """Holds a set of named zones and answers containment/entry questions."""

    def __init__(self, zones: list[Zone] | None = None) -> None:
        self.zones = zones or []

    def add_zone(self, zone: Zone) -> None:
        self.zones.append(zone)

    def zones_containing(self, point: tuple[float, float]) -> list[Zone]:
        return [z for z in self.zones if point_in_polygon(point, z.polygon)]

    def zone_names_containing(self, point: tuple[float, float]) -> list[str]:
        return [z.name for z in self.zones_containing(point)]

    def detect_zone_entry(
        self,
        previous_point: tuple[float, float] | None,
        current_point: tuple[float, float],
    ) -> list[Zone]:
        """
        Given an object's previous and current position, return zones it
        just ENTERED this frame (was outside, now inside). If
        `previous_point` is None (first frame we've seen this object),
        nothing counts as a fresh "entry" yet.
        """
        if previous_point is None:
            return []

        currently_in = set(z.name for z in self.zones_containing(current_point))
        previously_in = set(z.name for z in self.zones_containing(previous_point))
        newly_entered_names = currently_in - previously_in

        return [z for z in self.zones if z.name in newly_entered_names]
