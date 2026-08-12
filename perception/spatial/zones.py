"""Named polygon zones and point-in-polygon checks. Pure geometry, no model needed for this one."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]
    zone_type: str = "generic"


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting: cast a ray right from the point, count edge crossings, odd means inside."""
    x, y = point
    inside = False
    n = len(polygon)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


class ZoneManager:
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
        """Zones just entered this frame (was outside, now inside). First sighting never counts as entry."""
        if previous_point is None:
            return []

        currently_in = set(z.name for z in self.zones_containing(current_point))
        previously_in = set(z.name for z in self.zones_containing(previous_point))
        newly_entered_names = currently_in - previously_in

        return [z for z in self.zones if z.name in newly_entered_names]
