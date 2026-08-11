from perception.spatial.zones import Zone, ZoneManager, point_in_polygon

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_clearly_inside():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_clearly_outside():
    assert point_in_polygon((500, 500), SQUARE) is False


def test_point_outside_but_near_edge():
    assert point_in_polygon((100.5, 50), SQUARE) is False


def test_zone_manager_containment():
    zm = ZoneManager([Zone(name="restricted", polygon=SQUARE, zone_type="restricted")])
    assert zm.zone_names_containing((50, 50)) == ["restricted"]
    assert zm.zone_names_containing((500, 500)) == []


def test_zone_entry_detected_when_crossing_boundary():
    zm = ZoneManager([Zone(name="restricted", polygon=SQUARE)])
    entered = zm.detect_zone_entry(previous_point=(500, 500), current_point=(50, 50))
    assert [z.name for z in entered] == ["restricted"]


def test_zone_entry_not_detected_when_already_inside():
    zm = ZoneManager([Zone(name="restricted", polygon=SQUARE)])
    entered = zm.detect_zone_entry(previous_point=(40, 40), current_point=(60, 60))
    assert entered == []


def test_zone_entry_none_on_first_sighting():
    zm = ZoneManager([Zone(name="restricted", polygon=SQUARE)])
    entered = zm.detect_zone_entry(previous_point=None, current_point=(50, 50))
    assert entered == []
