from perception.tracking.tracker import TrackHistory


def test_velocity_none_with_less_than_two_samples():
    history = TrackHistory()
    history.update(track_id=1, position=(0, 0), timestamp=0.0)
    assert history.velocity(1) is None


def test_velocity_computed_correctly_for_known_motion():
    history = TrackHistory()
    # Moves from (0,0) to (10,20) over exactly 1 second -> velocity (10, 20).
    history.update(track_id=1, position=(0, 0), timestamp=0.0)
    history.update(track_id=1, position=(10, 20), timestamp=1.0)
    vx, vy = history.velocity(1)
    assert abs(vx - 10.0) < 1e-9
    assert abs(vy - 20.0) < 1e-9


def test_velocity_uses_most_recent_two_samples_only():
    history = TrackHistory()
    history.update(track_id=1, position=(0, 0), timestamp=0.0)
    history.update(track_id=1, position=(100, 100), timestamp=1.0)   # big jump
    history.update(track_id=1, position=(105, 100), timestamp=2.0)   # small jump
    vx, vy = history.velocity(1)
    # Should reflect the LAST movement (5 px/s in x), not the first big jump.
    assert abs(vx - 5.0) < 1e-9
    assert abs(vy - 0.0) < 1e-9


def test_prune_removes_inactive_tracks():
    history = TrackHistory()
    history.update(track_id=1, position=(0, 0), timestamp=0.0)
    history.update(track_id=2, position=(0, 0), timestamp=0.0)
    history.prune(active_ids={1})
    assert 1 in history.active_track_ids()
    assert 2 not in history.active_track_ids()


def test_history_respects_max_length():
    history = TrackHistory(max_history=3)
    for i in range(10):
        history.update(track_id=1, position=(i, 0), timestamp=float(i))
    assert len(history.get_history(1)) == 3
    # Should keep the MOST RECENT samples, not the oldest.
    assert history.get_history(1)[-1].position == (9, 0)
