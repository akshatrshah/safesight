from perception.risk.risk_model import (
    compute_interaction_features,
    HeuristicRiskModel,
    RiskLevel,
)


def test_ttc_computed_correctly_for_direct_approach():
    # Forklift at origin moving right at 80 px/s, person 50px away directly
    # ahead and stationary -> distance/closing_speed = 50/80 = 0.625s.
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(80, 0),
        person_position=(50, 0), person_velocity=(0, 0),
    )
    assert features.distance == 50.0
    assert abs(features.closing_speed - 80.0) < 1e-9
    assert abs(features.time_to_collision - 0.625) < 1e-9


def test_ttc_is_none_when_moving_apart():
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(-80, 0),  # moving AWAY
        person_position=(50, 0), person_velocity=(0, 0),
    )
    assert features.time_to_collision is None


def test_ttc_is_none_when_stationary():
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(0, 0),
        person_position=(50, 0), person_velocity=(0, 0),
    )
    assert features.closing_speed == 0.0
    assert features.time_to_collision is None


def test_heuristic_classifies_close_fast_approach_as_high_risk():
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(100, 0),
        person_position=(30, 0), person_velocity=(0, 0),
        person_in_forklift_zone=True,
    )
    model = HeuristicRiskModel()
    assert model.classify(features) == RiskLevel.HIGH


def test_heuristic_classifies_far_stationary_as_low_risk():
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(0, 0),
        person_position=(1000, 1000), person_velocity=(0, 0),
        person_in_forklift_zone=False,
    )
    model = HeuristicRiskModel()
    assert model.classify(features) == RiskLevel.LOW


def test_heuristic_score_is_bounded_zero_to_one():
    features = compute_interaction_features(
        forklift_position=(0, 0), forklift_velocity=(500, 500),
        person_position=(1, 1), person_velocity=(0, 0),
        person_in_forklift_zone=True,
    )
    model = HeuristicRiskModel()
    score = model.score(features)
    assert 0.0 <= score <= 1.0
