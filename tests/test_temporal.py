import numpy as np
import torch

from perception.temporal.activity_recognition import (
    generate_synthetic_dataset,
    extract_hand_features,
    BaselineClassifier,
    LSTMClassifier,
    WINDOW_SIZE,
    NUM_FEATURES,
    ACTIVITIES,
)


def test_synthetic_dataset_shapes():
    X, y = generate_synthetic_dataset(n_sequences_per_class=5)
    assert X.shape == (5 * len(ACTIVITIES), WINDOW_SIZE, NUM_FEATURES)
    assert y.shape == (5 * len(ACTIVITIES),)
    assert set(y.tolist()) == set(range(len(ACTIVITIES)))  # every class present


def test_extract_hand_features_shape():
    X, _ = generate_synthetic_dataset(n_sequences_per_class=2)
    features = extract_hand_features(X[0])
    assert features.shape == (6,)
    assert not np.isnan(features).any()


def test_baseline_classifier_forward_pass_shape():
    model = BaselineClassifier(n_hand_features=6, n_classes=len(ACTIVITIES))
    batch = torch.randn(8, 6)
    output = model(batch)
    assert output.shape == (8, len(ACTIVITIES))


def test_lstm_classifier_forward_pass_shape():
    model = LSTMClassifier(n_features=NUM_FEATURES, hidden_size=16, n_classes=len(ACTIVITIES))
    batch = torch.randn(8, WINDOW_SIZE, NUM_FEATURES)
    output = model(batch)
    assert output.shape == (8, len(ACTIVITIES))


def test_falling_sequences_have_larger_hip_height_drop_than_standing():
    # Sanity check on the synthetic generator itself: falling should show
    # a much bigger hip-height range than standing, since that's the
    # exact signal the generator is designed to encode.
    X, y = generate_synthetic_dataset(n_sequences_per_class=10)
    from perception.temporal.activity_recognition import Activity

    standing_idx = ACTIVITIES.index(Activity.STANDING)
    falling_idx = ACTIVITIES.index(Activity.FALLING)

    standing_ranges = [X[i][:, 0].max() - X[i][:, 0].min() for i in range(len(X)) if y[i] == standing_idx]
    falling_ranges = [X[i][:, 0].max() - X[i][:, 0].min() for i in range(len(X)) if y[i] == falling_idx]

    assert np.mean(falling_ranges) > np.mean(standing_ranges)
