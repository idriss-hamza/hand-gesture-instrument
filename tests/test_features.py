"""
Unit tests for landmark feature extraction and geometric classifications.
"""

import numpy as np
import pytest
from gesture_features import (
    extract_raw_landmarks,
    compute_scale,
    compute_angle,
    get_finger_extension_states,
    compute_pinch_distance,
    extract_feature_vector,
    rule_based_classify,
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP
)
from trainer import generate_canonical_hand, augment_landmarks


def test_extract_raw_landmarks_formats():
    # 21x3 numpy
    arr = np.zeros((21, 3), dtype=np.float32)
    assert extract_raw_landmarks(arr).shape == (21, 3)

    # 63 flattened
    arr_flat = np.zeros(63, dtype=np.float32)
    assert extract_raw_landmarks(arr_flat).shape == (21, 3)

    # list of tuples
    lst = [(0.1, 0.2, 0.3) for _ in range(21)]
    assert extract_raw_landmarks(lst).shape == (21, 3)


def test_compute_scale():
    hand = generate_canonical_hand('Open Palm')
    scale = compute_scale(hand)
    assert scale > 0.05
    assert scale < 0.5


def test_compute_angle():
    p1 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([0.0, -1.0, 0.0])
    # Straight line through p2 -> vectors (p1-p2) and (p3-p2) are opposite -> cos = -1
    cos_val = compute_angle(p1, p2, p3)
    assert abs(cos_val - (-1.0)) < 1e-4

    # Perpendicular
    p3_perp = np.array([1.0, 0.0, 0.0])
    cos_perp = compute_angle(p1, p2, p3_perp)
    assert abs(cos_perp) < 1e-4


def test_finger_extension_states():
    # Open Palm: all fingers extended
    hand_open = generate_canonical_hand('Open Palm')
    states_open = get_finger_extension_states(hand_open)
    assert states_open['index']
    assert states_open['middle']
    assert states_open['ring']
    assert states_open['pinky']
    assert states_open['thumb']

    # Fist: no fingers extended
    hand_fist = generate_canonical_hand('Fist')
    states_fist = get_finger_extension_states(hand_fist)
    assert not states_fist['index']
    assert not states_fist['middle']
    assert not states_fist['ring']
    assert not states_fist['pinky']

    # Peace: index and middle only
    hand_peace = generate_canonical_hand('Peace')
    states_peace = get_finger_extension_states(hand_peace)
    assert states_peace['index']
    assert states_peace['middle']
    assert not states_peace['ring']
    assert not states_peace['pinky']


def test_pinch_distance():
    hand_pinch = generate_canonical_hand('Pinch')
    pinch_d = compute_pinch_distance(hand_pinch)
    assert pinch_d < 0.20  # Tips touching

    hand_open = generate_canonical_hand('Open Palm')
    open_pinch_d = compute_pinch_distance(hand_open)
    assert open_pinch_d > 0.40


def test_feature_vector_dimension_and_invariance():
    hand = generate_canonical_hand('Index Point')
    feats1 = extract_feature_vector(hand)
    assert len(feats1) == 85
    assert not np.isnan(feats1).any()

    # Translation invariance: shift whole hand
    hand_shifted = hand + np.array([0.3, -0.4, 0.2], dtype=np.float32)
    feats_shifted = extract_feature_vector(hand_shifted)
    np.testing.assert_allclose(feats1, feats_shifted, atol=1e-3)

    # 3D Rotation invariance: rotate hand by 45, 90, 180 degrees
    import math
    from trainer import rotation_matrix_3d
    for angle in [45, 90, 180, -60]:
        R = rotation_matrix_3d(math.radians(15), math.radians(15), math.radians(angle))
        hand_rot = (R @ (hand - hand[0]).T).T + hand[0]
        feats_rot = extract_feature_vector(hand_rot)
        np.testing.assert_allclose(feats1, feats_rot, atol=1e-2)


def test_rule_based_classify_all_classes():
    classes = ['Fist', 'Index Point', 'Peace', 'Three', 'Four', 'Open Palm', 'Thumbs Up', 'Pinch', 'Rock On']
    for cls in classes:
        hand = generate_canonical_hand(cls)
        pred, conf = rule_based_classify(hand)
        assert pred == cls, f"Rule classify failed for {cls}: got {pred}"
        assert conf >= 0.80
