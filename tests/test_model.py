"""
Unit tests for Machine Learning gesture model.
"""

import os
import pytest
from model import GestureModel, GESTURE_CLASSES
from trainer import generate_canonical_hand, augment_landmarks, train_model
import numpy as np


@pytest.fixture(scope="module")
def trained_model():
    model_path = 'gesture_model.pkl'
    if not os.path.exists(model_path):
        train_model(samples_per_class=400, output_path=model_path)
    gm = GestureModel(model_path=model_path)
    return gm


def test_model_loads(trained_model):
    assert trained_model.model is not None
    assert len(trained_model.classes) == 13


def test_model_predict_all_classes(trained_model):
    for cls in GESTURE_CLASSES:
        trained_model.reset()
        hand = generate_canonical_hand(cls)
        pred, conf = trained_model.predict(hand, use_history=False)
        assert pred == cls, f"Failed prediction for {cls}, got {pred} (conf {conf:.2f})"
        assert conf >= 0.70


def test_model_augmented_robustness(trained_model):
    rng = np.random.RandomState(123)
    # Test 10 augmented variations of Peace and Fist
    for target in ['Fist', 'Peace', 'Open Palm', 'Pinch']:
        canonical = generate_canonical_hand(target)
        correct = 0
        total = 15
        for _ in range(total):
            aug = augment_landmarks(canonical, rng)
            pred, conf = trained_model.predict(aug, use_history=False)
            if pred == target:
                correct += 1
        accuracy = correct / total
        assert accuracy >= 0.85, f"Low augmented accuracy for {target}: {accuracy}"


def test_debounce_and_should_trigger(trained_model):
    trained_model.reset()
    hand_fist = generate_canonical_hand('Fist')
    pred, conf = trained_model.predict(hand_fist, use_history=False)

    # First trigger should be True
    trig1 = trained_model.should_trigger('Fist')
    assert trig1 is True

    # Immediate second trigger with same gesture should be False (debounced)
    trig2 = trained_model.should_trigger('Fist')
    assert trig2 is False
