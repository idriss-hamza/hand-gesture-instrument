"""
model.py
Machine Learning Gesture Classifier with temporal smoothing, hysteresis, and fallback logic.
Ensures zero-jitter, highly responsive gesture detection for musical performance.
"""

import os
import pickle
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from gesture_features import extract_feature_vector, rule_based_classify


GESTURE_CLASSES = [
    'Fist',         # C4
    'Index Point',  # D4
    'Peace',        # E4
    'Three',        # F4
    'Four',         # G4
    'Open Palm',    # A4
    'Thumbs Up',    # B4
    'Pinch',        # C5
    'Rock On',      # D5
    'Hang Loose',   # E5
    'Gun',          # F5
    'Pinky Up',     # G5
    'OK Sign'       # A5
]


class GestureModel:
    """
    Gesture classification model using trained ML estimator with confidence thresholding,
    temporal smoothing filter, and geometric rule-based verification.
    """

    def __init__(self, model_path: Optional[str] = None, smoothing_window: int = 5):
        self.model = None
        self.classes = GESTURE_CLASSES
        self.smoothing_window = smoothing_window
        self.prob_history = deque(maxlen=smoothing_window)
        self.last_predicted_gesture = "None"
        self.last_confidence = 0.0
        self.last_trigger_time = 0.0
        self.debounce_cooldown = 0.18  # Minimum seconds between repeated triggers of same note

        if model_path is None:
            # Check default locations
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_dir, 'gesture_model.pkl')
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            # Will use rule_based_classify until trained model is saved
            print("[GestureModel] Notice: Pre-trained model not found. Using geometric classifier.")

    def load(self, model_path: str):
        """Load trained model weights from pickle file."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data.get('model', data)
            self.classes = data.get('classes', self.classes)
        print(f"[GestureModel] Successfully loaded trained model from {model_path} ({len(self.classes)} classes)")

    def save(self, model_path: str):
        """Save model to disk."""
        data = {
            'model': self.model,
            'classes': self.classes,
            'saved_at': time.time()
        }
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[GestureModel] Saved model to {model_path}")

    def reset(self):
        """Reset temporal smoothing buffer and state."""
        self.prob_history.clear()
        self.last_predicted_gesture = "None"
        self.last_confidence = 0.0

    def predict(self, landmarks: np.ndarray, use_history: bool = True) -> Tuple[str, float]:
        """
        Predict gesture class and confidence for a set of 21 landmarks across all 13 classes.
        Uses ML classifier when loaded, combined with temporal probability smoothing.
        """
        features = extract_feature_vector(landmarks).reshape(1, -1)

        # 1. Get raw probabilities
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features)[0]
        else:
            # Rule based fallback
            rule_gest, rule_conf = rule_based_classify(landmarks)
            probs = np.zeros(len(self.classes), dtype=np.float32)
            if rule_gest in self.classes:
                idx = self.classes.index(rule_gest)
                probs[idx] = rule_conf
                rem = (1.0 - rule_conf) / max(1, len(self.classes) - 1)
                probs += rem
                probs[idx] = rule_conf
            else:
                probs.fill(1.0 / len(self.classes))

        # 2. Temporal smoothing across recent frames
        if use_history:
            self.prob_history.append(probs)
            smoothed_probs = np.mean(self.prob_history, axis=0)
        else:
            smoothed_probs = probs

        best_idx = int(np.argmax(smoothed_probs))
        confidence = float(smoothed_probs[best_idx])
        gesture_name = self.classes[best_idx]

        # 3. High-precision geometric sanity check and fusion
        rule_gest, rule_conf = rule_based_classify(landmarks)
        from gesture_features import compute_pinch_distance
        pinch_dist = compute_pinch_distance(landmarks)

        if gesture_name == rule_gest:
            confidence = max(confidence, rule_conf)
        else:
            # Special geometric disambiguations:
            if rule_gest == 'Pinch' and pinch_dist < 0.28:
                gesture_name = 'Pinch'
                confidence = max(confidence, rule_conf)
            elif rule_gest == 'OK Sign' and pinch_dist < 0.28:
                gesture_name = 'OK Sign'
                confidence = max(confidence, rule_conf)
            elif gesture_name in ('Pinch', 'OK Sign') and pinch_dist >= 0.32:
                gesture_name = rule_gest
                confidence = rule_conf
            elif rule_gest == 'Fist' and pinch_dist >= 0.30 and confidence < 0.90:
                gesture_name = 'Fist'
                confidence = max(confidence, rule_conf)
            elif rule_conf >= 0.95 and confidence < 0.85:
                gesture_name = rule_gest
                confidence = rule_conf
            elif rule_conf >= 0.90 and confidence < 0.70:
                gesture_name = rule_gest
                confidence = rule_conf

        self.last_predicted_gesture = gesture_name
        self.last_confidence = confidence
        return gesture_name, confidence

    def should_trigger(self, current_gesture: str, confidence_thresh: float = 0.65) -> bool:
        """
        Check if the gesture should trigger a musical note strike.
        Debounces repeated gestures while allowing crisp note transitions.
        """
        now = time.time()
        if self.last_confidence < confidence_thresh:
            return False

        if not hasattr(self, 'last_triggered_gesture'):
            self.last_triggered_gesture = "None"

        if current_gesture != self.last_triggered_gesture:
            self.last_triggered_gesture = current_gesture
            self.last_trigger_time = now
            return True

        # If holding same gesture, enforce debounce cooldown
        if now - self.last_trigger_time > self.debounce_cooldown:
            self.last_trigger_time = now
            return True

        return False
