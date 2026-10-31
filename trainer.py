"""
trainer.py
Dataset generation, augmentation, and machine learning model training for hand gestures.
Generates an anatomically accurate synthetic dataset with 3D rotations, scaling, and noise,
trains a high-accuracy Random Forest / Ensemble model, and evaluates test metrics.
"""

import argparse
import math
import os
import pickle
import time
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from gesture_features import extract_feature_vector, WRIST, THUMB_TIP, INDEX_TIP


GESTURE_CLASSES = [
    'Fist',         # C4 (Do)
    'Index Point',  # D4 (Re)
    'Peace',        # E4 (Mi)
    'Three',        # F4 (Fa)
    'Four',         # G4 (Sol)
    'Open Palm',    # A4 (La)
    'Thumbs Up',    # B4 (Ti)
    'Pinch',        # C5 (High Do)
    'Rock On',      # D5 (High Re)
    'Hang Loose',   # E5 (High Mi)
    'Gun',          # F5 (High Fa)
    'Pinky Up',     # G5 (High Sol)
    'OK Sign'       # A5 (High La)
]


def rotation_matrix_3d(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Compute 3D rotation matrix from Euler angles (radians)."""
    # Rotation about Z (roll)
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Rotation about Y (pitch)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float32)

    # Rotation about X (yaw)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(yaw), -math.sin(yaw)],
        [0, math.sin(yaw), math.cos(yaw)]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


def generate_canonical_hand(gesture: str) -> np.ndarray:
    """
    Generate canonical 3D landmarks for a human hand in standard posture.
    Returns (21, 3) numpy array.
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[WRIST] = [0.5, 0.8, 0.0]

    # Finger MCP positions relative to wrist
    mcp_offsets = {
        'thumb': np.array([-0.07, -0.06, 0.01], dtype=np.float32),
        'index': np.array([-0.05, -0.15, 0.0], dtype=np.float32),
        'middle': np.array([0.0, -0.17, 0.0], dtype=np.float32),
        'ring': np.array([0.05, -0.15, 0.0], dtype=np.float32),
        'pinky': np.array([0.09, -0.12, 0.0], dtype=np.float32),
    }

    # Finger segment lengths (PIP, DIP, TIP)
    lengths = {
        'thumb': [0.04, 0.035, 0.03],
        'index': [0.055, 0.038, 0.030],
        'middle': [0.060, 0.042, 0.032],
        'ring': [0.054, 0.038, 0.029],
        'pinky': [0.042, 0.028, 0.024],
    }

    # Determine extension state per finger for this gesture
    ext_thumb, ext_idx, ext_mid, ext_rng, ext_pnk = False, False, False, False, False

    if gesture == 'Fist':
        pass  # All false
    elif gesture == 'Index Point':
        ext_idx = True
    elif gesture == 'Peace':
        ext_idx, ext_mid = True, True
    elif gesture == 'Three':
        ext_idx, ext_mid, ext_rng = True, True, True
    elif gesture == 'Four':
        ext_idx, ext_mid, ext_rng, ext_pnk = True, True, True, True
    elif gesture == 'Open Palm':
        ext_thumb, ext_idx, ext_mid, ext_rng, ext_pnk = True, True, True, True, True
    elif gesture == 'Thumbs Up':
        ext_thumb = True
    elif gesture == 'Pinch':
        # Thumb and index tips meet; other fingers curled into palm
        ext_mid, ext_rng, ext_pnk = False, False, False
    elif gesture == 'OK Sign':
        # Thumb and index tips meet in circle; other 3 fingers extended
        ext_mid, ext_rng, ext_pnk = True, True, True
    elif gesture == 'Rock On':
        ext_idx, ext_pnk = True, True
    elif gesture == 'Hang Loose':
        ext_thumb, ext_pnk = True, True
    elif gesture == 'Gun':
        ext_thumb, ext_idx = True, True
    elif gesture == 'Pinky Up':
        ext_pnk = True

    # 1. Thumb chain (indices 1, 2, 3, 4)
    cmc = landmarks[WRIST] + mcp_offsets['thumb'] * 0.5
    mcp = landmarks[WRIST] + mcp_offsets['thumb']
    landmarks[1] = cmc
    landmarks[2] = mcp

    if gesture in ('Pinch', 'OK Sign'):
        # Position thumb tip near index tip contact point
        touch_point = landmarks[WRIST] + np.array([-0.05, -0.22, 0.03], dtype=np.float32)
        landmarks[3] = mcp + (touch_point - mcp) * 0.55
        landmarks[4] = touch_point
    elif ext_thumb:
        # Thumbs up or open thumb
        direction = np.array([-0.08, -0.06, 0.04], dtype=np.float32)
        if gesture == 'Thumbs Up':
            direction = np.array([-0.03, -0.12, 0.02], dtype=np.float32)  # Upward
        direction = direction / np.linalg.norm(direction)
        landmarks[3] = mcp + direction * lengths['thumb'][1]
        landmarks[4] = landmarks[3] + direction * lengths['thumb'][2]
    else:
        # Curled thumb against palm
        landmarks[3] = mcp + np.array([0.02, -0.03, -0.02], dtype=np.float32)
        landmarks[4] = landmarks[3] + np.array([0.03, 0.01, -0.01], dtype=np.float32)

    # 2. Four fingers
    finger_info = [
        ('index', 5, ext_idx, np.array([-0.01, -1.0, 0.0])),
        ('middle', 9, ext_mid, np.array([0.0, -1.0, 0.0])),
        ('ring', 13, ext_rng, np.array([0.01, -1.0, 0.0])),
        ('pinky', 17, ext_pnk, np.array([0.02, -1.0, 0.0])),
    ]

    for name, base_idx, is_ext, ext_dir in finger_info:
        mcp = landmarks[WRIST] + mcp_offsets[name]
        landmarks[base_idx] = mcp

        if gesture in ('Pinch', 'OK Sign') and name == 'index':
            # Index bends to touch thumb tip
            touch_point = landmarks[4]
            landmarks[base_idx + 1] = mcp + np.array([-0.01, -0.04, 0.02], dtype=np.float32)
            landmarks[base_idx + 2] = landmarks[base_idx + 1] + (touch_point - landmarks[base_idx + 1]) * 0.5
            landmarks[base_idx + 3] = touch_point
            continue

        l_pip, l_dip, l_tip = lengths[name]
        dir_norm = ext_dir / np.linalg.norm(ext_dir)

        if is_ext:
            # Extended straight
            pip = mcp + dir_norm * l_pip
            dip = pip + dir_norm * l_dip
            tip = dip + dir_norm * l_tip
        else:
            # Curled into fist / palm
            pip = mcp + dir_norm * (l_pip * 0.7) + np.array([0.0, 0.0, -0.02])
            dip = pip + np.array([0.0, 0.03, -0.03])  # bending inward
            tip = dip + np.array([0.0, 0.03, 0.01])   # resting on palm

        landmarks[base_idx + 1] = pip
        landmarks[base_idx + 2] = dip
        landmarks[base_idx + 3] = tip

    return landmarks


def augment_landmarks(canonical: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Apply realistic 3D geometric augmentation to hand landmarks:
    - 3D Rotation (yaw, pitch, roll up to +-35 deg)
    - Scale jitter (0.7x to 1.4x)
    - Translation jitter
    - Joint noise
    - Left/Right hand mirroring
    """
    hand = canonical.copy()
    wrist = hand[WRIST].copy()
    hand = hand - wrist

    # 1. 3D Rotation (realistic camera viewing angles)
    yaw = rng.uniform(-math.radians(45), math.radians(45))
    pitch = rng.uniform(-math.radians(45), math.radians(45))
    roll = rng.uniform(-math.radians(65), math.radians(65))
    R = rotation_matrix_3d(yaw, pitch, roll)
    hand = (R @ hand.T).T

    # 2. Scaling (close vs far hands)
    scale = rng.uniform(0.60, 1.50)
    hand = hand * scale

    # 3. Mirroring (Left vs Right hand)
    if rng.rand() > 0.5:
        hand[:, 0] = -hand[:, 0]

    # 4. Joint coordinate noise
    noise = rng.normal(0.0, 0.008, hand.shape).astype(np.float32)
    # Less noise on wrist and MCPs, slightly more on fingertips
    noise[0] *= 0.2
    for m in [2, 5, 9, 13, 17]:
        noise[m] *= 0.5
    hand += noise

    # 5. Translate to random screen position
    new_wrist = np.array([
        rng.uniform(0.3, 0.7),
        rng.uniform(0.4, 0.8),
        rng.uniform(-0.1, 0.1)
    ], dtype=np.float32)
    hand = hand + new_wrist

    return hand.astype(np.float32)


def generate_dataset(samples_per_class: int = 700, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented feature dataset for all gesture classes."""
    rng = np.random.RandomState(random_seed)
    X_list = []
    y_list = []

    print(f"[Trainer] Generating dataset: {samples_per_class} samples x {len(GESTURE_CLASSES)} classes...")
    start_t = time.time()

    for class_idx, gest_name in enumerate(GESTURE_CLASSES):
        canonical = generate_canonical_hand(gest_name)
        for _ in range(samples_per_class):
            aug_hand = augment_landmarks(canonical, rng)
            feats = extract_feature_vector(aug_hand)
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    elapsed = time.time() - start_t
    print(f"[Trainer] Generated {len(X)} samples with {X.shape[1]} features in {elapsed:.2f}s.")
    return X, y


def train_model(samples_per_class: int = 700, output_path: str = 'gesture_model.pkl') -> RandomForestClassifier:
    """Train Random Forest classifier and save model weights."""
    X, y = generate_dataset(samples_per_class=samples_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[Trainer] Training Random Forest classifier on {len(X_train)} samples...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[Trainer] Training completed in {train_time:.2f}s.")

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Trainer] Test Accuracy: {acc * 100:.2f}%\n")
    print("[Trainer] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=GESTURE_CLASSES))

    # Save to disk
    save_data = {
        'model': clf,
        'classes': GESTURE_CLASSES,
        'accuracy': float(acc),
        'timestamp': time.time(),
        'feature_dim': int(X.shape[1])
    }
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"[Trainer] Model successfully saved to {output_path}")

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument('--samples', type=int, default=700, help="Number of augmented samples per class")
    parser.add_argument('--output', type=str, default='gesture_model.pkl', help="Output model file path")
    args = parser.parse_args()

    train_model(samples_per_class=args.samples, output_path=args.output)
