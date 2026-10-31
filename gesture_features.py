"""
gesture_features.py
Robust geometric feature extraction from 21 MediaPipe hand landmarks.
Provides scale, translation, and rotation-invariant features for machine learning
and rule-based fallback classification.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

# Indices
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
THUMB_IP, THUMB_MCP, THUMB_CMC = 3, 2, 1


def extract_raw_landmarks(landmarks_obj) -> np.ndarray:
    """
    Convert MediaPipe landmark objects or list of dicts/tuples into an (21, 3) numpy array.
    """
    if isinstance(landmarks_obj, np.ndarray):
        if landmarks_obj.shape == (21, 3):
            return landmarks_obj.astype(np.float32)
        elif landmarks_obj.shape == (63,):
            return landmarks_obj.reshape((21, 3)).astype(np.float32)

    coords = []
    # If it's a MediaPipe NormalizedLandmarkList or list of landmark objects
    for lm in landmarks_obj:
        if hasattr(lm, 'x') and hasattr(lm, 'y'):
            z = getattr(lm, 'z', 0.0)
            coords.append([lm.x, lm.y, z])
        elif isinstance(lm, (list, tuple)):
            coords.append([lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0])
        elif isinstance(lm, dict):
            coords.append([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
    
    arr = np.array(coords, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks with 3 coords, got shape {arr.shape}")
    return arr


def compute_scale(landmarks: np.ndarray) -> float:
    """
    Compute reference hand scale: distance from wrist (0) to middle MCP (9).
    Falls back to wrist-index MCP or 1.0 if degenerate.
    """
    diff = landmarks[MIDDLE_MCP] - landmarks[WRIST]
    scale = float(np.linalg.norm(diff))
    if scale < 1e-4:
        diff_idx = landmarks[INDEX_MCP] - landmarks[WRIST]
        scale = float(np.linalg.norm(diff_idx))
    return max(scale, 1e-4)


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute cosine of angle at joint p2 between vectors (p1-p2) and (p3-p2).
    Returns value in [-1.0, 1.0]. 1.0 = straight/extended, -1.0 = folded back.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-5 or norm2 < 1e-5:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(cos_theta, -1.0, 1.0))


def get_finger_extension_states(landmarks: np.ndarray) -> Dict[str, bool]:
    """
    Determine whether each of the 5 fingers is extended vs curled.
    Returns dict: {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
    """
    scale = compute_scale(landmarks)
    wrist = landmarks[WRIST]

    # For fingers (index, middle, ring, pinky):
    def is_extended(tip_idx, dip_idx, pip_idx, mcp_idx):
        v1 = landmarks[pip_idx] - landmarks[mcp_idx]
        v2 = landmarks[dip_idx] - landmarks[pip_idx]
        v3 = landmarks[tip_idx] - landmarks[dip_idx]
        v_tot = landmarks[tip_idx] - landmarks[mcp_idx]
        seg_sum = float(np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(v3))
        straightness = float(np.linalg.norm(v_tot) / max(1e-4, seg_sum))

        d_tip = float(np.linalg.norm(landmarks[tip_idx] - wrist))
        d_pip = float(np.linalg.norm(landmarks[pip_idx] - wrist))
        d_mcp_tip = float(np.linalg.norm(landmarks[tip_idx] - landmarks[mcp_idx]) / scale)
        cos_ang = compute_angle(landmarks[mcp_idx], landmarks[pip_idx], landmarks[tip_idx])

        # Robust rotation & tilt invariant extension criteria:
        # Knuckle must not be curled (cos_ang <= 0.35, d_mcp_tip >= 0.36)
        # Fingertip must reach outward away from wrist (d_tip > d_pip * 0.90)
        # AND either high segment straightness or open PIP reach
        if d_mcp_tip < 0.36 or cos_ang > 0.35:
            return False
        reach_out = d_tip > d_pip * 0.90
        return bool(reach_out and (straightness > 0.63 or (d_mcp_tip > 0.38 and cos_ang < -0.15)))

    index_ext = bool(is_extended(INDEX_TIP, 7, INDEX_PIP, INDEX_MCP))
    middle_ext = bool(is_extended(MIDDLE_TIP, 11, MIDDLE_PIP, MIDDLE_MCP))
    ring_ext = bool(is_extended(RING_TIP, 15, RING_PIP, RING_MCP))
    pinky_ext = bool(is_extended(PINKY_TIP, 19, PINKY_PIP, PINKY_MCP))

    # Thumb extension: check thumb tip to wrist ratio and abduction away from palm
    d_tip_wrist = float(np.linalg.norm(landmarks[THUMB_TIP] - wrist))
    d_mcp_wrist = float(np.linalg.norm(landmarks[THUMB_MCP] - wrist))
    thumb_ratio = d_tip_wrist / max(1e-4, d_mcp_wrist)

    d_thumb_pinky_mcp = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[PINKY_MCP]) / scale)
    d_thumb_idx_mcp = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_MCP]) / scale)

    # Thumb is extended when reaching outward from MCP and abducted away from pinky/palm
    thumb_ext = bool(
        (thumb_ratio > 1.20 and d_thumb_pinky_mcp > 0.80)
        or (d_thumb_pinky_mcp > 0.92 and d_thumb_idx_mcp > 0.34)
    )

    return {
        'thumb': thumb_ext,
        'index': index_ext,
        'middle': middle_ext,
        'ring': ring_ext,
        'pinky': pinky_ext
    }


def compute_pinch_distance(landmarks: np.ndarray) -> float:
    """Normalized Euclidean distance between thumb tip and index tip."""
    scale = compute_scale(landmarks)
    d = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
    return float(d / scale)


def align_to_palm_frame(landmarks: np.ndarray) -> np.ndarray:
    """
    Transform 21 landmarks into the local coordinate system of the palm:
    - Origin at wrist (landmark 0)
    - Y-axis aligned with the vector from wrist to middle MCP (landmark 9)
    - X-axis orthogonal in the palm plane towards index MCP (landmark 5)
    - Z-axis orthogonal cross product (depth)
    This provides true mathematical 3D translation, scale, and rotation invariance.
    """
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MIDDLE_MCP]
    idx_mcp = landmarks[INDEX_MCP]

    y_axis = mid_mcp - wrist
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-5:
        scale = compute_scale(landmarks)
        return (landmarks - wrist) / scale
    y_axis = y_axis / y_norm

    v = idx_mcp - mid_mcp
    x_axis = v - np.dot(v, y_axis) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-5:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm > 1e-5:
        z_axis = z_axis / z_norm

    R = np.vstack([x_axis, y_axis, z_axis])
    rel = landmarks - wrist
    aligned = (R @ rel.T).T / y_norm
    aligned[:, 2] = np.abs(aligned[:, 2])  # Chirality (left/right) depth invariance
    return aligned.astype(np.float32)


def extract_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract a normalized, high-discrimination 85-dimensional feature vector:
    - 63 3D rotation, scale, and translation invariant relative palm coordinates
    - 5 fingertip-to-wrist normalized distances
    - 6 inter-fingertip normalized distances
    - 5 finger joint angles (curl / extension)
    - 5 binary finger states
    - 1 total extended finger count
    """
    landmarks = extract_raw_landmarks(landmarks)
    scale = compute_scale(landmarks)
    wrist = landmarks[WRIST]

    # 1. 63 mathematically 3D rotation, translation, and scale invariant coordinates
    aligned_coords = align_to_palm_frame(landmarks)
    features = list(aligned_coords.flatten())

    # 2. 5 fingertip to wrist distances
    tip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    for tip in tip_indices:
        d = float(np.linalg.norm(landmarks[tip] - wrist) / scale)
        features.append(d)

    # 3. Pairwise inter-fingertip distances
    pairs = [
        (THUMB_TIP, INDEX_TIP),
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, MIDDLE_TIP),
        (THUMB_TIP, PINKY_TIP),
    ]
    for p1, p2 in pairs:
        d = float(np.linalg.norm(landmarks[p1] - landmarks[p2]) / scale)
        features.append(d)

    # 4. Finger joint curl angles (cos theta)
    features.append(compute_angle(landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_TIP]))
    features.append(compute_angle(landmarks[MIDDLE_MCP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_TIP]))
    features.append(compute_angle(landmarks[RING_MCP], landmarks[RING_PIP], landmarks[RING_TIP]))
    features.append(compute_angle(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_TIP]))
    features.append(compute_angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_TIP]))

    # 5. Extension states & count
    states = get_finger_extension_states(landmarks)
    features.append(1.0 if states['thumb'] else 0.0)
    features.append(1.0 if states['index'] else 0.0)
    features.append(1.0 if states['middle'] else 0.0)
    features.append(1.0 if states['ring'] else 0.0)
    features.append(1.0 if states['pinky'] else 0.0)
    features.append(float(sum(states.values())))

    return np.array(features, dtype=np.float32)


def rule_based_classify(landmarks: np.ndarray) -> Tuple[str, float]:
    """
    Robust heuristic classifier based on anatomical geometry.
    Used for 100% reliable backup or verification across all 13 classes:
      - 'Fist'
      - 'Index Point'
      - 'Peace'
      - 'Three'
      - 'Four'
      - 'Open Palm'
      - 'Thumbs Up'
      - 'Pinch'
      - 'Rock On'
      - 'Hang Loose'
      - 'Gun'
      - 'Pinky Up'
      - 'OK Sign'
    """
    states = get_finger_extension_states(landmarks)
    pinch_dist = compute_pinch_distance(landmarks)
    scale = compute_scale(landmarks)
    wrist = landmarks[WRIST]
    num_ext = sum([states['index'], states['middle'], states['ring'], states['pinky']])
    idx_tip_pip_ratio = float(np.linalg.norm(landmarks[INDEX_TIP] - wrist) / max(1e-4, np.linalg.norm(landmarks[INDEX_PIP] - wrist)))
    pinky_reach = float(np.linalg.norm(landmarks[PINKY_TIP] - landmarks[PINKY_MCP]) / scale)

    # 1. OK Sign: thumb and index in pinch circle, and at least 2 of middle, ring, pinky extended
    if pinch_dist < 0.28 and states['middle'] and (states['ring'] or states['pinky']):
        return 'OK Sign', 0.98

    # 2. Pinch: thumb and index touching, index arched forward, other 3 fingers curled
    if pinch_dist < 0.28 and idx_tip_pip_ratio >= 0.88 and not states['middle'] and not states['ring'] and not states['pinky']:
        return 'Pinch', 0.98

    # 3. Fist: all 4 non-thumb fingers curled into palm and thumb not extended
    if num_ext == 0 and not states['thumb']:
        return 'Fist', 0.98
    if num_ext == 0 and not states['thumb'] and idx_tip_pip_ratio < 0.90:
        return 'Fist', 0.98

    # 4. Pinky Up: only pinky extended, other 4 curled
    if states['pinky'] and not states['index'] and not states['middle'] and not states['ring'] and not states['thumb']:
        return 'Pinky Up', 0.98

    # 5. Hang Loose: thumb and pinky extended, middle 3 curled
    if states['thumb'] and (states['pinky'] or pinky_reach > 0.45) and not states['index'] and not states['middle'] and not states['ring']:
        return 'Hang Loose', 0.98

    # 6. Rock On: index and pinky extended, middle and ring curled, thumb tucked
    if states['index'] and states['pinky'] and not states['middle'] and not states['ring']:
        return 'Rock On', 0.98

    # 7. Gun: thumb and index extended, middle/ring/pinky curled, open pinch distance
    if pinch_dist >= 0.30 and states['thumb'] and states['index'] and not states['middle'] and not states['ring'] and not states['pinky']:
        return 'Gun', 0.98

    # 8. Thumbs Up: only thumb extended, other 4 curled
    if states['thumb'] and num_ext == 0:
        return 'Thumbs Up', 0.98

    # 9. Extended finger counts
    if num_ext == 1 and states['index']:
        return 'Index Point', 0.96
    elif num_ext == 2 and states['index'] and states['middle']:
        return 'Peace', 0.96
    elif num_ext == 3 and states['index'] and states['middle'] and states['ring']:
        return 'Three', 0.96
    elif num_ext == 2 and states['thumb'] and states['index'] and states['middle'] and not states['ring'] and not states['pinky']:
        return 'Three', 0.94
    elif num_ext == 4 and not states['thumb']:
        return 'Four', 0.96
    elif num_ext >= 4 and states['thumb']:
        return 'Open Palm', 0.98
    elif num_ext == 4:
        return 'Open Palm', 0.90

    # Fallback to finger count
    count = sum(states.values())
    fallback_map = {
        0: 'Fist',
        1: 'Index Point',
        2: 'Peace',
        3: 'Three',
        4: 'Four',
        5: 'Open Palm'
    }
    return fallback_map.get(count, 'Open Palm'), 0.80


GESTURE_SYMBOLS: Dict[str, str] = {
    'Fist': 'Closed Fist',
    'Index Point': '1 Finger (Index)',
    'Peace': '2 Fingers (Peace)',
    'Three': '3 Fingers',
    'Four': '4 Fingers',
    'Open Palm': 'Open Palm (5 Fingers)',
    'Thumbs Up': 'Thumbs Up',
    'Pinch': 'Pinch / Tip Touch',
    'Rock On': 'Rock On (Index + Pinky)',
    'Hang Loose': 'Hang Loose / Shaka',
    'Gun': 'Gun / L-Shape',
    'Pinky Up': 'Pinky Only',
    'OK Sign': 'OK Sign (Circle)',
    'None': 'None'
}
