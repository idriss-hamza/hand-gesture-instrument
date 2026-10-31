"""
hand_detector.py
Real-time hand tracking and landmark detection using Google MediaPipe HandLandmarker.
Includes adaptive velocity-sensitive jitter reduction, spatial multi-hand tracking,
hand rotation (roll) calculation for pitch & volume control, and futuristic HUD drawing.
"""

from collections import deque
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 21-point hand skeleton connection graph
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

FINGERTIP_COLORS = {
    4: (0, 165, 255),   # Thumb: Orange
    8: (0, 255, 0),     # Index: Bright Green
    12: (255, 255, 0),  # Middle: Cyan
    16: (255, 100, 0),  # Ring: Blue
    20: (255, 0, 255),  # Pinky: Magenta
}


class HandInfo:
    """Stores detected hand state, smoothed landmarks, rotation angle, audio metadata, and tap events."""

    def __init__(self, raw_landmarks: np.ndarray, handedness: str = "Right", score: float = 1.0):
        self.raw_landmarks = raw_landmarks.copy()      # (21, 3) normalized [0..1]
        self.smoothed_landmarks = raw_landmarks.copy() # (21, 3) after filter
        self.handedness = handedness                   # "Left" or "Right"
        self.score = score
        self.gesture = "None"
        self.gesture_confidence = 0.0
        self.rotation_deg = 0.0                        # 2D tilt/rotation angle (-180..+180)
        self.pitch_shift_semitones = 0                 # Right hand pitch offset (-12..+12)
        self.volume_level = 0.85                       # Left hand volume level (0..1)
        self.is_tap = False                            # True on frames where a tap motion is detected
        self.tap_intensity = 0.0                       # Velocity/intensity of the tap
        self.tap_time = 0.0                            # Timestamp of most recent tap event
        self.compute_rotation()

    def compute_rotation(self) -> float:
        """
        Calculate 2D tilt/rotation angle of the hand from wrist (0) to middle MCP (9).
        0 degrees = pointing straight up.
        Positive degrees = tilted clockwise (to the right).
        Negative degrees = tilted counter-clockwise (to the left).
        """
        mid_mcp = self.smoothed_landmarks[9, :2]
        wrist = self.smoothed_landmarks[0, :2]
        dx = float(mid_mcp[0] - wrist[0])
        dy = float(mid_mcp[1] - wrist[1])
        # In image coords, -dy is upwards
        angle_rad = math.atan2(dx, -dy)
        self.rotation_deg = float(math.degrees(angle_rad))

        # Pitch shift for Right hand: every ~12 degrees is 1 semitone (-12 to +12)
        self.pitch_shift_semitones = int(np.clip(round(self.rotation_deg / 12.0), -12, 12))

        # Volume for Left hand: -60 deg = 0%, 0 deg = 65%, +60 deg = 100%
        if self.rotation_deg >= 0:
            self.volume_level = float(np.clip(0.65 + (self.rotation_deg / 60.0) * 0.35, 0.0, 1.0))
        else:
            self.volume_level = float(np.clip(0.65 + (self.rotation_deg / 60.0) * 0.65, 0.0, 1.0))

        return self.rotation_deg

    def get_pixel_coords(self, width: int, height: int) -> np.ndarray:
        """Convert normalized (21, 3) landmarks into integer pixel coords (21, 2)."""
        coords = np.zeros((21, 2), dtype=np.int32)
        coords[:, 0] = np.clip((self.smoothed_landmarks[:, 0] * width).astype(np.int32), 0, width - 1)
        coords[:, 1] = np.clip((self.smoothed_landmarks[:, 1] * height).astype(np.int32), 0, height - 1)
        return coords

    def get_bounding_box(self, width: int, height: int, padding: int = 20) -> Tuple[int, int, int, int]:
        """Compute pixel bounding box (x_min, y_min, x_max, y_max)."""
        pts = self.get_pixel_coords(width, height)
        x_min = max(0, int(np.min(pts[:, 0])) - padding)
        y_min = max(0, int(np.min(pts[:, 1])) - padding)
        x_max = min(width - 1, int(np.max(pts[:, 0])) + padding)
        y_max = min(height - 1, int(np.max(pts[:, 1])) + padding)
        return x_min, y_min, x_max, y_max


class TrackedHandState:
    """Internal persistent state for an individual tracked hand across frames, including tap kinematics."""

    def __init__(self, track_id: int, raw_landmarks: np.ndarray, handedness: str):
        self.track_id = track_id
        self.landmarks = raw_landmarks.copy()
        self.handedness = handedness
        self.handedness_history: deque = deque([handedness], maxlen=9)
        self.rotation_deg = 0.0
        self.last_seen = time.time()
        self.y_history: deque = deque(maxlen=8)
        self.tip_y_history: deque = deque(maxlen=8)
        self.scale_history: deque = deque(maxlen=8)
        self.last_tap_time = 0.0

    def detect_tap(self, current_landmarks: np.ndarray, now: float) -> Tuple[bool, float]:
        """
        Detect intuitive physical tap motion while holding a gesture:
        1. Vertical wrist dip & rebound (piano key strike motion).
        2. Fingertip dip & spring-back (fingers tapping down).
        3. Forward depth pulse (pushing hand slightly toward camera).
        """
        wrist_y = float(current_landmarks[0, 1])
        tips_y = float((current_landmarks[8, 1] + current_landmarks[12, 1]) / 2.0)
        scale = float(np.linalg.norm(current_landmarks[9, :2] - current_landmarks[0, :2]))

        self.y_history.append((now, wrist_y))
        self.tip_y_history.append((now, tips_y))
        self.scale_history.append((now, scale))

        if len(self.y_history) < 4:
            return False, 0.0

        if now - self.last_tap_time < 0.14:
            return False, 0.0

        # Check downward dip in the last 3-4 frames (within 120ms)
        t_recent, y_recent = self.y_history[-1]
        t_prev, y_prev = self.y_history[-2]
        t_old, y_old = self.y_history[-3]

        dy_dip = (y_prev - y_old)
        dy_recover = (y_recent - y_prev)
        tip_dip = (self.tip_y_history[-2][1] - self.tip_y_history[-3][1])
        d_scale = (self.scale_history[-1][1] - self.scale_history[-3][1])

        is_wrist_tap = (dy_dip > 0.016 and dy_recover <= 0.005)
        is_finger_tap = (tip_dip > 0.018)
        is_forward_tap = (d_scale > 0.024)

        if is_wrist_tap or is_finger_tap or is_forward_tap:
            self.last_tap_time = now
            intensity = max(0.5, min(1.0, dy_dip * 30.0 + tip_dip * 25.0 + d_scale * 20.0))
            return True, float(intensity)

        return False, 0.0


class HandDetector:
    """
    High-performance MediaPipe HandLandmarker wrapper with adaptive velocity-sensitive
    jitter reduction, persistent spatial hand tracking, and rotation extraction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_factor: float = 0.65
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'hand_landmarker.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model asset not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.smoothing_factor = smoothing_factor

        # Persistent hand trackers
        self.tracked_hands: Dict[int, TrackedHandState] = {}
        self.next_track_id = 0
        self.prev_hands: Dict[str, np.ndarray] = {}

    def process(self, frame_bgr: np.ndarray, is_mirrored: bool = True) -> List[HandInfo]:
        """
        Detect hands in a BGR frame (from OpenCV VideoCapture).
        Returns list of HandInfo objects with smoothed landmarks and rotation data.
        """
        now = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        # Cleanup expired tracks (> 0.6s)
        expired_ids = [tid for tid, th in self.tracked_hands.items() if now - th.last_seen > 0.6]
        for tid in expired_ids:
            del self.tracked_hands[tid]

        if not result.hand_landmarks:
            return []

        # Parse detected raw landmarks and handedness from MediaPipe
        detections = []
        for idx, lms in enumerate(result.hand_landmarks):
            raw = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(lms):
                raw[i] = [lm.x, lm.y, lm.z]

            handedness_label = "Right"
            score = 1.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                raw_label = result.handedness[idx][0].category_name
                score = result.handedness[idx][0].score
                # Invert handedness for mirrored camera view (selfie mode) if frame was mirrored
                if is_mirrored:
                    handedness_label = "Right" if raw_label == "Left" else "Left"
                else:
                    handedness_label = raw_label

            detections.append((raw, handedness_label, score))

        # Match detections to existing tracks by wrist spatial proximity
        matched_tracks = set()
        detected_hands: List[HandInfo] = []

        for raw, handedness_label, score in detections:
            wrist_pos = raw[0, :2]
            best_track_id = None
            min_dist = float('inf')

            for tid, th in self.tracked_hands.items():
                if tid in matched_tracks:
                    continue
                dist = float(np.linalg.norm(th.landmarks[0, :2] - wrist_pos))
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is not None and min_dist < 0.35:
                # Match found: update existing track
                matched_tracks.add(best_track_id)
                th = self.tracked_hands[best_track_id]

                # Adaptive velocity-sensitive smoothing:
                # Fast movement -> higher alpha (low latency); Still hand -> lower alpha (zero jitter)
                wrist_vel = min_dist
                alpha = float(np.clip(0.35 + wrist_vel * 3.0, 0.28, 0.85))

                smoothed = th.landmarks * (1.0 - alpha) + raw * alpha
                th.landmarks = smoothed
                th.last_seen = now
                th.handedness_history.append(handedness_label)

                # Stable handedness voting over recent frames to prevent flipping
                stable_handedness = max(set(th.handedness_history), key=th.handedness_history.count)
                th.handedness = stable_handedness

                hand_info = HandInfo(raw_landmarks=raw, handedness=stable_handedness, score=score)
                hand_info.smoothed_landmarks = smoothed

                # Compute rotation and smooth it with circular wrap-around handling
                raw_rot = hand_info.compute_rotation()
                rot_diff = (raw_rot - th.rotation_deg + 180.0) % 360.0 - 180.0
                th.rotation_deg = (th.rotation_deg + 0.30 * rot_diff + 180.0) % 360.0 - 180.0
                hand_info.rotation_deg = th.rotation_deg
                # Recalculate pitch/volume from smoothed rotation
                hand_info.pitch_shift_semitones = int(np.clip(round(th.rotation_deg / 12.0), -12, 12))
                if th.rotation_deg >= 0:
                    hand_info.volume_level = float(np.clip(0.65 + (th.rotation_deg / 60.0) * 0.35, 0.0, 1.0))
                else:
                    hand_info.volume_level = float(np.clip(0.65 + (th.rotation_deg / 60.0) * 0.65, 0.0, 1.0))

                # Detect intuitive physical tap motion
                is_tap, intensity = th.detect_tap(smoothed, now)
                hand_info.is_tap = is_tap
                hand_info.tap_intensity = intensity
                hand_info.tap_time = th.last_tap_time

                self.prev_hands[stable_handedness] = smoothed
                detected_hands.append(hand_info)

            else:
                # New hand track
                new_id = self.next_track_id
                self.next_track_id += 1
                th = TrackedHandState(new_id, raw, handedness_label)
                self.tracked_hands[new_id] = th
                matched_tracks.add(new_id)

                hand_info = HandInfo(raw_landmarks=raw, handedness=handedness_label, score=score)
                th.rotation_deg = hand_info.rotation_deg
                is_tap, intensity = th.detect_tap(raw, now)
                hand_info.is_tap = is_tap
                hand_info.tap_intensity = intensity
                hand_info.tap_time = th.last_tap_time
                self.prev_hands[handedness_label] = raw
                detected_hands.append(hand_info)

        return detected_hands

    def draw_hand_hud(self, image: np.ndarray, hand: HandInfo, is_control: Optional[bool] = None, draw_box: bool = True):
        """Draw vector cybernetic skeleton, palm polygon, reticle, and wrist arc inspired by GESTURE LIVE."""
        h, w = image.shape[:2]
        pts = hand.get_pixel_coords(w, h)
        now = time.time()

        # Hand Theme Color: Left Hand = Cybernetic Slate Cyan, Right Hand = Golden Amber
        ctrl = (hand.handedness != "Right") if is_control is None else is_control
        if ctrl:
            theme_color = (210, 190, 110)  # Cyan / slate-blue
            role_label = "L - CHORDS"
        else:
            theme_color = (40, 175, 255)   # Amber / gold
            role_label = "R - MELODY"

        # 1. Draw ultra-fine vector bone lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = tuple(pts[p1_idx])
            pt2 = tuple(pts[p2_idx])
            cv2.line(image, pt1, pt2, theme_color, 1, cv2.LINE_AA)

        # 2. Draw geometric palm polygon
        palm_indices = [0, 5, 9, 13, 17]
        palm_pts = np.array([pts[i] for i in palm_indices], dtype=np.int32)
        cv2.polylines(image, [palm_pts], True, theme_color, 1, cv2.LINE_AA)

        # 3. Draw palm center target reticle (+)
        palm_cx = int(np.mean([pts[i][0] for i in palm_indices]))
        palm_cy = int(np.mean([pts[i][1] for i in palm_indices]))
        cv2.circle(image, (palm_cx, palm_cy), 14, theme_color, 1, cv2.LINE_AA)
        cv2.line(image, (palm_cx - 7, palm_cy), (palm_cx + 7, palm_cy), theme_color, 1, cv2.LINE_AA)
        cv2.line(image, (palm_cx, palm_cy - 7), (palm_cx, palm_cy + 7), theme_color, 1, cv2.LINE_AA)

        # Hand Role Tag next to palm
        cv2.putText(image, role_label, (palm_cx + 18, palm_cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, theme_color, 1, cv2.LINE_AA)

        # 4. Draw joints & fingertip targeting rings
        for i, pt in enumerate(pts):
            p = tuple(pt)
            if i in (4, 8, 12, 16, 20):
                cv2.circle(image, p, 4, theme_color, 1, cv2.LINE_AA)
                cv2.circle(image, p, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(image, p, 2, (230, 240, 255), -1, cv2.LINE_AA)

        # Pinch annotation if tips are touching
        d_pinch = np.linalg.norm(hand.raw_landmarks[4] - hand.raw_landmarks[8])
        if d_pinch < 0.08:
            mid_pt = ((pts[4][0] + pts[8][0]) // 2, (pts[4][1] + pts[8][1]) // 2)
            cv2.line(image, mid_pt, (mid_pt[0] + 18, mid_pt[1] - 10), theme_color, 1, cv2.LINE_AA)
            cv2.putText(image, "INDEX", (mid_pt[0] + 20, mid_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (230, 240, 255), 1, cv2.LINE_AA)

        # 5. Draw Wrist Roll Radial Arc & Cutoff Gauge
        wrist_pt = tuple(pts[0])
        arc_r = 40
        # Background arc (-70 deg to +70 deg, centered upwards)
        cv2.ellipse(image, wrist_pt, (arc_r, arc_r), 0, -70 - 90, 70 - 90, (50, 65, 80), 1, cv2.LINE_AA)
        rot_clamped = float(np.clip(hand.rotation_deg, -70, 70))
        rot_rad = math.radians(rot_clamped - 90)
        bead_x = int(wrist_pt[0] + arc_r * math.cos(rot_rad))
        bead_y = int(wrist_pt[1] + arc_r * math.sin(rot_rad))
        is_fist = (getattr(hand, 'gesture', '') == 'Fist')
        if is_fist:
            cv2.circle(image, (bead_x, bead_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (bead_x, bead_y), 7, theme_color, 1, cv2.LINE_AA)
            status_text = "DIAL"
            status_col = theme_color
        else:
            cv2.circle(image, (bead_x, bead_y), 3, (160, 175, 190), -1, cv2.LINE_AA)
            status_text = "LOCKED"
            status_col = (110, 130, 150)

        # Cutoff digital readout
        cutoff_val = float(np.clip((rot_clamped + 60.0) / 120.0, 0.0, 1.0))
        cv2.putText(image, status_text, (wrist_pt[0] - 18, wrist_pt[1] + arc_r + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, status_col, 1, cv2.LINE_AA)
        cv2.putText(image, f"{cutoff_val:.2f}", (wrist_pt[0] - 12, wrist_pt[1] + arc_r + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 220, 240), 1, cv2.LINE_AA)

        # 6. Dynamic Tap Ripple Effect
        tap_age = now - hand.tap_time
        if tap_age < 0.35:
            alpha_fade = max(0.0, 1.0 - tap_age / 0.35)
            ripple_r1 = int(10 + tap_age * 80)
            ripple_color = (0, int(220 * alpha_fade), int(255 * alpha_fade))
            for tip_idx in (8, 12, 4):
                tip_pt = tuple(pts[tip_idx])
                cv2.circle(image, tip_pt, ripple_r1, ripple_color, 1, cv2.LINE_AA)

        # 7. Optional corner ticks if draw_box is requested
        if draw_box:
            x_min, y_min, x_max, y_max = hand.get_bounding_box(w, h)
            c_len = 12
            cv2.line(image, (x_min, y_min), (x_min + c_len, y_min), (60, 80, 100), 1)
            cv2.line(image, (x_min, y_min), (x_min + c_len, y_min), (60, 80, 100), 1)
            cv2.line(image, (x_max, y_min), (x_max - c_len, y_min), (60, 80, 100), 1)
            cv2.line(image, (x_max, y_min), (x_max, y_min + c_len), (60, 80, 100), 1)
            cv2.line(image, (x_min, y_max), (x_min + c_len, y_max), (60, 80, 100), 1)
            cv2.line(image, (x_min, y_max), (x_min - c_len, y_max), (60, 80, 100), 1)
            cv2.line(image, (x_max, y_max), (x_max - c_len, y_max), (60, 80, 100), 1)
            cv2.line(image, (x_max, y_max), (x_max, y_max - c_len), (60, 80, 100), 1)
