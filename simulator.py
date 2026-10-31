"""
simulator.py
Synthetic hand simulation and mock camera provider for testing and demonstration.
Renders synthetic hand postures directly onto video frames with realistic movement.
"""

import math
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from trainer import generate_canonical_hand, augment_landmarks, rotation_matrix_3d
from song_library import get_jingle_bells, SongNote


class HandSimulator:
    """
    Generates video frames with an animated synthetic hand performing gestures.
    Used for automated tests, demonstrations, and camera-free execution.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.current_gesture = "Open Palm"
        self.song = get_jingle_bells()
        self.note_idx = 0
        self.note_start_time = time.time()
        self.start_time = time.time()

    def set_gesture(self, gesture: str):
        self.current_gesture = gesture

    def advance_song(self):
        """Advance to next note in Jingle Bells."""
        now = time.time()
        current_note: SongNote = self.song.notes[self.note_idx]
        beat_sec = (60.0 / self.song.bpm) * current_note.duration

        if now - self.note_start_time >= beat_sec:
            self.note_idx = (self.note_idx + 1) % len(self.song.notes)
            self.note_start_time = now
            self.current_gesture = self.song.notes[self.note_idx].gesture

    def get_current_landmarks(self) -> np.ndarray:
        """Get canonical landmarks for current gesture with subtle breathing motion."""
        canonical = generate_canonical_hand(self.current_gesture)
        t = time.time() - self.start_time
        # Subtle organic bobbing motion
        offset = np.array([
            0.02 * math.sin(t * 1.5),
            0.015 * math.cos(t * 2.0),
            0.0
        ], dtype=np.float32)
        landmarks = canonical + offset
        return landmarks

    def render_frame(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Render a camera frame (BGR) showing a dark stage background and the animated hand.
        Returns (frame, landmarks_array, gesture_name).
        """
        self.advance_song()
        lms = self.get_current_landmarks()

        # Create dark gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            v = int(18 + 15 * (y / self.height))
            frame[y, :] = (v, v - 2, v + 6)

        # Draw grid lines for studio effect
        grid_color = (35, 30, 42)
        for x in range(0, self.width, 60):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, 60):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)

        # Draw hand bones and joints onto the simulated camera frame
        pts = np.zeros((21, 2), dtype=np.int32)
        pts[:, 0] = np.clip((lms[:, 0] * self.width).astype(np.int32), 0, self.width - 1)
        pts[:, 1] = np.clip((lms[:, 1] * self.height).astype(np.int32), 0, self.height - 1)

        # Bones
        from hand_detector import HAND_CONNECTIONS, FINGERTIP_COLORS
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[p1]), tuple(pts[p2]), (200, 200, 220), 4, cv2.LINE_AA)
            cv2.line(frame, tuple(pts[p1]), tuple(pts[p2]), (255, 255, 255), 2, cv2.LINE_AA)

        # Fingertips
        for i in range(21):
            color = FINGERTIP_COLORS.get(i, (220, 220, 220))
            radius = 9 if i in FINGERTIP_COLORS else 5
            cv2.circle(frame, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[i]), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        # Footer tag
        cv2.putText(
            frame, "[SIMULATED CAMERA STREAM]", (16, self.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA
        )

        return frame, lms, self.current_gesture


class MockVideoCapture:
    """Mock OpenCV VideoCapture replacement for simulation/testing."""

    def __init__(self, simulator: Optional[HandSimulator] = None):
        self.simulator = simulator or HandSimulator()
        self.opened = True
        self.last_landmarks = None

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self.opened:
            return False, np.zeros((480, 640, 3), dtype=np.uint8)
        frame, lms, _ = self.simulator.render_frame()
        self.last_landmarks = lms
        return True, frame

    def get_simulated_hands(self):
        """Provides simulated HandInfo for gesture recognition and interactive testing."""
        from hand_detector import HandInfo
        if self.last_landmarks is None:
            self.last_landmarks = self.simulator.get_current_landmarks()
        hand_info = HandInfo(raw_landmarks=self.last_landmarks, handedness="Right", score=0.99)
        return [hand_info]

    def release(self):
        self.opened = False
