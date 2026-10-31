"""
instrument_modes.py
Interactive instrument engines for Hand Gesture Instrument:
1. SongGuideMode (Interactive play-along / rhythm tutorial for Jingle Bells and more)
2. GestureScaleMode (Direct gesture-to-note diatonic scale instrument)
3. AirPianoMode (Virtual spatial piano keyboard played with fingertip taps)
4. ThereminMode (Continuous spatial pitch and volume control)
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from audio_engine import AudioEngine, note_to_freq
from song_library import Song, SongNote, SONG_CATALOG, get_jingle_bells
from hand_detector import HandInfo
from cybernetic_hud import CyberneticHUD


def draw_studio_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int] = (0, 140, 255),
    thickness: int = 2,
    with_core: bool = True
):
    """
    Render high-resolution studio-grade typography:
    - Multi-pass feathered drop shadow for contrast against any background
    - Smooth anti-aliased vibrant color pass
    - Specular core highlight pass for crisp, razor-sharp legibility
    """
    x, y = org
    shadow_color = (10, 10, 14)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
        cv2.putText(image, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness + 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    if with_core and thickness >= 2:
        # Subtle core highlight preserving hue (keeps blue channel low for orange)
        b_boost = 40 if color[0] >= 50 else 0
        core_color = (min(255, color[0] + b_boost), min(255, color[1] + 35), min(255, color[2] + 20))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, core_color, max(1, thickness - 1), cv2.LINE_AA)


GESTURE_TO_NOTE_MAP = {
    'Fist': 'C4',
    'Index Point': 'D4',
    'Peace': 'E4',
    'Three': 'F4',
    'Four': 'G4',
    'Open Palm': 'A4',
    'Thumbs Up': 'B4',
    'Pinch': 'C5',
    'Rock On': 'D5',
    'Hang Loose': 'E5',
    'Gun': 'F5',
    'Pinky Up': 'G5',
    'OK Sign': 'A5'
}

NOTE_COLORS = {
    'C4': (0, 0, 255),       # Red
    'D4': (0, 165, 255),     # Orange
    'E4': (0, 255, 255),     # Yellow
    'F4': (0, 255, 0),       # Green
    'G4': (255, 255, 0),     # Cyan
    'A4': (255, 0, 0),       # Blue
    'B4': (238, 130, 238),   # Violet
    'Bb4': (200, 100, 255),  # Lavender
    'C5': (255, 20, 147),    # Deep Pink
    'D5': (180, 105, 255),   # Hot Pink
    'E5': (0, 215, 255),     # Amber
    'F5': (50, 205, 50),     # Lime Green
    'G5': (255, 140, 0),     # Dark Orange
    'A5': (220, 20, 60),     # Crimson
    'CHORD_C': (255, 215, 0) # Gold
}


class SongGuideMode:
    """
    Interactive song tutorial & play-along mode.
    Guides the user note-by-note through songs like Jingle Bells.
    """

    def __init__(self, audio: AudioEngine, song: Optional[Song] = None):
        self.audio = audio
        self.song = song or get_jingle_bells()
        self.current_note_idx = 0
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.total_attempts = 0
        self.correct_hits = 0
        self.is_completed = False
        self.autoplay = False
        self.last_autoplay_time = time.time()
        self.feedback_text = "Make the gesture to play the note!"
        self.feedback_color = (0, 255, 255)
        self.feedback_expire = time.time() + 3.0
        self.burst_particles: List[Dict] = []
        self.note_triggered_time = 0.0
        self.waiting_for_release = False
        self.last_triggered_gesture = "None"
        self.last_wrong_time = 0.0
        self.hud = CyberneticHUD()

    def set_song(self, song: Song):
        self.song = song
        self.reset()

    def reset(self):
        self.current_note_idx = 0
        self.score = 0
        self.streak = 0
        self.is_completed = False
        self.feedback_text = f"Playing {self.song.title}!"
        self.feedback_color = (0, 255, 200)
        self.feedback_expire = time.time() + 2.5
        self.burst_particles.clear()
        self.waiting_for_release = False
        self.last_triggered_gesture = "None"
        self.last_wrong_time = 0.0

    def get_current_note(self) -> Optional[SongNote]:
        if self.current_note_idx < len(self.song.notes):
            return self.song.notes[self.current_note_idx]
        return None

    def trigger_note_success(self, pitch_shift_semitones: int = 0):
        """Called when correct gesture is matched."""
        target_note = self.get_current_note()
        if not target_note:
            return

        # Ring note with recommended timbre (transposed if hand rotated)
        self.audio.play_note(
            target_note.note,
            velocity=0.9,
            duration=1.2,
            instrument=self.song.recommended_instrument,
            pitch_shift_semitones=pitch_shift_semitones
        )

        # Update scoring
        self.score += 100 + self.streak * 10
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self.correct_hits += 1
        self.total_attempts += 1

        self.feedback_text = f"PERFECT! ({target_note.note})"
        self.feedback_color = (0, 255, 0)
        self.feedback_expire = time.time() + 1.2
        self.note_triggered_time = time.time()

        # Advance note
        self.current_note_idx += 1
        if self.current_note_idx >= len(self.song.notes):
            self.is_completed = True
            self.feedback_text = "SONG COMPLETED! GREAT JOB!"
            self.feedback_color = (0, 255, 255)
            self.feedback_expire = time.time() + 10.0
            self.can_strike = True
        else:
            # If the next note requires the exact same gesture, require a re-strike / release
            next_note = self.song.notes[self.current_note_idx]
            if next_note.gesture == target_note.gesture:
                self.can_strike = False
            else:
                self.can_strike = True

        self.last_triggered_gesture = target_note.gesture

    @property
    def waiting_for_release(self) -> bool:
        return not getattr(self, 'can_strike', True)

    @waiting_for_release.setter
    def waiting_for_release(self, val: bool):
        self.can_strike = not val

    def trigger_tap(self):
        """Force advance on current target note (e.g. from Spacebar or physical tap)."""
        if not self.is_completed:
            self.trigger_note_success()

    def update(self, detected_gesture: str, confidence: float, pitch_shift_semitones: int = 0, is_tap: bool = False):
        """Update logic with current frame detection and tap support."""
        now = time.time()

        # Update burst particles
        alive_particles = []
        for p in self.burst_particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.05
            if p['life'] > 0:
                alive_particles.append(p)
        self.burst_particles = alive_particles

        if self.is_completed:
            return

        target_note = self.get_current_note()
        if not target_note:
            return

        # Handle Autoplay / Demo mode
        if self.autoplay:
            beat_sec = (60.0 / self.song.bpm) * target_note.duration
            if now - self.last_autoplay_time >= beat_sec:
                self.trigger_note_success(pitch_shift_semitones=pitch_shift_semitones)
                self.last_autoplay_time = now
            return

        # Auto-advance rests
        if target_note.note == 'REST':
            self.current_note_idx += 1
            self.can_strike = True
            return

        # Check if user released or relaxed gesture between consecutive identical notes, or tapped
        just_released = False
        if not self.can_strike:
            if is_tap:
                self.can_strike = True
            elif confidence < 0.60 or detected_gesture != target_note.gesture or detected_gesture in ('None', 'Rest'):
                self.can_strike = True
                just_released = True

        # Interactive play: match gesture
        if detected_gesture == target_note.gesture and confidence >= 0.70:
            if self.can_strike or is_tap:
                self.trigger_note_success(pitch_shift_semitones=pitch_shift_semitones)
        elif not just_released and detected_gesture not in ('None', 'Rest') and confidence >= 0.70:
            # Clear feedback on wrong gesture (only if not transitioning from a release)
            if now - self.last_wrong_time > 0.6:
                self.streak = 0
                self.total_attempts += 1
                self.feedback_text = f"Detected: {detected_gesture}  (Need: {target_note.gesture})"
                self.feedback_color = (0, 140, 255)
                self.feedback_expire = now + 1.2
                self.last_wrong_time = now

    def draw_hud(self, image: np.ndarray):
        """
        Draw clean, high-resolution Cybernetic HUD:
        Shows the needed note in the center hero display and highlights it on the pitch ladder.
        """
        now = time.time()
        target_note = self.get_current_note()
        if target_note:
            from song_library import GESTURE_TO_SYMBOL_MAP
            symbol_name = GESTURE_TO_SYMBOL_MAP.get(target_note.gesture, target_note.gesture)
            disp_note = target_note.note
            disp_sym = symbol_name
            disp_gest = target_note.gesture
        else:
            disp_note = "FINISHED!" if self.is_completed else None
            disp_sym = "SONG COMPLETE" if self.is_completed else None
            disp_gest = "None"

        hint = None
        if self.waiting_for_release:
            hint = "TAP HAND OR PRESS [SPACE] TO STRIKE NEXT NOTE"
        elif now < self.feedback_expire and self.feedback_text:
            hint = self.feedback_text

        waveform = self.audio.get_waveform_preview() if hasattr(self.audio, 'get_waveform_preview') else None

        self.hud.draw_hud(
            frame=image,
            active_note=disp_note,
            active_symbol=disp_sym,
            active_gesture=disp_gest,
            master_volume=self.audio.master_volume,
            instrument_name=self.audio.instrument,
            waveform_data=waveform,
            key_name=self.song.title.upper(),
            bpm=self.song.bpm,
            hint_text=hint
        )


class FreeInstrumentMode:
    """
    Free Instrument Mode (No guidance, pure expressive musical performance).
    Plays whatever gesture the user displays on the screen in real time.
    Features:
    - Tap-to-Repeat: Hold gesture and tap hand or press Space to replay note cleanly!
    - Gesture Transition Debounce: Requires 2-frame stability before switching notes,
      preventing transient curling fingers from playing accidental intermediate notes.
    - High-Resolution Studio Typography: Drop-glow feathered shadow with crisp specular glint.
    """

    def __init__(self, audio: AudioEngine):
        self.audio = audio
        self.active_note: Optional[str] = None
        self.active_base_note: Optional[str] = None
        self.active_gesture: str = "None"
        self.active_symbol: str = "None"
        self.pitch_shift_semitones: int = 0
        self.confidence: float = 0.0
        self.last_note_time: float = 0.0
        self.last_triggered_gesture: Optional[str] = None
        self.octave_offset: int = 0
        self.strike_count: int = 0
        self.display_note: Optional[str] = None
        self.display_symbol: Optional[str] = None
        self.display_pitch_shift: int = 0
        self.display_expire: float = 0.0
        self.hud = CyberneticHUD()
        self.active_chord_gesture: Optional[str] = None
        self.melody_hand_pos: Optional[Tuple[int, int]] = None
        self.last_chord_gesture: Optional[str] = None

    def reset(self):
        self.active_note = None
        self.active_base_note = None
        self.active_gesture = "None"
        self.active_symbol = "None"
        self.pitch_shift_semitones = 0
        self.confidence = 0.0
        self.last_triggered_gesture = None
        self.display_note = None
        self.display_symbol = None
        self.display_pitch_shift = 0
        self.display_expire = 0.0
        self.active_chord_gesture = None
        self.last_chord_gesture = None

    def trigger_tap(self):
        """Force a note strike on the current/active note (e.g. from Spacebar or external trigger)."""
        now = time.time()
        if now - self.last_note_time < 0.10:
            return
        note = self.active_note or self.display_note
        if note:
            if self.active_base_note == 'CHORD_C':
                self.audio.play_note('C4', duration=1.2)
                self.audio.play_note('E4', duration=1.2)
                self.audio.play_note('G4', duration=1.2)
            else:
                self.audio.play_note(note, velocity=0.92, duration=1.2)
            self.last_note_time = now
            self.strike_count += 1
            self.display_note = note
            self.display_symbol = self.active_symbol or self.display_symbol
            self.display_pitch_shift = self.pitch_shift_semitones

    def update(
        self,
        gesture: str,
        confidence: float,
        pitch_shift_semitones: int = 0,
        is_tap: bool = False,
        chord_gesture: Optional[str] = None,
        hand_pos: Optional[Tuple[int, int]] = None
    ):
        now = time.time()
        self.pitch_shift_semitones = pitch_shift_semitones
        self.confidence = confidence
        self.melody_hand_pos = hand_pos
        self.active_chord_gesture = chord_gesture

        # Left hand triggers polyphonic chords (degrees 1..7)
        if chord_gesture and chord_gesture not in ('None', 'Rest'):
            c_idx = self.hud.get_chord_for_gesture(chord_gesture)
            if c_idx is not None and chord_gesture != getattr(self, 'last_chord_gesture', None):
                chord_info = self.hud.chord_catalog[c_idx]
                self.audio.play_chord(chord_info['notes'], velocity=0.82, duration=1.8)
                self.last_chord_gesture = chord_gesture

        # Melody hand note triggering
        if confidence < 0.60 or gesture in ('None', 'Rest'):
            self.active_note = None
            self.active_base_note = None
            self.active_gesture = "None"
            self.active_symbol = "None"
            return

        base_note = GESTURE_TO_NOTE_MAP.get(gesture)
        if not base_note:
            return

        from audio_engine import shift_note_semitones
        from song_library import GESTURE_TO_SYMBOL_MAP

        note = shift_note_semitones(base_note, pitch_shift_semitones) if (pitch_shift_semitones != 0 and base_note != 'CHORD_C') else base_note
        sym = GESTURE_TO_SYMBOL_MAP.get(gesture, gesture)

        # 1. Tap-to-Repeat while holding the gesture
        if is_tap and note and (now - self.last_note_time >= 0.12):
            if base_note == 'CHORD_C':
                self.audio.play_note('C4', duration=1.2)
                self.audio.play_note('E4', duration=1.2)
                self.audio.play_note('G4', duration=1.2)
            else:
                self.audio.play_note(note, velocity=0.92, duration=1.2)
            self.last_note_time = now
            self.strike_count += 1
            self.display_note = note
            self.display_symbol = sym
            self.display_pitch_shift = pitch_shift_semitones
            return

        self.active_gesture = gesture
        self.active_base_note = base_note
        self.active_symbol = sym

        if note and note != self.active_note:
            self.active_note = note
            same_gesture = (gesture == self.last_triggered_gesture)
            # Rate limit audio triggers if rotating rapidly on boundary
            if not (same_gesture and (now - self.last_note_time < 0.12)):
                if base_note == 'CHORD_C':
                    self.audio.play_note('C4', duration=1.2)
                    self.audio.play_note('E4', duration=1.2)
                    self.audio.play_note('G4', duration=1.2)
                else:
                    self.audio.play_note(note, velocity=0.88, duration=1.2)
                self.last_note_time = now
                self.strike_count += 1

            self.last_triggered_gesture = gesture
            self.display_note = note
            self.display_symbol = self.active_symbol
            self.display_pitch_shift = pitch_shift_semitones

    def draw_hud(self, image: np.ndarray):
        """
        Draw clean, high-contrast Cybernetic HUD directly on screen:
        Shows the detected note or last note hit with vertical pitch ladder, chord table, and animated waveform.
        """
        show_note = self.active_note or self.display_note
        show_sym = self.active_symbol if self.active_note else self.display_symbol
        show_gest = self.active_gesture if self.active_note else (self.active_gesture if self.active_gesture != "None" else None)
        show_shift = self.pitch_shift_semitones if self.active_note else self.display_pitch_shift

        waveform = self.audio.get_waveform_preview() if hasattr(self.audio, 'get_waveform_preview') else None

        self.hud.draw_hud(
            frame=image,
            active_note=show_note,
            active_symbol=show_sym,
            active_gesture=show_gest,
            pitch_shift_semitones=show_shift,
            master_volume=self.audio.master_volume,
            instrument_name=self.audio.instrument,
            waveform_data=waveform,
            melody_hand_pos=self.melody_hand_pos,
            chord_gesture=self.active_chord_gesture,
            key_name="C MAJOR",
            bpm=122
        )


# Backward compatibility: GestureScaleMode aliases FreeInstrumentMode
GestureScaleMode = FreeInstrumentMode


class AirPianoMode:
    """
    Virtual spatial piano keyboard.
    Focuses strictly on the index fingertip (single finger pointer) per hand,
    ignoring thumb, palm, and other fingers to eliminate accidental notes.
    Features:
    - Index Finger Laser Pointer: Targeting reticle and vertical guide ray to active key
    - Hover Aiming Preview: Visual highlight showing which key is aimed at before striking
    - Responsive Touch & Tap: Clean note trigger on boundary touch or downward tap
    - Spacebar Tap: Tap active key with Spacebar
    - High-contrast cybernetic keyboard with visual keypress depression and ripple bursts
    """

    def __init__(self, audio: AudioEngine):
        self.audio = audio
        self.keys = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        self.pressed_key: Optional[str] = None
        self.pressed_keys: List[str] = []
        self.hovered_keys: List[str] = []
        self.last_press_times: Dict[str, float] = {}
        self.finger_pointers: List[Dict] = []
        self.key_bursts: List[Dict] = []
        self.last_tap_time: float = 0.0

    def trigger_tap(self):
        """Replay or force strike on the currently hovered/pressed key (e.g. from Spacebar)."""
        now = time.time()
        target = self.pressed_key or (self.hovered_keys[0] if self.hovered_keys else 'C4')
        if target and (now - self.last_tap_time > 0.10):
            self.audio.play_note(target, velocity=0.92, duration=1.2)
            self.last_press_times[target] = now
            self.last_tap_time = now
            self.pressed_key = target
            if target not in self.pressed_keys:
                self.pressed_keys.append(target)
            self._add_burst(target)

    def _add_burst(self, note: str):
        if note in self.keys:
            idx = self.keys.index(note)
            self.key_bursts.append({
                'key_idx': idx,
                'note': note,
                'birth': time.time(),
                'color': NOTE_COLORS.get(note, (0, 220, 255))
            })

    def update(self, hands: List[HandInfo], frame_width: int, frame_height: int):
        """Track index fingertip pointer only and detect key presses with zero thumb/palm interference."""
        now = time.time()
        key_w = frame_width // len(self.keys)
        key_y1 = frame_height - 145
        key_y2 = frame_height - 15

        current_pressed_set = set()
        current_hovered_set = set()
        self.finger_pointers = []

        for hand in hands:
            pts = hand.get_pixel_coords(frame_width, frame_height)
            # Strictly focus on ONE finger: Index Fingertip (landmark 8)
            tip = pts[8]
            ix, iy = int(tip[0]), int(tip[1])

            # Calculate horizontal key index
            key_idx = max(0, min(ix // key_w, len(self.keys) - 1))
            aimed_note = self.keys[key_idx]
            current_hovered_set.add(aimed_note)

            # Check if index finger is pressing the key:
            # 1. Inside key boundary (iy >= key_y1 and iy <= key_y2 + 15)
            # 2. Or downward kinematic tap while hovering above key (iy >= key_y1 - 40 and is_tap)
            is_tap = getattr(hand, 'is_tap', False)
            is_touching = (key_y1 <= iy <= key_y2 + 15)
            is_tapping = (key_y1 - 40 <= iy <= key_y2 + 15) and is_tap

            is_pressed = is_touching or is_tapping
            if is_pressed:
                current_pressed_set.add(aimed_note)

            self.finger_pointers.append({
                'pos': (ix, iy),
                'key_idx': key_idx,
                'note': aimed_note,
                'is_pressed': is_pressed,
                'handedness': getattr(hand, 'handedness', 'Right')
            })

        # Trigger notes for newly pressed keys
        new_keys = current_pressed_set - set(self.pressed_keys)
        for k in new_keys:
            last_t = self.last_press_times.get(k, 0.0)
            if now - last_t > 0.12:
                self.audio.play_note(k, velocity=0.88, duration=1.2)
                self.last_press_times[k] = now
                self._add_burst(k)

        # Allow tap re-triggering while remaining on same key
        for p in self.finger_pointers:
            if p['is_pressed']:
                k = p['note']
                last_t = self.last_press_times.get(k, 0.0)
                if (now - last_t > 0.18) and any(getattr(h, 'is_tap', False) for h in hands):
                    self.audio.play_note(k, velocity=0.88, duration=1.2)
                    self.last_press_times[k] = now
                    self._add_burst(k)

        self.pressed_keys = sorted(list(current_pressed_set))
        self.pressed_key = self.pressed_keys[0] if self.pressed_keys else None
        self.hovered_keys = sorted(list(current_hovered_set))

        # Clean up expired visual bursts (last 0.4s)
        self.key_bursts = [b for b in self.key_bursts if now - b['birth'] < 0.40]

    def draw_hud(self, image: np.ndarray):
        """Draw illuminated cybernetic piano keyboard with laser pointer and hover aiming."""
        h, w = image.shape[:2]
        now = time.time()
        key_w = w // len(self.keys)
        y1 = h - 145
        y2 = h - 15

        # 1. Top Mode & Aiming Banner
        cx = w // 2
        mode_str = "AIR PIANO  ·  ONE-FINGER POINTER"
        (mw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(image, mode_str, (cx - mw // 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        if self.pressed_key:
            status_str = f"PLAYING: {self.pressed_key}"
            status_color = (0, 255, 140)
        elif self.hovered_keys:
            status_str = f"AIMING: {self.hovered_keys[0]}  (TOUCH KEY TO PLAY)"
            status_color = (0, 180, 255)
        else:
            status_str = "POINT WITH YOUR INDEX FINGER TO PLAY KEYS"
            status_color = (150, 170, 190)

        (sw, _), _ = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(image, status_str, (cx - sw // 2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA)

        # 2. Draw Piano Keys
        # Strike Boundary Bar
        cv2.line(image, (0, y1), (w, y1), (60, 80, 100), 1, cv2.LINE_AA)

        for i, note in enumerate(self.keys):
            x1 = i * key_w
            x2 = x1 + key_w
            is_down = (note in self.pressed_keys)
            is_hovered = (note in self.hovered_keys) and not is_down
            accent_color = NOTE_COLORS.get(note, (0, 200, 255))

            # Visual key depression offset: moves down by 6px when pressed
            key_top = y1 + 6 if is_down else y1

            # Key Body Background
            if is_down:
                # Active key: luminous vibrant fill
                fill_color = accent_color
                border_color = (255, 255, 255)
                border_thickness = 2
            elif is_hovered:
                # Hovered key: translucent amber/cyan highlight
                fill_color = (225, 235, 245)
                border_color = (0, 180, 255)
                border_thickness = 2
            else:
                # Resting key: clean off-white piano key
                fill_color = (240, 242, 245)
                border_color = (25, 30, 40)
                border_thickness = 1

            # Draw key rectangle
            cv2.rectangle(image, (x1 + 3, key_top), (x2 - 3, y2), fill_color, -1)
            cv2.rectangle(image, (x1 + 3, key_top), (x2 - 3, y2), border_color, border_thickness, cv2.LINE_AA)

            # Glowing top header bar on each key
            header_color = (255, 255, 255) if is_down else accent_color
            cv2.rectangle(image, (x1 + 3, key_top), (x2 - 3, key_top + 10), header_color, -1)

            # Note label inside key
            text_color = (15, 15, 20) if (not is_down or np.mean(accent_color) > 130) else (255, 255, 255)
            (tw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            cv2.putText(image, note, (x1 + (key_w - tw) // 2, y2 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_color, 2, cv2.LINE_AA)

            # Subtle tick mark at bottom of key
            cv2.circle(image, (x1 + key_w // 2, y2 - 8), 2, (100, 110, 120), -1)

        # 3. Draw Visual Strike Bursts / Radiant Ripples
        for b in self.key_bursts:
            age = now - b['birth']
            if age < 0.40:
                alpha = max(0.0, 1.0 - age / 0.40)
                bx = b['key_idx'] * key_w + key_w // 2
                by = y1
                r = int(8 + age * 70)
                burst_col = (int(b['color'][0] * alpha), int(b['color'][1] * alpha), int(b['color'][2] * alpha))
                cv2.circle(image, (bx, by), r, burst_col, 2, cv2.LINE_AA)

        # 4. Draw Index Finger Laser Pointer & Targeting Reticle
        for p in self.finger_pointers:
            ix, iy = p['pos']
            aimed_note = p['note']
            is_down = p['is_pressed']
            ptr_color = (0, 255, 120) if is_down else (0, 200, 255)

            # Laser guide ray from index finger down to key line
            if iy < y1:
                cv2.line(image, (ix, iy + 14), (ix, y1), ptr_color, 1, cv2.LINE_AA)
                cv2.circle(image, (ix, y1), 4, ptr_color, -1, cv2.LINE_AA)

            # Circular targeting reticle on index fingertip
            cv2.circle(image, (ix, iy), 14, ptr_color, 1, cv2.LINE_AA)
            cv2.circle(image, (ix, iy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(image, (ix - 18, iy), (ix - 6, iy), ptr_color, 1, cv2.LINE_AA)
            cv2.line(image, (ix + 6, iy), (ix + 18, iy), ptr_color, 1, cv2.LINE_AA)
            cv2.line(image, (ix, iy - 18), (ix, iy - 6), ptr_color, 1, cv2.LINE_AA)
            cv2.line(image, (ix, iy + 6), (ix, iy + 18), ptr_color, 1, cv2.LINE_AA)

            # Cybernetic Floating Aim Badge above finger
            badge_text = f"POINTER · {aimed_note}"
            (bw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            bx1 = max(4, ix - bw // 2 - 6)
            by1 = max(10, iy - 32)
            cv2.rectangle(image, (bx1, by1), (bx1 + bw + 12, by1 + 16), (18, 22, 28), -1)
            cv2.rectangle(image, (bx1, by1), (bx1 + bw + 12, by1 + 16), ptr_color, 1, cv2.LINE_AA)
            cv2.putText(image, badge_text, (bx1 + 6, by1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 245, 255), 1, cv2.LINE_AA)


class ThereminMode:
    """
    Air Theremin mode:
    Right hand controls continuous Pitch (horizontal) and Volume (vertical).
    Left hand controls Vibrato depth.
    """

    def __init__(self, audio: AudioEngine):
        self.audio = audio
        self.freq = 440.0
        self.volume = 0.0
        self.vibrato = 0.0
        self.active = False

    def update(self, hands: List[HandInfo], frame_width: int, frame_height: int):
        if not hands:
            self.active = False
            self.audio.update_theremin(self.freq, 0.0, 0.0, enabled=False)
            return

        self.active = True
        right_hand = None
        left_hand = None

        for h in hands:
            if h.handedness == 'Right':
                right_hand = h
            elif h.handedness == 'Left':
                left_hand = h

        if right_hand is None:
            right_hand = hands[0]

        # Pitch from right hand X (130 Hz to 1046 Hz, C3 to C6)
        rx = float(right_hand.smoothed_landmarks[8, 0])  # index tip X [0..1]
        ry = float(right_hand.smoothed_landmarks[8, 1])  # index tip Y [0..1]

        # In mirrored camera view, moving hand to the right (rx increases) increases pitch
        norm_x = float(np.clip(rx, 0.0, 1.0))
        # Pitch mapping: exponential scale across 3 octaves
        min_f, max_f = 130.81, 1046.50
        self.freq = float(min_f * ((max_f / min_f) ** norm_x))

        # Volume from hand height (higher hand = louder)
        self.volume = float(np.clip(1.0 - ry, 0.0, 1.0))

        # Vibrato from left hand ONLY if a second hand is present
        if len(hands) >= 2 and left_hand is not None and left_hand is not right_hand:
            lx = float(left_hand.smoothed_landmarks[8, 0])
            self.vibrato = float(np.clip(lx, 0.0, 1.0))
        else:
            self.vibrato = 0.0

        self.audio.update_theremin(self.freq, self.volume, self.vibrato, enabled=True)

    def draw_hud(self, image: np.ndarray):
        """Draw Theremin pitch grid and real-time audio oscilloscope."""
        h, w = image.shape[:2]

        # Pitch guide banner
        cv2.putText(
            image, f"THEREMIN | PITCH: {int(self.freq)} Hz   VOL: {int(self.volume * 100)}%",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Oscilloscope waveform widget at top-right
        scope_w, scope_h = 220, 80
        sx, sy = w - scope_w - 20, 20
        cv2.rectangle(image, (sx, sy), (sx + scope_w, sy + scope_h), (10, 10, 20), -1)
        cv2.rectangle(image, (sx, sy), (sx + scope_w, sy + scope_h), (0, 200, 255), 1)

        wave = self.audio.get_waveform_preview()
        if len(wave) > 1:
            mid_y = sy + scope_h // 2
            pts = []
            for i, val in enumerate(wave):
                x = sx + int((i / len(wave)) * scope_w)
                y = mid_y - int(val * (scope_h // 2 - 5))
                pts.append((x, np.clip(y, sy + 2, sy + scope_h - 2)))

            for i in range(len(pts) - 1):
                cv2.line(image, pts[i], pts[i + 1], (0, 255, 200), 2, cv2.LINE_AA)
