"""
cybernetic_hud.py
Sci-Fi Cybernetic HUD Engine inspired by GESTURE LIVE (v1.4)

Provides a clean, minimalist, high-contrast studio-grade musical HUD:
1. Top Header: [ GESTURE LIVE ] pill badge · Bar/Beat counter · 122 BPM · Key/Mode
2. Center Hero Display: Roman degree superscript, large glowing Note/Chord, voicing notes
3. Left Panel: L / CHORD INSTRUMENT (7 chord degrees with glowing active selection box)
4. Right Panel: R / MELODY & Vertical Pitch Ladder (C4..A5) with glowing cursor & guideline
5. Bottom Oscilloscope: Animated golden audio waveform, Space/Room slider, and footer menu
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# 7-Degree Diatonic Chord Voicings (A Minor and C Major)
CHORD_CATALOG_A_MINOR = [
    {"deg": "01", "pose": "|",     "roman": "I^7",   "name": "Am9",   "notes": ["A2", "E3", "G3", "C4", "B4"], "voicing": "A2 E3 G3 C4 B4"},
    {"deg": "02", "pose": "||",    "roman": "II^7",  "name": "Bø7.5", "notes": ["B2", "F3", "A3", "D4"],       "voicing": "B2 F3 A3 D4"},
    {"deg": "03", "pose": "|||",   "roman": "III^7", "name": "Cmaj9", "notes": ["C3", "G3", "B3", "D4", "E4"], "voicing": "C3 G3 B3 D4 E4"},
    {"deg": "04", "pose": "||||",  "roman": "iv^7",  "name": "Dm9",   "notes": ["D3", "A3", "C4", "E4", "F4"], "voicing": "D3 A3 C4 E4 F4"},
    {"deg": "05", "pose": "|||||", "roman": "v^9",   "name": "Em9",   "notes": ["E3", "B3", "D4", "F4", "E5"], "voicing": "E3 B3 D4 F4 E5"},
    {"deg": "06", "pose": "VI",    "roman": "VI^7",  "name": "Fmaj9", "notes": ["F3", "C4", "E4", "G4", "A4"], "voicing": "F3 C4 E4 G4 A4"},
    {"deg": "07", "pose": "VII",   "roman": "VII^7", "name": "G9",    "notes": ["G3", "D4", "F4", "A4", "B4"], "voicing": "G3 D4 F4 A4 B4"},
]

CHORD_CATALOG_C_MAJOR = [
    {"deg": "01", "pose": "|",     "roman": "I^7",   "name": "Cmaj7", "notes": ["C3", "G3", "B3", "E4"],       "voicing": "C3 G3 B3 E4"},
    {"deg": "02", "pose": "||",    "roman": "ii^7",  "name": "Dm7",   "notes": ["D3", "A3", "C4", "F4"],       "voicing": "D3 A3 C4 F4"},
    {"deg": "03", "pose": "|||",   "roman": "iii^7", "name": "Em7",   "notes": ["E3", "B3", "D4", "G4"],       "voicing": "E3 B3 D4 G4"},
    {"deg": "04", "pose": "||||",  "roman": "IV^7",  "name": "Fmaj7", "notes": ["F3", "C4", "E4", "A4"],       "voicing": "F3 C4 E4 A4"},
    {"deg": "05", "pose": "|||||", "roman": "V^7",   "name": "G7",    "notes": ["G3", "D4", "F4", "B4"],       "voicing": "G3 D4 F4 B4"},
    {"deg": "06", "pose": "VI",    "roman": "vi^7",  "name": "Am7",   "notes": ["A2", "E3", "G3", "C4"],       "voicing": "A2 E3 G3 C4"},
    {"deg": "07", "pose": "VII",   "roman": "vii°7", "name": "Bø7",   "notes": ["B2", "F3", "A3", "D4"],       "voicing": "B2 F3 A3 D4"},
]

# Vertical Pitch Ladder Notes (from bottom to top)
PITCH_LADDER_NOTES = [
    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
    'C5', 'D5', 'E5', 'F5', 'G5', 'A5'
]


class CyberneticHUD:
    """
    Renders the futuristic, minimalist sci-fi cybernetic instrument HUD
    directly inspired by GESTURE LIVE:
    - Zero overlapping text (each element has dedicated, bounded pixel coordinates)
    - Clean vector lines and translucent dark slate backdrop panels
    - Real-time animated audio oscilloscope waveform
    - Vertical pitch ladder on the right with glowing marker and guide line
    - 7-degree chord table on the left with glowing active row box
    """

    def __init__(self):
        self.start_time = time.time()
        self.beat_count = 1
        self.bar_count = 1
        self.last_beat_time = time.time()
        self.active_chord_idx: Optional[int] = None
        self.chord_catalog = CHORD_CATALOG_C_MAJOR

    def get_chord_for_gesture(self, gesture: str) -> Optional[int]:
        """Map gesture to chord catalog index (0..6)."""
        mapping = {
            'Index Point': 0,
            'Peace': 1,
            'Three': 2,
            'Four': 3,
            'Open Palm': 4,
            'Thumbs Up': 5,
            'Rock On': 5,
            'Hang Loose': 6,
            'Gun': 6,
            'OK Sign': 4,  # Em7 / Em9 variation
            'Pinch': 0,
        }
        return mapping.get(gesture, None)

    def draw_hud(
        self,
        frame: np.ndarray,
        active_note: Optional[str] = None,
        active_symbol: Optional[str] = None,
        active_gesture: Optional[str] = None,
        pitch_shift_semitones: int = 0,
        master_volume: float = 0.85,
        instrument_name: str = "SAW",
        waveform_data: Optional[np.ndarray] = None,
        melody_hand_pos: Optional[Tuple[int, int]] = None,
        chord_gesture: Optional[str] = None,
        key_name: str = "C MAJOR",
        bpm: int = 120,
        hint_text: Optional[str] = None
    ):
        """Draw the full Cybernetic HUD on the provided camera frame."""
        h, w = frame.shape[:2]
        now = time.time()
        elapsed = now - self.start_time

        # Update simulated bar / beat counter
        if now - self.last_beat_time >= (60.0 / bpm):
            self.beat_count += 1
            if self.beat_count > 4:
                self.beat_count = 1
                self.bar_count += 1
            self.last_beat_time = now

        # ---------------------------------------------------------
        # 1. TOP HEADER (y = 8 .. 28)
        # ---------------------------------------------------------
        # [ GESTURE LIVE ] pill badge (x = 16, y = 8 .. 26)
        cv2.rectangle(frame, (16, 7), (122, 27), (18, 22, 28), -1)
        cv2.rectangle(frame, (16, 7), (122, 27), (65, 75, 88), 1, cv2.LINE_AA)
        cv2.putText(frame, "GESTURE LIVE", (22, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 225, 235), 1, cv2.LINE_AA)

        # Bar Counter (Left of center)
        cx = w // 2
        bar_text = f"BAR {self.bar_count} · {self.beat_count}/4 · RUN"
        cv2.putText(frame, bar_text, (cx - 85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 158, 175), 1, cv2.LINE_AA)

        # Key & Tempo (Right corner)
        key_text = key_name.upper()
        (kw, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, key_text, (w - kw - 20, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 240), 1, cv2.LINE_AA)

        sub_text = f"{bpm} BPM · SWING 56 · V1.4"
        (sw, _), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(frame, sub_text, (w - sw - 20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 138, 155), 1, cv2.LINE_AA)

        # ---------------------------------------------------------
        # 2. CENTER HERO NOTE / CHORD DISPLAY (y = 36 .. 90)
        # ---------------------------------------------------------
        display_note = active_note
        if display_note:
            # Check if this note maps to a known chord voicing
            roman = "·"
            voicing = "TAP HAND OR [SPACE] TO REPEAT"
            if display_note in ('CHORD_C', 'Cmaj7'):
                display_title = "Cmaj7"
                roman = "I^7"
                voicing = "C3  G3  B3  E4"
            elif display_note in ('Em7', 'Em9'):
                display_title = display_note
                roman = "v^7" if display_note == 'Em7' else "v^9"
                voicing = "E3  B3  D4  G4"
            else:
                display_title = display_note
                if pitch_shift_semitones != 0:
                    sign = f"+{pitch_shift_semitones}" if pitch_shift_semitones > 0 else f"{pitch_shift_semitones}"
                    display_title = f"{display_note} ({sign}st)"
                if active_symbol and active_symbol != "None":
                    voicing = f"{active_symbol.upper()}  ·  REPLAY [SPACE]"
                elif active_gesture and active_gesture != "None":
                    voicing = f"{active_gesture.upper()}  ·  REPLAY [SPACE]"

            # Roman degree superscript (y = 44)
            (rw, _), _ = cv2.getTextSize(roman, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, roman, (cx - rw // 2, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)

            # Hero Text (y = 70)
            (nw, _), _ = cv2.getTextSize(display_title, cv2.FONT_HERSHEY_SIMPLEX, 1.15, 2)
            # Crisp drop shadow
            cv2.putText(frame, display_title, (cx - nw // 2 + 1, 71), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (10, 12, 16), 3, cv2.LINE_AA)
            # Glowing vivid orange/amber core
            cv2.putText(frame, display_title, (cx - nw // 2, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 140, 255), 2, cv2.LINE_AA)
            # Specular center highlight
            cv2.putText(frame, display_title, (cx - nw // 2, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (220, 240, 255), 1, cv2.LINE_AA)

            # Voicing / Subtitle (y = 86)
            (vw, _), _ = cv2.getTextSize(voicing, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            cv2.putText(frame, voicing, (cx - vw // 2, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (165, 185, 205), 1, cv2.LINE_AA)
        else:
            # Idle prompt
            idle_title = "GESTURE READY"
            idle_sub = hint_text or "SHOW HAND GESTURE TO PLAY  ·  TAP TO REPEAT"
            (iw, _), _ = cv2.getTextSize(idle_title, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
            (isw, _), _ = cv2.getTextSize(idle_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            cv2.putText(frame, idle_title, (cx - iw // 2, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, idle_sub, (cx - isw // 2, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 170, 190), 1, cv2.LINE_AA)

        # ---------------------------------------------------------
        # 3. LEFT PANEL: "L / CHORD INSTRUMENT" (y = 80 .. 275)
        # ---------------------------------------------------------
        lx = 18
        cv2.putText(frame, "L / CHORD INSTRUMENT", (lx, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 205, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "PALM FWD · 7 CHORDS", (lx, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 130, 150), 1, cv2.LINE_AA)

        # Table header
        cv2.putText(frame, "DEG   POSE   CHORD", (lx, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (130, 150, 170), 1, cv2.LINE_AA)

        # Determine active chord row
        selected_chord_idx = self.get_chord_for_gesture(chord_gesture) if chord_gesture else None

        row_y = 138
        for i, item in enumerate(self.chord_catalog):
            is_active = (i == selected_chord_idx)

            if is_active:
                # Glowing selection box around active chord row
                cv2.rectangle(frame, (lx - 4, row_y - 12), (lx + 148, row_y + 4), (18, 28, 40), -1)
                cv2.rectangle(frame, (lx - 4, row_y - 12), (lx + 148, row_y + 4), (0, 140, 255), 1, cv2.LINE_AA)
                row_color = (0, 180, 255)
            else:
                row_color = (130, 145, 160)

            line_str = f"{item['deg']}   {item['pose']:<5} {item['roman']:<4} {item['name']}"
            cv2.putText(frame, line_str, (lx, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, row_color, 1, cv2.LINE_AA)
            row_y += 16

        # Wrist Roll / Cutoff slider gauge
        slider_y = row_y + 14
        vol_mode = "DIAL" if chord_gesture == "Fist" else "LOCKED"
        cv2.putText(frame, f"WRIST ROLL · VOL {int(master_volume * 100)}% [{vol_mode}]", (lx, slider_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 140, 160), 1, cv2.LINE_AA)
        gauge_w = 110
        cv2.line(frame, (lx, slider_y + 6), (lx + gauge_w, slider_y + 6), (45, 55, 68), 2, cv2.LINE_AA)
        fill_w = int(gauge_w * np.clip(master_volume, 0.0, 1.0))
        cv2.line(frame, (lx, slider_y + 6), (lx + fill_w, slider_y + 6), (0, 160, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (lx + fill_w, slider_y + 6), 3, (220, 240, 255), -1, cv2.LINE_AA)

        # ---------------------------------------------------------
        # 4. RIGHT PANEL: "R / MELODY" & VERTICAL PITCH LADDER
        # ---------------------------------------------------------
        rx = w - 165
        inst_label = instrument_name.upper()
        cv2.putText(frame, "R / MELODY", (rx, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 205, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{inst_label} · 14 NOTES · DRY", (rx, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 130, 150), 1, cv2.LINE_AA)

        # Voice selection hints
        cv2.putText(frame, "PINCH - VOICE", (rx, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (130, 150, 170), 1, cv2.LINE_AA)
        voice_items = [
            ("INDEX", "BELL"),
            ("MIDDLE", "PIANO"),
            ("RING", "VIOLIN"),
            ("PINKY", "SYNTH"),
        ]
        vy = 138
        for finger, vname in voice_items:
            dot = "·" if vname.lower() in instrument_name.lower() else " "
            v_color = (0, 180, 255) if dot == "·" else (120, 135, 150)
            cv2.putText(frame, f"{dot} {finger:<7} {vname}", (rx, vy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, v_color, 1, cv2.LINE_AA)
            vy += 15

        # Vertical Pitch Ladder on Far Right (x = w - 38, y = 80 .. h - 60)
        ladder_x = w - 36
        ladder_y_top = 85
        ladder_y_bottom = h - 65
        cv2.line(frame, (ladder_x, ladder_y_top), (ladder_x, ladder_y_bottom), (50, 62, 75), 1, cv2.LINE_AA)

        num_notes = len(PITCH_LADDER_NOTES)
        note_y_step = (ladder_y_bottom - ladder_y_top) / max(1, num_notes - 1)

        active_marker_y = None

        # Draw tick marks from C4 (bottom) to A5 (top)
        for i, n in enumerate(PITCH_LADDER_NOTES):
            # i=0 is C4 at ladder_y_bottom, i=12 is A5 at ladder_y_top
            ny = int(ladder_y_bottom - i * note_y_step)
            is_ladder_active = (active_note == n)

            if is_ladder_active:
                active_marker_y = ny
                # Highlight tick
                cv2.line(frame, (ladder_x - 10, ny), (ladder_x + 10, ny), (0, 140, 255), 2, cv2.LINE_AA)
            else:
                tick_len = 6 if n.endswith('4') and n[0] in ('C', 'E', 'G') else 4
                cv2.line(frame, (ladder_x - tick_len, ny), (ladder_x + tick_len, ny), (70, 85, 100), 1, cv2.LINE_AA)

        # Draw glowing cursor box for active note on ladder
        if active_marker_y is not None and active_note:
            box_h = 18
            box_w = 40
            bx = ladder_x - box_w // 2
            by = active_marker_y - box_h // 2
            # Dark backing
            cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (18, 22, 28), -1)
            # Glowing border
            cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 140, 255), 1, cv2.LINE_AA)
            # Note label inside cursor
            (ntw, _), _ = cv2.getTextSize(active_note, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(frame, active_note, (ladder_x - ntw // 2, active_marker_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 240, 255), 1, cv2.LINE_AA)

            # Horizontal dotted guideline from hand to pitch ladder
            if melody_hand_pos is not None:
                hx, hy = melody_hand_pos
                # Draw subtle guide line
                cv2.line(frame, (hx + 30, hy), (ladder_x - 24, active_marker_y), (40, 120, 180), 1, cv2.LINE_AA)

        # ---------------------------------------------------------
        # 5. BOTTOM AUDIO WAVEFORM & FOOTER (y = h - 45 .. h - 4)
        # ---------------------------------------------------------
        # Smooth animated golden sine wave across the bottom
        wave_y_center = h - 28
        wave_start_x = 20
        wave_end_x = w - 20
        num_wave_pts = min(120, (wave_end_x - wave_start_x) // 5)

        # Dynamic amplitude responds to active playing
        base_amp = 7.0 if active_note else 2.5
        wave_pts = []

        if waveform_data is not None and len(waveform_data) >= num_wave_pts:
            step = len(waveform_data) // num_wave_pts
            for idx in range(num_wave_pts):
                px = int(wave_start_x + idx * (wave_end_x - wave_start_x) / num_wave_pts)
                sample_val = float(waveform_data[idx * step])
                py = int(wave_y_center - np.clip(sample_val * 16.0, -14.0, 14.0))
                wave_pts.append((px, py))
        else:
            # High-fidelity synthetic oscilloscope sine wave with subtle harmonics
            for idx in range(num_wave_pts):
                px = int(wave_start_x + idx * (wave_end_x - wave_start_x) / num_wave_pts)
                phase = px * 0.035 - elapsed * 4.0
                harmonic = px * 0.070 + elapsed * 2.5
                py = int(wave_y_center + base_amp * math.sin(phase) + (base_amp * 0.35) * math.sin(harmonic))
                wave_pts.append((px, py))

        # Draw audio waveform line
        if len(wave_pts) >= 2:
            pts_arr = np.array(wave_pts, dtype=np.int32).reshape((-1, 1, 2))
            # Delicate golden amber wave
            cv2.polylines(frame, [pts_arr], False, (30, 185, 245), 1, cv2.LINE_AA)

        # Bottom tags
        cv2.putText(frame, "L - CHORDS · palm fwd", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 130, 150), 1, cv2.LINE_AA)

        # Center Room/Space scrubber slider
        space_x = cx - 45
        cv2.putText(frame, "SPACE", (space_x - 38, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 120, 140), 1, cv2.LINE_AA)
        cv2.line(frame, (space_x, h - 38), (space_x + 90, h - 38), (50, 60, 72), 1, cv2.LINE_AA)
        cv2.circle(frame, (space_x + 52, h - 38), 3, (190, 210, 230), -1, cv2.LINE_AA)
        cv2.putText(frame, "ROOM", (space_x + 98, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 120, 140), 1, cv2.LINE_AA)

        r_note_tag = f"R - MELODY · {display_note or instrument_name}"
        (rw2, _), _ = cv2.getTextSize(r_note_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        cv2.putText(frame, r_note_tag, (w - rw2 - 20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 130, 150), 1, cv2.LINE_AA)

        # Footer Strip along bottom edge
        footer_text = "KEY   SCALE   BPM   SWING   SNAP CHORDS   VOICE   CLICK"
        (ftw, _), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        cv2.putText(frame, footer_text, (cx - ftw // 2, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 95, 110), 1, cv2.LINE_AA)
