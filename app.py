"""
app.py
Main Interactive Application for Hand Gesture Musical Instrument.
Integrates Computer Vision, ML Gesture Recognition, Polyphonic Sound Engine,
hand rotation for pitch and volume control, and clean uncluttered camera HUD.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from audio_engine import AudioEngine
from hand_detector import HandDetector, HandInfo
from model import GestureModel
from song_library import SONG_CATALOG, get_jingle_bells, get_song, print_library_to_terminal, save_guide_to_file
from instrument_modes import SongGuideMode, FreeInstrumentMode, GestureScaleMode, AirPianoMode, ThereminMode
from simulator import MockVideoCapture


class HandGestureInstrumentApp:
    """Complete application managing video stream, gesture classification, modes, and clean HUD."""

    MODE_SONG_GUIDE = 1
    MODE_FREE_INSTRUMENT = 2
    MODE_GESTURE_SCALE = 2  # Alias for backward compatibility
    MODE_AIR_PIANO = 3
    MODE_THEREMIN = 4

    def __init__(
        self,
        camera_id: int = 0,
        model_path: str = 'gesture_model.pkl',
        initial_song: str = 'jingle_bells',
        initial_instrument: str = 'bell',
        simulate: bool = False,
        headless: bool = False,
        clean_hud: bool = False,
        lefty: bool = False,
        mode: Optional[str] = None,
        free_instrument: bool = False,
        crop_top: float = 0.0
    ):
        self.camera_id = camera_id
        self.simulate = simulate
        self.headless = headless
        self.clean_hud = clean_hud
        self.melody_hand = "Left" if lefty else "Right"
        self.control_hand = "Right" if lefty else "Left"
        # Digital camera top-cropping (supports ratio 0.0..0.85 or percentage 0..85%)
        crop_val = float(crop_top)
        if 1.0 < crop_val <= 100.0:
            crop_val = crop_val / 100.0
        self.crop_top = float(np.clip(round(crop_val, 2), 0.0, 0.85))
        self.crop_feedback_text = ""
        self.crop_feedback_expire = 0.0

        # Determine initial mode
        if free_instrument or (mode and str(mode).lower() in ('free', 'free_instrument', '2')):
            self.current_mode = self.MODE_FREE_INSTRUMENT
        elif mode and str(mode).lower() in ('piano', '3'):
            self.current_mode = self.MODE_AIR_PIANO
        elif mode and str(mode).lower() in ('theremin', '4'):
            self.current_mode = self.MODE_THEREMIN
        else:
            self.current_mode = self.MODE_SONG_GUIDE

        # Save plain text guide if missing
        if not os.path.exists("SONG_LIBRARY_AND_GESTURES.txt"):
            save_guide_to_file("SONG_LIBRARY_AND_GESTURES.txt")

        # 1. Initialize Audio Engine
        print("[App] Starting Audio Engine...")
        self.audio = AudioEngine()
        self.audio.set_instrument(initial_instrument)

        # 2. Initialize Gesture Recognition Model
        print(f"[App] Loading Gesture Model ({model_path})...")
        self.gesture_model = GestureModel(model_path=model_path)

        # 3. Initialize MediaPipe Hand Detector
        print("[App] Initializing MediaPipe Hand Detector with rotation & adaptive filtering...")
        self.hand_detector = HandDetector(smoothing_factor=0.65)

        # 4. Initialize Modes
        song = get_song(initial_song) or get_jingle_bells()
        self.song_mode = SongGuideMode(self.audio, song=song)
        self.free_mode = FreeInstrumentMode(self.audio)
        self.scale_mode = self.free_mode  # Backward compatibility
        self.piano_mode = AirPianoMode(self.audio)
        self.theremin_mode = ThereminMode(self.audio)

        # 5. Initialize Camera
        self.cap = None
        self._init_camera()

        # Performance & FPS tracking
        self.prev_frame_time = time.time()
        self.fps = 30.0

        # Rotary Clutch: only a closed fist engages rotation controls (volume / pitch shift).
        # Open hands and other gestures lock values so hands can return to center/neutral without changing settings.
        self.locked_volume: float = self.audio.master_volume
        self.locked_pitch_shift: int = 0

    def _init_camera(self):
        """Try opening physical camera; fallback to synthetic simulator if unavailable."""
        if not self.simulate:
            print(f"[App] Opening camera device index {self.camera_id}...")
            cap = cv2.VideoCapture(self.camera_id)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    print(f"[App] Successfully connected to physical camera ({test_frame.shape[1]}x{test_frame.shape[0]}).")
                    return
                cap.release()

            print("[App] Physical camera not found or cannot be opened.")

        # Fallback to Mock / Simulator
        print("[App] Activating Synthetic Hand Simulator & Mock Camera stream.")
        self.simulate = True
        self.cap = MockVideoCapture()

    def cycle_song(self):
        """Cycle to next song in catalog and display note details in terminal."""
        song_keys = list(SONG_CATALOG.keys())
        current_key = 'jingle_bells'
        for k, s in SONG_CATALOG.items():
            if s.title == self.song_mode.song.title:
                current_key = k
                break
        idx = (song_keys.index(current_key) + 1) % len(song_keys)
        next_key = song_keys[idx]
        next_song = SONG_CATALOG[next_key]
        self.song_mode.set_song(next_song)
        print(f"\n[App] Switched song to: {next_song.title}")
        print_library_to_terminal(song_key=next_key)

    def swap_hand_roles(self):
        """Swap melody (pitch) and control (volume) hand roles for right/left-handed users."""
        self.melody_hand, self.control_hand = self.control_hand, self.melody_hand
        print(f"\n[App] Swapped hand roles: Melody Hand = {self.melody_hand} (Pitch), Control Hand = {self.control_hand} (Volume)")

    def adjust_crop(self, delta: float):
        """Dynamically adjust digital top crop ratio (clamped between 0.0 and 0.85) and trigger toast."""
        self.crop_top = float(np.clip(round(self.crop_top + delta, 2), 0.0, 0.85))
        pct = int(round(self.crop_top * 100))
        print(f"\n[App] Camera Top Crop adjusted to: {pct}%")
        self.crop_feedback_text = f"Top Crop: {pct}%"
        self.crop_feedback_expire = time.time() + 1.8

    def crop_and_scale(self, frame: np.ndarray) -> np.ndarray:
        """Crop the top fraction of the camera frame while strictly preserving aspect ratio (no stretching)."""
        if frame is None or frame.size == 0 or self.crop_top <= 0.0:
            return frame
        h, w = frame.shape[:2]
        crop_y = int(round(h * self.crop_top))
        if crop_y <= 0:
            return frame
        if crop_y >= h - 10:
            crop_y = h - 10

        rem_h = h - crop_y
        # Preserve original camera aspect ratio (w / h)
        target_w = int(round(rem_h * (float(w) / float(h))))
        if target_w <= w:
            crop_x1 = max(0, (w - target_w) // 2)
            crop_x2 = min(w, crop_x1 + target_w)
            cropped = frame[crop_y:h, crop_x1:crop_x2]
        else:
            adj_h = int(round(w * (float(h) / float(w))))
            crop_x1 = 0
            crop_x2 = w
            cropped = frame[crop_y:min(h, crop_y + adj_h), 0:w]

        self.last_crop_bounds = (crop_y, h, crop_x1, crop_x2)
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    def draw_top_hud(self, frame: np.ndarray, pitch_shift: int = 0):
        """Draw minimalist top status bar without obscuring video or center note card."""
        if self.clean_hud or self.current_mode in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE):
            return  # Clean camera-only view: no top status bar

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Mode & Instrument on Left corner (x < 180)
        mode_names = {
            self.MODE_SONG_GUIDE: "Song Guide",
            self.MODE_FREE_INSTRUMENT: "Free Instrument",
            self.MODE_AIR_PIANO: "Piano",
            self.MODE_THEREMIN: "Theremin"
        }
        mode_text = f"[{mode_names.get(self.current_mode, '')}] Inst: {self.audio.instrument.upper()}"
        cv2.putText(frame, mode_text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

        # Status text on Right corner (keeps center clear for orange note card)
        vol_pct = int(self.audio.master_volume * 100)
        pitch_str = f"+{pitch_shift}st" if pitch_shift > 0 else f"{pitch_shift}st"
        crop_pct = int(round(self.crop_top * 100))
        crop_str = f" | Crop: {crop_pct}%" if crop_pct > 0 else ""
        status_text = f"Vol: {vol_pct}% | Pitch: {pitch_str}{crop_str} | {int(self.fps)} FPS"
        font_scale = 0.40 if w <= 640 else 0.46
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(frame, status_text, (w - tw - 12, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), 1, cv2.LINE_AA)

    def draw_footer_help(self, frame: np.ndarray):
        """Draw quick shortcuts help bar at the bottom."""
        if self.clean_hud or self.current_mode in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE):
            return  # Clean camera-only view: no footer

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 22), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        help_text = "[1] Song | [2/F] Free Inst | [3] Piano | [4] Theremin | [Space] Tap | [ / ] Crop | [I] Timbre | [Q] Quit"
        cv2.putText(frame, help_text, (15, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    def step(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Process one video frame. Returns (should_continue, rendered_frame)."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Calculate FPS
        now = time.time()
        dt = now - self.prev_frame_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_frame_time = now

        # Mirror camera frame horizontally for intuitive mirror view (if physical webcam)
        if not self.simulate:
            frame = cv2.flip(frame, 1)

        # Apply digital top crop to eliminate physical obstruction and scale to full window
        if self.crop_top > 0.0:
            frame = self.crop_and_scale(frame)

        h, w = frame.shape[:2]

        # 1. Detect Hand Landmarks with rotation and spatial tracking
        if self.simulate and hasattr(self.cap, 'get_simulated_hands'):
            detected_hands = self.cap.get_simulated_hands()
            if self.crop_top > 0.0:
                scale_denom = max(1e-5, 1.0 - self.crop_top)
                for hand in detected_hands:
                    hand.raw_landmarks[:, 1] = (hand.raw_landmarks[:, 1] - self.crop_top) / scale_denom
                    hand.smoothed_landmarks[:, 1] = (hand.smoothed_landmarks[:, 1] - self.crop_top) / scale_denom
        else:
            detected_hands = self.hand_detector.process(frame, is_mirrored=not self.simulate)

        # 2. Predict Gestures and assign hand roles (Melody Hand vs Control Hand)
        melody_hand_obj = None
        control_hand_obj = None
        other_hand_obj = None

        for hand in detected_hands:
            gest, conf = self.gesture_model.predict(hand.smoothed_landmarks)
            hand.gesture = gest
            hand.gesture_confidence = conf

            if hand.handedness == self.melody_hand:
                if melody_hand_obj is None or conf > melody_hand_obj.gesture_confidence:
                    melody_hand_obj = hand
            elif hand.handedness == self.control_hand:
                if control_hand_obj is None or conf > control_hand_obj.gesture_confidence:
                    control_hand_obj = hand
            else:
                other_hand_obj = hand

        primary_gesture = "None"
        primary_conf = 0.0
        active_pitch_shift = 0

        primary_tap = False

        # Free mode: clean hand view (no bounding box or badge rectangle)
        is_free_mode = (self.current_mode in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE))
        draw_box = not (is_free_mode or self.clean_hud)

        melody_hand_pos = None
        chord_gesture = None

        # Handle Single-Hand Play (1 hand in view) vs Dual-Hand Play (>= 2 hands in view)
        if len(detected_hands) == 1:
            # Single hand in view: acts as Melody Hand, pitch shifting enabled via Closed Fist, volume intact
            single_hand = detected_hands[0]
            primary_gesture = single_hand.gesture
            primary_conf = single_hand.gesture_confidence
            primary_tap = getattr(single_hand, 'is_tap', False)

            # ROTARY CLUTCH: Only a closed fist controls pitch shift!
            # Open hands and other gestures lock pitch shift; turning open hand has no effect.
            if single_hand.gesture == 'Fist':
                self.locked_pitch_shift = single_hand.pitch_shift_semitones
            active_pitch_shift = self.locked_pitch_shift

            if hasattr(single_hand, 'smoothed_landmarks') and single_hand.smoothed_landmarks is not None:
                melody_hand_pos = (int(single_hand.smoothed_landmarks[0, 0] * w), int(single_hand.smoothed_landmarks[0, 1] * h))
            self.hand_detector.draw_hand_hud(frame, single_hand, is_control=False, draw_box=draw_box)
        else:
            # Multiple hands in view: control hand dials volume & triggers chords, melody hand plays notes
            if control_hand_obj is not None:
                # ROTARY CLUTCH: Only a closed fist controls volume!
                # Open hands (e.g. Open Palm, chords) lock volume; turning open hand has no effect.
                if control_hand_obj.gesture == 'Fist':
                    self.locked_volume = control_hand_obj.volume_level
                    self.audio.set_master_volume(self.locked_volume)
                else:
                    self.audio.set_master_volume(self.locked_volume)
                chord_gesture = control_hand_obj.gesture

            if melody_hand_obj is not None:
                primary_gesture = melody_hand_obj.gesture
                primary_conf = melody_hand_obj.gesture_confidence
                primary_tap = getattr(melody_hand_obj, 'is_tap', False)
                # ROTARY CLUTCH: Only a closed fist controls pitch shift!
                if melody_hand_obj.gesture == 'Fist':
                    self.locked_pitch_shift = melody_hand_obj.pitch_shift_semitones
                active_pitch_shift = self.locked_pitch_shift
                if hasattr(melody_hand_obj, 'smoothed_landmarks') and melody_hand_obj.smoothed_landmarks is not None:
                    melody_hand_pos = (int(melody_hand_obj.smoothed_landmarks[0, 0] * w), int(melody_hand_obj.smoothed_landmarks[0, 1] * h))
            elif other_hand_obj is not None:
                primary_gesture = other_hand_obj.gesture
                primary_conf = other_hand_obj.gesture_confidence
                primary_tap = getattr(other_hand_obj, 'is_tap', False)
                if other_hand_obj.gesture == 'Fist':
                    self.locked_pitch_shift = other_hand_obj.pitch_shift_semitones
                active_pitch_shift = self.locked_pitch_shift
                if hasattr(other_hand_obj, 'smoothed_landmarks') and other_hand_obj.smoothed_landmarks is not None:
                    melody_hand_pos = (int(other_hand_obj.smoothed_landmarks[0, 0] * w), int(other_hand_obj.smoothed_landmarks[0, 1] * h))

            for hand in detected_hands:
                is_ctrl = (hand is control_hand_obj)
                self.hand_detector.draw_hand_hud(frame, hand, is_control=is_ctrl, draw_box=draw_box)

        # Draw Global Top HUD for AirPiano and Theremin (Free Mode and Song Guide use CyberneticHUD)
        if self.current_mode not in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE, self.MODE_SONG_GUIDE):
            self.draw_top_hud(frame, pitch_shift=active_pitch_shift)

        # 3. Update Current Active Mode
        if self.current_mode == self.MODE_SONG_GUIDE:
            self.song_mode.update(primary_gesture, primary_conf, pitch_shift_semitones=active_pitch_shift, is_tap=primary_tap)
            self.song_mode.draw_hud(frame)

        elif self.current_mode in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE):
            self.free_mode.update(
                primary_gesture, primary_conf,
                pitch_shift_semitones=active_pitch_shift,
                is_tap=primary_tap,
                chord_gesture=chord_gesture,
                hand_pos=melody_hand_pos
            )
            self.free_mode.draw_hud(frame)

        elif self.current_mode == self.MODE_AIR_PIANO:
            self.piano_mode.update(detected_hands, w, h)
            self.piano_mode.draw_hud(frame)

        elif self.current_mode == self.MODE_THEREMIN:
            self.theremin_mode.update(detected_hands, w, h)
            self.theremin_mode.draw_hud(frame)

        # Draw Global Footer Help for AirPiano and Theremin
        if self.current_mode not in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE, self.MODE_SONG_GUIDE):
            self.draw_footer_help(frame)

        # Draw temporary crop adjustment toast if active (positioned below note cards and prompt)
        if time.time() < self.crop_feedback_expire and self.crop_feedback_text:
            text = self.crop_feedback_text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            cx = w // 2
            x1 = cx - tw // 2 - 14
            y1 = 135
            x2 = cx + tw // 2 + 14
            y2 = y1 + th + 16
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, text, (cx - tw // 2, y1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

        return True, frame

    def run(self, max_frames: Optional[int] = None):
        """Main application loop."""
        print("\n" + "=" * 65)
        print(" HAND GESTURE INSTRUMENT")
        print(" Minimalist Camera HUD: Shows Hand Detection + Orange Note + Symbol")
        print(" Hand Controls (Fist Clutch Rotary System):")
        print("   * RIGHT HAND: Plays notes | Closed Fist + Rotate to Shift Pitch (Open hand locks pitch)")
        print("   * LEFT HAND:  Chords & Volume | Closed Fist + Rotate to Adjust Volume (Open hand locks volume)")
        print(" Keyboard Shortcuts:")
        print("   [1]   Song Guide Mode (Interactive song tutorial, Orange Note + Symbol)")
        print("   [2/F] Free Instrument Mode (Plays any gesture freely, Orange Note + Symbol)")
        print("   [3]   Air Piano (Virtual keyboard)")
        print("   [4]   Air Theremin (Continuous spatial pitch & volume)")
        print("   [[]   Decrease Camera Top Crop (-5%)")
        print("   []]   Increase Camera Top Crop (+5%)")
        print("   [S]   Next Song (Prints notes to terminal)")
        print("   [L]   Print Songs Library & Gestures Guide in Terminal")
        print("   [I]   Next Instrument Timbre (Bell, Piano, Violin, Synth)")
        print("   [P]   Toggle Autoplay Demo")
        print("   [R]   Reset Song & Unlock Pitch/Volume")
        print("   [Q]   Quit")
        print(" Full guide written to: SONG_LIBRARY_AND_GESTURES.txt")
        print("=" * 65 + "\n")

        win_name = "Hand Gesture Instrument"
        if not self.headless:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, 720)

        frame_count = 0
        try:
            while True:
                success, frame = self.step()
                if not success:
                    break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    print(f"[App] Reached requested {max_frames} frames limit.")
                    break

                if not self.headless:
                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (27, ord('q'), ord('Q')):
                        print("[App] Exiting on user request.")
                        break
                    elif key == ord('1'):
                        self.current_mode = self.MODE_SONG_GUIDE
                    elif key in (ord('2'), ord('f'), ord('F')):
                        self.current_mode = self.MODE_FREE_INSTRUMENT
                    elif key == ord('3'):
                        self.current_mode = self.MODE_AIR_PIANO
                    elif key == ord('4'):
                        self.current_mode = self.MODE_THEREMIN
                    elif key in (ord('s'), ord('S')):
                        self.cycle_song()
                    elif key in (ord('l'), ord('L')):
                        print_library_to_terminal()
                    elif key in (ord('i'), ord('I')):
                        new_inst = self.audio.cycle_instrument()
                        print(f"[App] Instrument changed to: {new_inst}")
                    elif key in (ord('h'), ord('H')):
                        self.swap_hand_roles()
                    elif key in (ord('c'), ord('C')):
                        self.clean_hud = not self.clean_hud
                        print(f"[App] Clean Camera HUD: {'ENABLED' if self.clean_hud else 'DISABLED'}")
                    elif key in (ord('p'), ord('P')):
                        self.song_mode.autoplay = not self.song_mode.autoplay
                        print(f"[App] Autoplay: {self.song_mode.autoplay}")
                    elif key in (ord('r'), ord('R')):
                        self.song_mode.reset()
                        self.free_mode.reset()
                        self.locked_pitch_shift = 0
                        self.locked_volume = 0.85
                        self.audio.set_master_volume(0.85)
                        print("[App] Song and free mode reset (pitch shift & volume unlocked to default).")
                    elif key in (ord('['), ord('{'), ord('-'), ord('_')):
                        self.adjust_crop(-0.05)
                    elif key in (ord(']'), ord('}'), ord('='), ord('+')):
                        self.adjust_crop(0.05)
                    elif key == ord(' '):
                        if self.current_mode == self.MODE_SONG_GUIDE:
                            self.song_mode.trigger_tap()
                        elif self.current_mode in (self.MODE_FREE_INSTRUMENT, self.MODE_GESTURE_SCALE):
                            self.free_mode.trigger_tap()
                        elif self.current_mode == self.MODE_AIR_PIANO:
                            self.piano_mode.trigger_tap()

        finally:
            self.close()

    def close(self):
        """Cleanly release camera and sound resources."""
        print("[App] Releasing resources...")
        if self.cap:
            self.cap.release()
        self.audio.stop()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("[App] Closed successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description="Hand Gesture Musical Instrument")
    parser.add_argument('--camera', type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument('--model', type=str, default='gesture_model.pkl', help="Path to trained model")
    parser.add_argument('--song', type=str, default='jingle_bells', help="Initial song name")
    parser.add_argument('--instrument', type=str, default='bell', help="Initial instrument (bell, piano, violin, synth)")
    parser.add_argument('--simulate', action='store_true', help="Force synthetic simulation mode")
    parser.add_argument('--headless', action='store_true', help="Run without opening GUI window")
    parser.add_argument('--clean', action='store_true', help="Clean camera-only HUD mode (hides top bar & footer)")
    parser.add_argument('--lefty', action='store_true', help="Left-handed mode (Left Hand = Melody/Pitch, Right Hand = Volume)")
    parser.add_argument('--free', '--free-instrument', dest='free_mode', action='store_true',
                        help="Launch directly in Free Instrument mode (no guidance, freely plays whatever you gesture)")
    parser.add_argument('--mode', type=str, default=None,
                        choices=['song', 'free', 'free_instrument', 'piano', 'theremin'],
                        help="Initial mode (song, free, piano, theremin)")
    parser.add_argument('--crop-top', type=float, default=0.0,
                        help="Fraction of top camera frame to crop out (default: 0.0, no crop)")
    parser.add_argument('--frames', type=int, default=None, help="Stop after N frames (for tests)")
    parser.add_argument('--library', action='store_true', help="Print songs library and gestures guide to terminal and exit")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.library:
        print_library_to_terminal()
        return

    app = HandGestureInstrumentApp(
        camera_id=args.camera,
        model_path=args.model,
        initial_song=args.song,
        initial_instrument=args.instrument,
        simulate=args.simulate,
        headless=args.headless,
        clean_hud=args.clean,
        lefty=args.lefty,
        mode=args.mode,
        free_instrument=args.free_mode,
        crop_top=args.crop_top
    )
    app.run(max_frames=args.frames)


if __name__ == '__main__':
    main()
