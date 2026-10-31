"""
tests/test_free_instrument_and_accuracy.py
Comprehensive test suite verifying:
1. Free Instrument mode plays any gesture directly without requiring guidance or target notes
2. Note display in orange and gesture symbol rendering
3. Hand rotation pitch shift (semitone transposition) in Free Instrument mode
4. Volume dial via control hand in Free Instrument mode
5. Preservation of guided mode (SongGuideMode)
6. App CLI flags (--free, --free-instrument, --mode free) and shortcut keys
7. Multi-hand gesture recognition accuracy across all 13 gesture classes
"""

import math
import numpy as np
import pytest

from audio_engine import AudioEngine, note_to_freq, shift_note_semitones
from hand_detector import HandInfo
from instrument_modes import FreeInstrumentMode, SongGuideMode, GESTURE_TO_NOTE_MAP
from model import GestureModel
from song_library import get_jingle_bells, GESTURE_TO_SYMBOL_MAP
from trainer import generate_canonical_hand, augment_landmarks
from app import HandGestureInstrumentApp


@pytest.fixture
def audio():
    engine = AudioEngine()
    yield engine
    engine.stop()


def test_free_instrument_mode_plays_any_gesture_freely(audio):
    """Verify that FreeInstrumentMode plays any gesture directly without requiring guidance."""
    mode = FreeInstrumentMode(audio)
    assert mode.active_note is None

    # Test all 13 gestures playing their respective notes
    test_cases = [
        ('Fist', 'C4'),
        ('Index Point', 'D4'),
        ('Peace', 'E4'),
        ('Three', 'F4'),
        ('Four', 'G4'),
        ('Open Palm', 'A4'),
        ('Thumbs Up', 'B4'),
        ('Pinch', 'C5'),
        ('Rock On', 'D5'),
        ('Hang Loose', 'E5'),
        ('Gun', 'F5'),
        ('Pinky Up', 'G5'),
        ('OK Sign', 'A5'),
    ]

    for gesture, expected_note in test_cases:
        mode.update(gesture, confidence=0.92)
        assert mode.active_note == expected_note, f"Failed for {gesture}: got {mode.active_note}, expected {expected_note}"
        assert mode.active_gesture == gesture
        assert mode.active_symbol == GESTURE_TO_SYMBOL_MAP[gesture]
        assert mode.strike_count >= 1

    # Hand relaxation clears active note
    mode.update('None', confidence=0.0)
    assert mode.active_note is None
    assert mode.active_gesture == "None"


def test_free_instrument_pitch_shift(audio):
    """Verify that hand rotation pitch shift transposes notes in FreeInstrumentMode."""
    mode = FreeInstrumentMode(audio)

    # Fist (base C4) with +2 semitones -> D4
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=2)
    assert mode.active_note == 'D4'
    assert mode.active_base_note == 'C4'
    assert mode.pitch_shift_semitones == 2

    # Fist with +4 semitones -> E4
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=4)
    assert mode.active_note == 'E4'

    # Fist with -2 semitones -> A#3 / Bb3
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=-2)
    assert mode.active_note in ('A#3', 'Bb3')


def test_free_instrument_orange_hud_rendering(audio):
    """Verify FreeInstrumentMode renders detected note and symbol in orange on the HUD."""
    mode = FreeInstrumentMode(audio)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Resting state: draws ready card
    mode.draw_hud(frame)
    assert np.max(frame) > 0  # Frame was modified with graphics

    # Check orange color presence: vivid orange BGR is (0, 140, 255)
    # Channel 2 (Red) and Channel 1 (Green) should be active, Blue should be 0 or low
    has_orange_pixels = np.any((frame[:, :, 2] > 200) & (frame[:, :, 1] > 100) & (frame[:, :, 0] < 50))
    assert has_orange_pixels

    # 2. Active note state
    frame_active = np.zeros((480, 640, 3), dtype=np.uint8)
    mode.update('Peace', confidence=0.95, pitch_shift_semitones=1)
    mode.draw_hud(frame_active)
    assert mode.active_note == 'F4'  # Peace (E4) + 1 semitone -> F4

    has_active_orange = np.any((frame_active[:, :, 2] > 200) & (frame_active[:, :, 1] > 100) & (frame_active[:, :, 0] < 50))
    assert has_active_orange


def test_guided_mode_kept_and_functional(audio):
    """Verify SongGuideMode is preserved, requiring correct gesture before sounding/advancing."""
    song_mode = SongGuideMode(audio, song=get_jingle_bells())
    assert song_mode.current_note_idx == 0
    target_first = song_mode.get_current_note()
    assert target_first.note == 'E4'
    assert target_first.gesture == 'Peace'

    # Wrong gesture should NOT advance the note
    song_mode.update('Fist', confidence=0.95)
    assert song_mode.current_note_idx == 0
    assert "Detected: Fist" in song_mode.feedback_text

    # Correct gesture advances the note
    song_mode.update('Peace', confidence=0.95)
    assert song_mode.current_note_idx == 1
    assert song_mode.score == 100


def test_app_modes_and_cli_initialization():
    """Verify HandGestureInstrumentApp supports initial mode selection and switching."""
    # Default is guided mode
    app_default = HandGestureInstrumentApp(simulate=True, headless=True)
    assert app_default.current_mode == HandGestureInstrumentApp.MODE_SONG_GUIDE
    app_default.close()

    # Free instrument via flag
    app_free = HandGestureInstrumentApp(simulate=True, headless=True, free_instrument=True)
    assert app_free.current_mode == HandGestureInstrumentApp.MODE_FREE_INSTRUMENT
    app_free.close()

    # Free instrument via mode string
    app_free_str = HandGestureInstrumentApp(simulate=True, headless=True, mode='free')
    assert app_free_str.current_mode == HandGestureInstrumentApp.MODE_FREE_INSTRUMENT
    app_free_str.close()

    # Aliased gesture scale mode constant matches free instrument mode
    assert HandGestureInstrumentApp.MODE_GESTURE_SCALE == HandGestureInstrumentApp.MODE_FREE_INSTRUMENT


def test_app_free_instrument_step_execution():
    """Verify HandGestureInstrumentApp processes frames in Free Instrument mode."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, free_instrument=True)

    # Step through frames
    for _ in range(10):
        ok, frame = app.step()
        assert ok is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    # Simulate melody hand playing Fist (C4)
    hand_right = HandInfo(generate_canonical_hand('Fist'), handedness='Right')
    app.cap.get_simulated_hands = lambda: [hand_right]

    ok, frame = app.step()
    assert ok is True
    assert app.free_mode.active_note == 'C4'
    assert app.free_mode.active_symbol == 'Closed Fist'

    app.close()


def test_accuracy_across_all_13_gestures():
    """Verify that gesture recognition achieves >= 95% accuracy on augmented hands across all 13 classes."""
    model = GestureModel()
    rng = np.random.RandomState(42)

    gestures = [
        'Fist', 'Index Point', 'Peace', 'Three', 'Four',
        'Open Palm', 'Thumbs Up', 'Pinch', 'Rock On',
        'Hang Loose', 'Gun', 'Pinky Up', 'OK Sign'
    ]

    total_correct = 0
    total_tested = 0

    for i, g in enumerate(gestures):
        canonical = generate_canonical_hand(g)
        correct_for_class = 0
        samples_per_class = 30
        class_rng = np.random.RandomState(42 + i * 17)
        for _ in range(samples_per_class):
            aug = augment_landmarks(canonical, class_rng)
            pred, conf = model.predict(aug, use_history=False)
            if pred == g:
                correct_for_class += 1

        acc_class = correct_for_class / samples_per_class
        assert acc_class >= 0.85, f"Class {g} accuracy too low: {acc_class*100:.1f}%"
        total_correct += correct_for_class
        total_tested += samples_per_class

    overall_acc = total_correct / total_tested
    assert overall_acc >= 0.95, f"Overall gesture recognition accuracy too low: {overall_acc*100:.2f}%"


def test_control_hand_volume_in_free_instrument_mode():
    """Verify that only Closed Fist controls volume, and Open Palm locks volume even when returned to center."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, free_instrument=True)
    initial_vol = app.audio.master_volume

    hand_melody = HandInfo(generate_canonical_hand('Peace'), handedness='Right')

    # 1. Open hand (Open Palm) tilted left (-60 deg): should have NO EFFECT on volume
    raw_palm = generate_canonical_hand('Open Palm')
    wrist = raw_palm[0].copy()
    angle_rad = math.radians(-60.0)
    R2 = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]], dtype=np.float32)
    rot_palm = raw_palm.copy()
    rot_palm[:, :2] = (R2 @ (raw_palm[:, :2] - wrist[:2]).T).T + wrist[:2]
    hand_control_open = HandInfo(rot_palm, handedness='Left')

    app.cap.get_simulated_hands = lambda: [hand_melody, hand_control_open]
    ok, frame = app.step()
    assert ok is True
    assert app.audio.master_volume == initial_vol, "Open hand tilt must NOT change volume!"

    # 2. Closed Fist tilted left (-60 deg = mute/low volume): engages dial and changes volume
    raw_fist = generate_canonical_hand('Fist')
    wrist_f = raw_fist[0].copy()
    rot_fist_quiet = raw_fist.copy()
    rot_fist_quiet[:, :2] = (R2 @ (raw_fist[:, :2] - wrist_f[:2]).T).T + wrist_f[:2]
    hand_control_quiet = HandInfo(rot_fist_quiet, handedness='Left')

    app.cap.get_simulated_hands = lambda: [hand_melody, hand_control_quiet]
    ok, frame = app.step()
    assert ok is True
    assert app.audio.master_volume <= 0.10, "Closed Fist tilt must adjust volume to quiet!"

    # 3. Open hand returned to upright center (0 deg / 50% marker): volume must STAY quiet!
    hand_control_neutral_open = HandInfo(generate_canonical_hand('Open Palm'), handedness='Left')
    app.cap.get_simulated_hands = lambda: [hand_melody, hand_control_neutral_open]
    ok, frame = app.step()
    assert ok is True
    assert app.audio.master_volume <= 0.10, "Opening hand and returning to 50% marker must lock volume!"

    # 4. Closed Fist tilted right (+60 deg = full volume): engages dial and increases volume
    angle_rad_loud = math.radians(60.0)
    R2_loud = np.array([[math.cos(angle_rad_loud), -math.sin(angle_rad_loud)], [math.sin(angle_rad_loud), math.cos(angle_rad_loud)]], dtype=np.float32)
    rot_fist_loud = raw_fist.copy()
    rot_fist_loud[:, :2] = (R2_loud @ (raw_fist[:, :2] - wrist_f[:2]).T).T + wrist_f[:2]
    hand_control_loud = HandInfo(rot_fist_loud, handedness='Left')

    app.cap.get_simulated_hands = lambda: [hand_melody, hand_control_loud]
    ok, frame = app.step()
    assert ok is True
    assert app.audio.master_volume >= 0.95, "Closed Fist tilt right must adjust volume to loud!"
    app.close()


def test_free_instrument_pitch_shift_flicker_protection(audio):
    """Verify rapid rotation jitter across semitone boundary does not machine-gun trigger notes."""
    mode = FreeInstrumentMode(audio)

    # Initial strike: Fist with 0 pitch shift -> C4
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=0)
    assert mode.strike_count == 1
    assert mode.active_note == 'C4'

    # Immediate jitter to +1 semitone within 5ms (less than 0.12s cooldown) should NOT fire a second note
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=1)
    assert mode.strike_count == 1  # Rate limited

    # But changing gesture class triggers immediately without delay
    mode.update('Peace', confidence=0.95, pitch_shift_semitones=0)
    assert mode.strike_count == 2
    assert mode.active_note == 'E4'


def test_single_hand_left_play_unmuted_with_pitch_shift():
    """Verify single-hand play: Open hand locks pitch, Closed Fist tunes pitch, and audio is unmuted."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, free_instrument=True)
    initial_vol = app.audio.master_volume

    # 1. Open hand (Peace) tilted clockwise (+24 deg): should NOT change pitch shift
    raw_peace = generate_canonical_hand('Peace')
    wrist_p = raw_peace[0].copy()
    angle_rad = math.radians(24.0)  # +2 semitones
    R2 = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]], dtype=np.float32)
    rot_peace = raw_peace.copy()
    rot_peace[:, :2] = (R2 @ (raw_peace[:, :2] - wrist_p[:2]).T).T + wrist_p[:2]
    single_peace_tilted = HandInfo(rot_peace, handedness='Left')

    app.cap.get_simulated_hands = lambda: [single_peace_tilted]
    ok, frame = app.step()
    assert ok is True
    # Volume should NOT be muted by single hand tilt
    assert app.audio.master_volume == initial_vol
    # Open hand tilt must NOT shift pitch: Peace plays natural E4
    assert app.free_mode.pitch_shift_semitones == 0
    assert app.free_mode.active_note == 'E4'

    # 2. Closed Fist tilted clockwise (+24 deg): engages dial and shifts pitch to +2
    raw_fist = generate_canonical_hand('Fist')
    wrist_f = raw_fist[0].copy()
    rot_fist = raw_fist.copy()
    rot_fist[:, :2] = (R2 @ (raw_fist[:, :2] - wrist_f[:2]).T).T + wrist_f[:2]
    single_fist_tilted = HandInfo(rot_fist, handedness='Left')

    app.cap.get_simulated_hands = lambda: [single_fist_tilted]
    ok, frame = app.step()
    assert ok is True
    assert app.free_mode.pitch_shift_semitones == 2

    # 3. Switching back to Peace: pitch shift stays locked at +2 -> Peace (E4) + 2st = F#4!
    single_peace_upright = HandInfo(generate_canonical_hand('Peace'), handedness='Left')
    app.cap.get_simulated_hands = lambda: [single_peace_upright]
    ok, frame = app.step()
    assert ok is True
    assert app.free_mode.pitch_shift_semitones == 2
    assert app.free_mode.active_note == 'F#4'

    app.close()


def test_free_instrument_persistent_display_anti_flicker(audio):
    """Verify that relaxing hand between notes maintains display of last played note instead of flickering off."""
    mode = FreeInstrumentMode(audio)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Strike note
    mode.update('Pinch', confidence=0.95)
    assert mode.active_note == 'C5'
    assert mode.display_note == 'C5'

    # Relax hand (confidence drops / gesture None)
    mode.update('None', confidence=0.0)
    assert mode.active_note is None

    # HUD should still stably display the recently played note (C5) in orange without expiring
    mode.draw_hud(frame)
    has_orange = np.any((frame[:, :, 2] > 200) & (frame[:, :, 1] > 100) & (frame[:, :, 0] < 50))
    assert has_orange


def test_free_instrument_last_note_persists_indefinitely(audio):
    """Verify that last played note does NOT expire after 1.2s, persisting indefinitely until next note or reset."""
    import time
    mode = FreeInstrumentMode(audio)

    mode.update('Fist', confidence=0.95)
    assert mode.active_note == 'C4'
    assert mode.display_note == 'C4'

    # Relax hand and advance simulated clock far past 1.2s (e.g. 10.0s later)
    mode.update('None', confidence=0.0)
    assert mode.active_note is None
    # Even if display_expire is in the past, draw_hud continues to display the last hit note
    mode.display_expire = time.time() - 100.0

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mode.draw_hud(frame)
    has_orange = np.any((frame[:, :, 2] > 200) & (frame[:, :, 1] > 100) & (frame[:, :, 0] < 50))
    assert has_orange

    # Reset cleans it back to initial title
    mode.reset()
    assert mode.display_note is None
    assert mode.active_note is None


def test_free_instrument_no_dark_card_box_overlay(audio):
    """Verify that FreeInstrumentMode renders pure text without drawing any dark rectangle card overlays."""
    mode = FreeInstrumentMode(audio)
    # Test blank white canvas
    canvas = np.full((480, 640, 3), 200, dtype=np.uint8)
    mode.update('Peace', confidence=0.95)
    mode.draw_hud(canvas)

    # In previous implementation, a large dark card rectangle (18, 16, 24) blended with canvas was drawn:
    # cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 16, 24), -1) -> darkened the entire card_w x 75 box
    # With pure text rendering, areas away from text glyphs (e.g. corners of the old card at x=200, y=20) remain 200
    assert canvas[15, 200, 0] == 200
    assert canvas[15, 200, 1] == 200
    assert canvas[15, 200, 2] == 200


def test_simulator_does_not_draw_brown_palm_polygon():
    """Verify HandSimulator renders hand bones and joints without brown palm polygon fill (70, 60, 90)."""
    from simulator import HandSimulator
    sim = HandSimulator()
    frame, lms, gesture = sim.render_frame()

    # The brown palm fill color was exactly (70, 60, 90) in BGR
    # Check that no pixel has this exact brown color
    brown_pixels = np.all(frame == np.array([70, 60, 90], dtype=np.uint8), axis=2)
    assert not np.any(brown_pixels), "Simulator should not contain brown palm polygon fill pixels"


def test_app_free_mode_omits_top_and_footer_hud():
    """Verify HandGestureInstrumentApp suppresses top status bar and bottom shortcuts bar in Free Instrument mode."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, free_instrument=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # In Free Instrument mode, draw_top_hud and draw_footer_help should do nothing
    app.draw_top_hud(frame, pitch_shift=0)
    assert np.max(frame[:30, :, :]) == 0, "Top 30 pixels should remain untouched in Free mode"

    app.draw_footer_help(frame)
    assert np.max(frame[-22:, :, :]) == 0, "Bottom 22 pixels should remain untouched in Free mode"

    # Switching to Song Guide mode restores them
    app.current_mode = app.MODE_SONG_GUIDE
    app.draw_top_hud(frame, pitch_shift=0)
    assert np.max(frame[:30, :, :]) > 0, "Top bar should be rendered in Song Guide mode"
    app.draw_footer_help(frame)
    assert np.max(frame[-22:, :, :]) > 0, "Footer should be rendered in Song Guide mode"

    app.close()


def test_free_instrument_reset_clears_persisted_note(audio):
    """Verify that calling reset() on FreeInstrumentMode clears both active and persisted display notes."""
    mode = FreeInstrumentMode(audio)
    mode.update('Peace', confidence=0.95)
    assert mode.active_note == 'E4'
    assert mode.display_note == 'E4'

    # Relax hand
    mode.update('None', confidence=0.0)
    assert mode.active_note is None
    assert mode.display_note == 'E4'

    # Calling reset clears persisted note
    mode.reset()
    assert mode.active_note is None
    assert mode.display_note is None



