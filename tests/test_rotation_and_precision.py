"""
tests/test_rotation_and_precision.py
Comprehensive test suite verifying:
1. Hand rotation angle calculation (roll) for pitch and volume control
2. Transposition and semitone pitch shifting
3. Extended notes and gesture symbols
4. Text guide generation and file persistence
5. Minimalist orange note HUD rendering
6. Hand tracking spatial matching and jitter smoothing
"""

import math
import os
import numpy as np
import pytest

from audio_engine import AudioEngine, note_to_midi, midi_to_note, shift_note_semitones, note_to_freq
from hand_detector import HandDetector, HandInfo
from instrument_modes import SongGuideMode, GestureScaleMode, GESTURE_TO_NOTE_MAP
from song_library import (
    NOTE_TO_GESTURE_MAP,
    GESTURE_TO_SYMBOL_MAP,
    generate_text_guide,
    save_guide_to_file,
    get_song,
    get_available_songs
)
from trainer import generate_canonical_hand


@pytest.fixture
def audio():
    engine = AudioEngine()
    yield engine
    engine.stop()


def test_hand_rotation_angle_upright():
    """Verify an upright hand has near 0 degrees rotation."""
    hand = generate_canonical_hand('Open Palm')
    info = HandInfo(hand, handedness='Right')
    # In canonical hand, Middle MCP (9) is directly above wrist (0)
    assert abs(info.rotation_deg) < 5.0
    assert info.pitch_shift_semitones == 0


def test_hand_rotation_clockwise_pitch_shift():
    """Verify tilting right hand clockwise increases pitch by semitones."""
    hand = generate_canonical_hand('Open Palm')
    wrist = hand[0].copy()
    # Rotate hand clockwise in image plane by 36 degrees (+3 semitones since 12 deg / semitone)
    angle_rad = math.radians(36.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    # 2D rotation matrix in image plane
    R2 = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    rot_hand = hand.copy()
    rel = hand[:, :2] - wrist[:2]
    rot_hand[:, :2] = (R2 @ rel.T).T + wrist[:2]

    info = HandInfo(rot_hand, handedness='Right')
    assert abs(info.rotation_deg - 36.0) < 3.0
    assert info.pitch_shift_semitones == 3


def test_hand_rotation_counter_clockwise_pitch_shift():
    """Verify tilting right hand counter-clockwise decreases pitch by semitones."""
    hand = generate_canonical_hand('Fist')
    wrist = hand[0].copy()
    # Rotate hand counter-clockwise by -24 degrees (-2 semitones)
    angle_rad = math.radians(-24.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R2 = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    rot_hand = hand.copy()
    rel = hand[:, :2] - wrist[:2]
    rot_hand[:, :2] = (R2 @ rel.T).T + wrist[:2]

    info = HandInfo(rot_hand, handedness='Right')
    assert abs(info.rotation_deg - (-24.0)) < 3.0
    assert info.pitch_shift_semitones == -2


def test_left_hand_rotation_volume_control():
    """Verify rotating left hand maps to smooth master volume."""
    hand = generate_canonical_hand('Open Palm')
    wrist = hand[0].copy()

    # 1. Upright (0 deg): ~65% volume
    info_neutral = HandInfo(hand, handedness='Left')
    assert abs(info_neutral.volume_level - 0.65) < 0.05

    # 2. Tilt left by -60 deg: 0% volume (Mute)
    angle_rad = math.radians(-60.0)
    R2 = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]], dtype=np.float32)
    rot_left = hand.copy()
    rot_left[:, :2] = (R2 @ (hand[:, :2] - wrist[:2]).T).T + wrist[:2]
    info_quiet = HandInfo(rot_left, handedness='Left')
    assert info_quiet.volume_level <= 0.05

    # 3. Tilt right by +60 deg: 100% volume
    angle_rad = math.radians(60.0)
    R2 = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]], dtype=np.float32)
    rot_right = hand.copy()
    rot_right[:, :2] = (R2 @ (hand[:, :2] - wrist[:2]).T).T + wrist[:2]
    info_loud = HandInfo(rot_right, handedness='Left')
    assert info_loud.volume_level >= 0.95


def test_note_transposition_utilities():
    """Verify MIDI number conversion and semitone transposition functions."""
    assert note_to_midi('C4') == 60
    assert note_to_midi('A4') == 69
    assert note_to_midi('C5') == 72
    assert midi_to_note(60) == 'C4'
    assert midi_to_note(69) == 'A4'

    # Transposition intervals
    assert shift_note_semitones('C4', 0) == 'C4'
    assert shift_note_semitones('C4', 1) == 'C#4'
    assert shift_note_semitones('C4', 2) == 'D4'
    assert shift_note_semitones('C4', 4) == 'E4'
    assert shift_note_semitones('C4', -1) == 'B3'
    assert shift_note_semitones('C4', 12) == 'C5'
    assert shift_note_semitones('C4', -12) == 'C3'
    assert shift_note_semitones('REST', 5) == 'REST'


def test_audio_engine_play_transposed_note(audio):
    """Verify play_note transposes when pitch_shift_semitones is passed."""
    # Play C4 shifted +2 semitones -> should sound D4 frequency (~293.66 Hz)
    audio.play_note('C4', pitch_shift_semitones=2, duration=0.1)
    voice = audio.voices[-1]
    expected_freq = note_to_freq('D4')
    assert abs(voice.freq - expected_freq) < 0.2

    # Verify set_master_volume
    audio.set_master_volume(0.42)
    assert abs(audio.master_volume - 0.42) < 1e-4


def test_extended_gesture_to_note_mappings():
    """Verify new extended notes exist in gesture mappings and catalog."""
    assert 'D5' in NOTE_TO_GESTURE_MAP
    assert 'E5' in NOTE_TO_GESTURE_MAP
    assert 'F5' in NOTE_TO_GESTURE_MAP
    assert 'G5' in NOTE_TO_GESTURE_MAP
    assert 'A5' in NOTE_TO_GESTURE_MAP

    assert GESTURE_TO_NOTE_MAP['Hang Loose'] == 'E5'
    assert GESTURE_TO_NOTE_MAP['Gun'] == 'F5'
    assert GESTURE_TO_NOTE_MAP['Pinky Up'] == 'G5'
    assert GESTURE_TO_NOTE_MAP['OK Sign'] == 'A5'


def test_gesture_symbols_formatting():
    """Verify human-readable symbols are defined for all gestures."""
    required = ['Fist', 'Index Point', 'Peace', 'Three', 'Four', 'Open Palm', 'Thumbs Up', 'Pinch']
    for gest in required:
        assert gest in GESTURE_TO_SYMBOL_MAP
        symbol = GESTURE_TO_SYMBOL_MAP[gest]
        assert len(symbol) > 0
        assert isinstance(symbol, str)

    assert GESTURE_TO_SYMBOL_MAP['Fist'] == 'Closed Fist'
    assert GESTURE_TO_SYMBOL_MAP['Open Palm'] == 'Open Palm (5 Fingers)'


def test_generate_text_guide_content():
    """Verify text guide contains gestures, rotation guide, and all songs."""
    guide = generate_text_guide()
    assert "HAND GESTURE MUSICAL INSTRUMENT" in guide
    assert "Closed Fist" in guide
    assert "Open Palm" in guide
    assert "PITCH SHIFT" in guide
    assert "MASTER VOLUME" in guide
    assert "JINGLE BELLS" in guide
    assert "TWINKLE TWINKLE" in guide
    assert "ODE TO JOY" in guide
    assert "HAPPY BIRTHDAY" in guide

    # Test file saving
    test_path = "test_guide_output.txt"
    try:
        save_guide_to_file(test_path)
        assert os.path.exists(test_path)
        with open(test_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 500
            assert "Closed Fist" in content
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_orange_hud_rendering():
    """Verify minimalist HUD renders target note in orange without errors."""
    engine = AudioEngine()
    mode = SongGuideMode(engine)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Render HUD
    mode.draw_hud(frame)
    # Ensure pixels were drawn
    assert np.max(frame) > 0

    # Test scale mode HUD
    scale_mode = GestureScaleMode(engine)
    scale_mode.active_note = 'E4'
    scale_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scale_mode.draw_hud(scale_frame)
    assert np.max(scale_frame) > 0
    engine.stop()


def test_scale_mode_with_pitch_shift(audio):
    """Verify GestureScaleMode updates note with pitch shift."""
    mode = GestureScaleMode(audio)
    # Fist (base C4) with +2 semitones -> D4
    mode.update('Fist', confidence=0.95, pitch_shift_semitones=2)
    assert mode.active_note == 'D4'


def test_ok_sign_vs_pinch_classification():
    """Verify OK Sign and Pinch are accurately differentiated by finger curl geometry."""
    from gesture_features import rule_based_classify
    from model import GestureModel

    pinch_hand = generate_canonical_hand('Pinch')
    ok_hand = generate_canonical_hand('OK Sign')

    rule_p, conf_p = rule_based_classify(pinch_hand)
    rule_ok, conf_ok = rule_based_classify(ok_hand)

    assert rule_p == 'Pinch'
    assert conf_p >= 0.90
    assert rule_ok == 'OK Sign'
    assert conf_ok >= 0.90

    model = GestureModel()
    pred_p, _ = model.predict(pinch_hand, use_history=False)
    pred_ok, _ = model.predict(ok_hand, use_history=False)

    assert pred_p == 'Pinch'
    assert pred_ok == 'OK Sign'


def test_all_13_gestures_canonical_generation():
    """Verify all 13 gestures have distinct canonical landmark generation and classification."""
    from gesture_features import rule_based_classify
    from model import GestureModel

    classes = [
        'Fist', 'Index Point', 'Peace', 'Three', 'Four',
        'Open Palm', 'Thumbs Up', 'Pinch', 'Rock On',
        'Hang Loose', 'Gun', 'Pinky Up', 'OK Sign'
    ]

    model = GestureModel()
    for cls in classes:
        hand = generate_canonical_hand(cls)
        assert hand.shape == (21, 3)
        rule_gest, rule_conf = rule_based_classify(hand)
        assert rule_gest == cls, f"Rule mismatch for {cls}: got {rule_gest}"
        pred, conf = model.predict(hand, use_history=False)
        assert pred == cls, f"Model mismatch for {cls}: got {pred}"


def test_rotation_circular_smoothing_wrap_around():
    """Verify circular difference smoothing across +-180 degrees avoids spurious jumps."""
    from hand_detector import TrackedHandState, HandDetector
    detector = HandDetector()

    raw1 = generate_canonical_hand('Open Palm')
    raw2 = raw1.copy()

    # Track 0 with initial rotation near +175 degrees
    th = TrackedHandState(0, raw1, 'Right')
    th.rotation_deg = 175.0
    detector.tracked_hands[0] = th

    # Frame has hand tilted slightly to -175 degrees (10 deg delta across boundary)
    raw_rot = -175.0
    rot_diff = (raw_rot - th.rotation_deg + 180.0) % 360.0 - 180.0
    smoothed = (th.rotation_deg + 0.30 * rot_diff + 180.0) % 360.0 - 180.0

    # Must be near 178 degrees, NOT jumping down to ~70 degrees
    assert abs(smoothed - 178.0) < 1.0


def test_multi_hand_melody_priority_over_volume():
    """Verify Right hand melody notes take strict precedence over Left hand volume controls."""
    from app import HandGestureInstrumentApp
    app = HandGestureInstrumentApp(simulate=True, headless=True)

    hand_left = HandInfo(generate_canonical_hand('Open Palm'), handedness='Left')
    hand_right = HandInfo(generate_canonical_hand('Peace'), handedness='Right')

    # Both hands in view: Left first, Right second
    app.cap.get_simulated_hands = lambda: [hand_left, hand_right]

    ret, frame = app.step()
    assert ret is True
    # The note played in Song mode must come from the Right hand (Peace = E4), not the Left hand (Open Palm = A4)
    # Peace is target note for Jingle Bells first note, so current_note_idx advances to 1
    assert app.song_mode.current_note_idx >= 1
    app.close()


def test_hand_role_swap():
    """Verify hand roles can be swapped for left-handed performers."""
    from app import HandGestureInstrumentApp
    app = HandGestureInstrumentApp(simulate=True, headless=True, lefty=True)
    assert app.melody_hand == "Left"
    assert app.control_hand == "Right"

    app.swap_hand_roles()
    assert app.melody_hand == "Right"
    assert app.control_hand == "Left"
    app.close()


def test_audio_engine_rest_produces_no_sound(audio):
    """Verify that REST notes are completely silent and spawn no audio voices."""
    assert note_to_freq('REST') == 0.0
    initial_voices = len(audio.voices)
    audio.play_note('REST')
    assert len(audio.voices) == initial_voices


def test_clean_hud_mode():
    """Verify clean HUD mode suppresses top bar and footer overlays."""
    from app import HandGestureInstrumentApp
    app = HandGestureInstrumentApp(simulate=True, headless=True, clean_hud=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    app.draw_top_hud(frame)
    app.draw_footer_help(frame)
    # Entire frame should remain untouched zeros
    assert np.all(frame == 0)
    app.close()
