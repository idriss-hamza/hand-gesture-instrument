"""
Unit tests for instrument playing modes.
"""

import numpy as np
import pytest
from audio_engine import AudioEngine
from instrument_modes import SongGuideMode, GestureScaleMode, AirPianoMode, ThereminMode
from song_library import get_jingle_bells
from hand_detector import HandInfo


@pytest.fixture
def audio():
    engine = AudioEngine()
    yield engine
    engine.stop()


def test_song_guide_mode_interactive_success(audio):
    mode = SongGuideMode(audio, song=get_jingle_bells())
    assert mode.current_note_idx == 0
    first_note = mode.get_current_note()
    assert first_note.note == 'E4'
    assert first_note.gesture == 'Peace'

    # Match gesture
    mode.update('Peace', confidence=0.95)
    assert mode.current_note_idx == 1
    assert mode.score == 100
    assert mode.streak == 1


def test_song_guide_mode_autoplay(audio):
    mode = SongGuideMode(audio, song=get_jingle_bells())
    mode.autoplay = True
    mode.last_autoplay_time = 0.0  # Force immediate trigger

    mode.update('None', confidence=0.0)
    assert mode.current_note_idx == 1


def test_gesture_scale_mode(audio):
    mode = GestureScaleMode(audio)
    # Fist -> C4
    mode.update('Fist', confidence=0.9)
    assert mode.active_note == 'C4'

    # Peace -> E4
    mode.update('Peace', confidence=0.9)
    assert mode.active_note == 'E4'


def test_air_piano_mode(audio):
    mode = AirPianoMode(audio)
    # Create mock hand with index fingertip over key 0 (C4)
    raw = np.zeros((21, 3), dtype=np.float32)
    # Put index tip (index 8) at (x=40, y=420)
    raw[8] = [40.0 / 640.0, 420.0 / 480.0, 0.0]
    hand = HandInfo(raw)

    mode.update([hand], 640, 480)
    assert mode.pressed_key == 'C4'


def test_theremin_mode(audio):
    mode = ThereminMode(audio)
    raw = np.zeros((21, 3), dtype=np.float32)
    # Index tip at center
    raw[8] = [0.5, 0.5, 0.0]
    hand = HandInfo(raw, handedness='Right')

    mode.update([hand], 640, 480)
    assert mode.active is True
    assert mode.freq > 130.0 and mode.freq < 1050.0
    assert mode.volume > 0.0


def test_song_guide_mode_holding_gesture_does_not_auto_advance(audio):
    """Verify that passively holding a gesture does NOT machine-gun fire repeated identical notes."""
    mode = SongGuideMode(audio, song=get_jingle_bells())
    # Note 0: E4 (Peace)
    mode.update('Peace', confidence=0.95)
    assert mode.current_note_idx == 1

    # Next note is ALSO E4 (Peace). Holding Peace should NOT advance note 1!
    for _ in range(25):
        mode.update('Peace', confidence=0.95)
    assert mode.current_note_idx == 1  # Still on note 1!

    # Release gesture (user opens or relaxes hand)
    mode.update('None', confidence=0.0)
    assert mode.waiting_for_release is False

    # Re-strike gesture for note 1
    mode.update('Peace', confidence=0.95)
    assert mode.current_note_idx == 2


def test_song_guide_mode_wrong_gesture_feedback(audio):
    mode = SongGuideMode(audio, song=get_jingle_bells())
    assert mode.get_current_note().gesture == 'Peace'

    # Make incorrect gesture
    mode.update('Fist', confidence=0.90)
    assert mode.streak == 0
    assert mode.total_attempts == 1
    assert "Detected: Fist" in mode.feedback_text
    assert mode.current_note_idx == 0


def test_air_piano_polyphony(audio):
    mode = AirPianoMode(audio)
    # Hand 1 index finger on key 0 (C4)
    raw1 = np.zeros((21, 3), dtype=np.float32)
    raw1[8] = [40.0 / 640.0, 420.0 / 480.0, 0.0]
    hand1 = HandInfo(raw1, handedness='Left')

    # Hand 2 index finger on key 4 (G4, x=360)
    raw2 = np.zeros((21, 3), dtype=np.float32)
    raw2[8] = [360.0 / 640.0, 420.0 / 480.0, 0.0]
    hand2 = HandInfo(raw2, handedness='Right')

    mode.update([hand1, hand2], 640, 480)
    assert 'C4' in mode.pressed_keys
    assert 'G4' in mode.pressed_keys
    assert len(mode.pressed_keys) == 2


def test_theremin_pitch_direction_and_vibrato_isolation(audio):
    mode = ThereminMode(audio)
    # Hand on left (x=0.2)
    raw_left = np.zeros((21, 3), dtype=np.float32)
    raw_left[8] = [0.2, 0.5, 0.0]
    hand_left = HandInfo(raw_left, handedness='Right')
    mode.update([hand_left], 640, 480)
    freq_left = mode.freq
    # Single hand must NOT trigger vibrato
    assert mode.vibrato == 0.0

    # Hand on right (x=0.8)
    raw_right = np.zeros((21, 3), dtype=np.float32)
    raw_right[8] = [0.8, 0.5, 0.0]
    hand_right = HandInfo(raw_right, handedness='Right')
    mode.update([hand_right], 640, 480)
    freq_right = mode.freq

    # Right side of screen must have higher pitch than left side
    assert freq_right > freq_left

    # Adding a second hand (left hand) enables vibrato
    raw_vibrato = np.zeros((21, 3), dtype=np.float32)
    raw_vibrato[8] = [0.7, 0.5, 0.0]
    hand_vibrato = HandInfo(raw_vibrato, handedness='Left')
    mode.update([hand_right, hand_vibrato], 640, 480)
    assert mode.vibrato > 0.0
