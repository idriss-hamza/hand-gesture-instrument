"""
Edge case and stress tests for Hand Gesture Musical Instrument.
Tests degenerate landmarks, audio voice overflow, frequency boundaries,
rapid gesture changes, and song completion boundaries.
"""

import numpy as np
import pytest
from audio_engine import AudioEngine, Voice, note_to_freq
from gesture_features import extract_feature_vector, rule_based_classify, compute_scale
from model import GestureModel
from song_library import get_jingle_bells
from instrument_modes import SongGuideMode, GestureScaleMode, AirPianoMode, ThereminMode
from hand_detector import HandInfo


def test_degenerate_zero_landmarks():
    zero_lms = np.zeros((21, 3), dtype=np.float32)
    scale = compute_scale(zero_lms)
    assert scale == 1e-4  # Guarded against division by zero

    feats = extract_feature_vector(zero_lms)
    assert len(feats) == 85
    assert not np.isnan(feats).any()

    # Rule based classification should not crash
    pred, conf = rule_based_classify(zero_lms)
    assert isinstance(pred, str)


def test_extreme_audio_frequencies():
    engine = AudioEngine()
    # Boundary frequencies
    engine.play_freq(20.0, duration=0.05)
    engine.play_freq(8000.0, duration=0.05)
    engine.play_freq(-50.0, duration=0.05)   # Negative frequency should be clamped to >= 20Hz
    engine.play_freq(100000.0, duration=0.05) # Above Nyquist clamped to 8000Hz

    wave = engine.get_waveform_preview()
    assert not np.isnan(wave).any()
    engine.stop()


def test_polyphony_overflow_handling():
    engine = AudioEngine()
    # Rapidly spawn 50 voices in quick succession
    for i in range(50):
        engine.play_note('C4', duration=1.0)

    # Active voice list should be capped to 16
    assert len(engine.voices) <= 16
    engine.stop()


def test_song_guide_completion_boundary():
    engine = AudioEngine()
    song = get_jingle_bells()
    mode = SongGuideMode(engine, song=song)

    # Fast forward through all notes to completion
    for _ in range(len(song.notes) + 5):
        mode.trigger_note_success()

    assert mode.is_completed is True
    assert mode.get_current_note() is None
    # Calling update after completion should not crash
    mode.update('Peace', 0.95)
    assert mode.is_completed is True

    # Reset
    mode.reset()
    assert mode.is_completed is False
    assert mode.current_note_idx == 0
    engine.stop()


def test_air_piano_out_of_bounds_finger():
    engine = AudioEngine()
    mode = AirPianoMode(engine)

    # Fingertip off screen / out of keys area (e.g. near top of screen y=20)
    raw = np.zeros((21, 3), dtype=np.float32)
    raw[8] = [0.5, 0.05, 0.0]
    hand = HandInfo(raw)

    mode.update([hand], 640, 480)
    assert mode.pressed_key is None
    engine.stop()


def test_theremin_empty_hands():
    engine = AudioEngine()
    mode = ThereminMode(engine)

    mode.update([], 640, 480)
    assert mode.active is False
    assert engine.theremin.enabled is False
    engine.stop()


def test_unknown_note_handling():
    # Empty string or unknown notes should fallback gracefully without crashing
    f1 = note_to_freq('')
    assert f1 == 440.0
    f2 = note_to_freq('Z9')
    assert f2 > 0.0


def test_fist_never_misclassified_as_pinch():
    """Verify tight fists with close thumb/index proximity are never misclassified as Pinch."""
    from trainer import generate_canonical_hand, augment_landmarks
    fist = generate_canonical_hand('Fist')
    rng = np.random.RandomState(42)
    model = GestureModel()

    for _ in range(100):
        aug_fist = augment_landmarks(fist, rng)
        pred, conf = model.predict(aug_fist, use_history=False)
        assert pred != 'Pinch', f"Fist was falsely misclassified as Pinch (conf={conf:.2f})"
        assert pred in ('Fist', 'Thumbs Up')
