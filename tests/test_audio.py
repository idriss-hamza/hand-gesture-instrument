"""
Unit tests for audio synthesis engine.
"""

import numpy as np
import pytest
from audio_engine import AudioEngine, Voice, ThereminController, note_to_freq, NOTE_TO_SEMITONE


def test_note_to_freq_standard():
    assert abs(note_to_freq('A4') - 440.0) < 0.01
    assert abs(note_to_freq('C4') - 261.63) < 0.1
    assert abs(note_to_freq('C5') - 523.25) < 0.2
    assert abs(note_to_freq('E4') - 329.63) < 0.1
    assert abs(note_to_freq('G4') - 392.00) < 0.1


def test_note_to_freq_accidentals():
    freq_cs4 = note_to_freq('C#4')
    freq_db4 = note_to_freq('Db4')
    assert abs(freq_cs4 - freq_db4) < 1e-4
    assert freq_cs4 > note_to_freq('C4')
    assert freq_cs4 < note_to_freq('D4')


def test_voice_render_all_instruments():
    instruments = ['bell', 'piano', 'violin', 'synth', 'flute']
    sr = 44100
    frames = 512

    for inst in instruments:
        voice = Voice(freq=440.0, instrument=inst, velocity=0.8, duration=0.5)
        samples = voice.render(sr, frames)

        assert isinstance(samples, np.ndarray)
        assert len(samples) == frames
        assert not np.isnan(samples).any(), f"NaN in {inst} samples"
        assert not np.isinf(samples).any(), f"Inf in {inst} samples"
        assert np.max(np.abs(samples)) <= 1.5, f"Excessive amplitude in {inst}"


def test_voice_expiry():
    sr = 44100
    voice = Voice(freq=440.0, duration=0.01)
    samples = voice.render(sr, 1000)
    assert not voice.active


def test_theremin_controller():
    tc = ThereminController()
    tc.update(freq=440.0, volume=0.7, vibrato=0.5, enabled=True)

    samples = tc.render(44100, 512)
    assert len(samples) == 512
    assert not np.isnan(samples).any()
    assert np.max(np.abs(samples)) > 0.0

    # Test disabled
    tc.update(freq=440.0, volume=0.0, enabled=False)
    # Render several blocks to allow decay
    for _ in range(20):
        samples = tc.render(44100, 512)
    assert np.max(np.abs(samples)) < 0.05


def test_audio_engine_lifecycle():
    engine = AudioEngine()
    engine.play_note('E4', duration=0.1)
    engine.play_freq(440.0, duration=0.1)
    assert engine.set_instrument('piano')
    assert engine.instrument == 'piano'
    assert engine.cycle_instrument() in AudioEngine.INSTRUMENTS

    wave = engine.get_waveform_preview()
    assert len(wave) == 128
    engine.stop()
