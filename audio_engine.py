"""
audio_engine.py
Real-time polyphonic audio synthesis engine for Hand Gesture Instrument.
Supports Piano, Bell (Jingle Bells), Violin, Synth, and Continuous Theremin voices.
"""

import math
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


# Standard note name to MIDI number mapping
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}


SEMITONE_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_to_midi(note_name: str) -> int:
    """Convert note string (e.g., 'C4', 'A#4') to MIDI note number (C4 = 60, A4 = 69)."""
    note_name = note_name.strip()
    if not note_name or note_name == 'REST':
        return 69  # A4 default
    if len(note_name) >= 2 and note_name[1] in ('#', 'b'):
        pitch_class = note_name[:2]
        octave_str = note_name[2:]
    else:
        pitch_class = note_name[:1]
        octave_str = note_name[1:]
    try:
        octave = int(octave_str) if octave_str else 4
    except ValueError:
        octave = 4
    semitone = NOTE_TO_SEMITONE.get(pitch_class, 9)
    return 12 * (octave + 1) + semitone


def midi_to_note(midi_num: int) -> str:
    """Convert MIDI note number (0..127) back to note name string."""
    midi_num = max(0, min(int(midi_num), 127))
    semitone = midi_num % 12
    octave = (midi_num // 12) - 1
    pitch_class = SEMITONE_TO_NOTE[semitone]
    return f"{pitch_class}{octave}"


def shift_note_semitones(note_name: str, semitones: int) -> str:
    """Transpose a note name by a number of semitones (e.g. C4 + 2 -> D4)."""
    if not note_name or note_name == 'REST' or note_name.startswith('CHORD'):
        return note_name
    shifted_midi = note_to_midi(note_name) + semitones
    return midi_to_note(shifted_midi)


def note_to_freq(note_name: str) -> float:
    """
    Convert note name (e.g., 'C4', 'E4', 'G#4', 'A4', 'C5') to frequency in Hz.
    Standard A4 = 440 Hz (MIDI note 69).
    """
    note_name = note_name.strip()
    if not note_name:
        return 440.0
    if note_name == 'REST':
        return 0.0
    
    # Extract note letter, accidental, and octave
    if len(note_name) >= 2 and note_name[1] in ('#', 'b'):
        pitch_class = note_name[:2]
        octave_str = note_name[2:]
    else:
        pitch_class = note_name[:1]
        octave_str = note_name[1:]
    
    try:
        octave = int(octave_str) if octave_str else 4
    except ValueError:
        octave = 4

    semitone = NOTE_TO_SEMITONE.get(pitch_class, 9)  # default A
    midi_num = 12 * (octave + 1) + semitone
    freq = 440.0 * (2.0 ** ((midi_num - 69) / 12.0))
    return float(freq)


class Voice:
    """Represents a single active sounding voice."""

    def __init__(self, freq: float, instrument: str = 'piano', velocity: float = 0.8, duration: float = 1.0):
        self.freq = max(20.0, min(freq, 8000.0))
        self.instrument = instrument.lower()
        self.velocity = max(0.0, min(velocity, 1.0))
        self.duration = duration
        self.start_time = time.time()
        self.age = 0.0
        self.phase = 0.0
        self.active = True

    def render(self, sample_rate: int, frames: int) -> np.ndarray:
        """Render `frames` audio samples for this voice."""
        if not self.active:
            return np.zeros(frames, dtype=np.float32)

        dt = 1.0 / sample_rate
        t = self.age + np.arange(frames, dtype=np.float32) * dt
        self.age += frames * dt

        if self.age >= self.duration:
            self.active = False

        omega = 2.0 * np.pi * self.freq
        wave = np.zeros(frames, dtype=np.float32)

        if self.instrument == 'bell':
            # Shimmering metallic bell overtones (characteristic of jingle bells)
            envelope = np.exp(-t * 2.8) * self.velocity
            # Modal bell frequencies: f0, 2.76*f0, 5.4*f0, 8.9*f0
            wave = (
                0.55 * np.sin(omega * t) +
                0.28 * np.sin(omega * 2.76 * t) * np.exp(-t * 3.2) +
                0.12 * np.sin(omega * 5.40 * t) * np.exp(-t * 4.5) +
                0.05 * np.sin(omega * 8.93 * t) * np.exp(-t * 6.0)
            ) * envelope

        elif self.instrument == 'piano':
            # Acoustic piano with fundamental and subtle rich harmonics + gentle detune
            envelope = np.exp(-t * 2.0) * (1.0 - np.exp(-t * 80.0)) * self.velocity
            wave = (
                0.60 * np.sin(omega * t) +
                0.25 * np.sin(omega * 2.001 * t) * np.exp(-t * 1.5) +
                0.10 * np.sin(omega * 3.003 * t) * np.exp(-t * 2.5) +
                0.05 * np.sin(omega * 4.005 * t) * np.exp(-t * 3.5)
            ) * envelope

        elif self.instrument in ('violin', 'flute'):
            # Expressive classical violin bowed string synthesis:
            # - Gentle bowing attack onset (50ms) with rich sustained body
            # - Sawtooth harmonic distribution with body resonance (1st through 7th harmonics)
            # - Natural expressive 5.5Hz vibrato ramping in after attack
            attack = np.minimum(t / 0.06, 1.0)
            decay = np.exp(-np.maximum(0.0, t - 0.45) * 1.4)
            envelope = attack * decay * self.velocity
            # Expressive vibrato depth builds smoothly as note sustains
            vib_depth = 0.012 * np.minimum(t / 0.12, 1.0)
            vibrato = 1.0 + vib_depth * np.sin(2.0 * np.pi * 5.5 * t)
            # Harmonically rich bowed string timbre with wood resonance
            wave = (
                0.46 * np.sin(omega * vibrato * t) +
                0.26 * np.sin(omega * 2.0 * vibrato * t) +
                0.16 * np.sin(omega * 3.0 * vibrato * t) +
                0.10 * np.sin(omega * 4.0 * vibrato * t) +
                0.06 * np.sin(omega * 5.0 * vibrato * t) +
                0.04 * np.sin(omega * 6.0 * vibrato * t) +
                0.02 * np.sin(omega * 7.0 * vibrato * t)
            ) * envelope

        elif self.instrument == 'synth':
            # Analog poly synth (warm pulse/saw waveform)
            envelope = np.exp(-t * 1.6) * self.velocity
            # Approximated bandlimited sawtooth
            wave = (
                0.50 * np.sin(omega * t) +
                0.25 * np.sin(omega * 2.0 * t) +
                0.15 * np.sin(omega * 3.0 * t) +
                0.10 * np.sin(omega * 4.0 * t)
            ) * envelope

        else:
            # Clean sine wave default
            envelope = np.exp(-t * 2.0) * self.velocity
            wave = np.sin(omega * t) * envelope

        return wave.astype(np.float32)


class ThereminController:
    """Controls continuous pitch and volume for Theremin mode."""

    def __init__(self):
        self.target_freq = 440.0
        self.current_freq = 440.0
        self.target_volume = 0.0
        self.current_volume = 0.0
        self.phase = 0.0
        self.vibrato_phase = 0.0
        self.vibrato_depth = 0.0  # Semitones or Hz
        self.enabled = False

    def update(self, freq: float, volume: float, vibrato: float = 0.0, enabled: bool = True):
        self.target_freq = max(65.0, min(freq, 2093.0))  # C2 to C7
        self.target_volume = max(0.0, min(volume, 1.0))
        self.vibrato_depth = max(0.0, min(vibrato, 1.0))
        self.enabled = enabled

    def render(self, sample_rate: int, frames: int) -> np.ndarray:
        if not self.enabled and self.current_volume < 1e-4:
            return np.zeros(frames, dtype=np.float32)

        dt = 1.0 / sample_rate
        # Smooth interpolation (portamento) to prevent clicks
        alpha_freq = 1.0 - math.exp(-dt * 40.0)
        alpha_vol = 1.0 - math.exp(-dt * 30.0)

        out = np.zeros(frames, dtype=np.float32)
        vibrato_freq = 5.5  # 5.5 Hz classic theremin vibrato

        for i in range(frames):
            self.current_freq += (self.target_freq - self.current_freq) * alpha_freq
            effective_target_vol = self.target_volume if self.enabled else 0.0
            self.current_volume += (effective_target_vol - self.current_volume) * alpha_vol

            vibrato_mod = math.sin(self.vibrato_phase) * (self.vibrato_depth * 18.0)
            instant_freq = max(20.0, self.current_freq + vibrato_mod)

            self.phase += 2.0 * math.pi * instant_freq * dt
            if self.phase > 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

            self.vibrato_phase += 2.0 * math.pi * vibrato_freq * dt
            if self.vibrato_phase > 2.0 * math.pi:
                self.vibrato_phase -= 2.0 * math.pi

            # Theremin waveform: fundamental + subtle warm odd harmonics
            sample = (
                math.sin(self.phase) * 0.85 +
                math.sin(self.phase * 3.0) * 0.10 +
                math.sin(self.phase * 5.0) * 0.05
            )
            out[i] = sample * self.current_volume * 0.7

        return out


class AudioEngine:
    """
    Main audio synthesis engine.
    Runs continuous audio stream and renders polyphonic instrument voices and Theremin.
    """

    INSTRUMENTS = ['piano', 'bell', 'violin', 'synth']

    def __init__(self, sample_rate: int = 44100, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.voices: List[Voice] = []
        self.lock = threading.Lock()
        self.instrument = 'bell'  # Default to bell for Jingle Bells!
        self.theremin = ThereminController()
        self.is_running = False
        self.stream: Optional[sd.OutputStream] = None
        self.waveform_preview = np.zeros(128, dtype=np.float32)
        self.master_volume = 0.85
        self.start()

    def start(self):
        """Start the audio stream."""
        if not HAS_SOUNDDEVICE:
            print("[AudioEngine] sounddevice not available. Running in headless silent mode.")
            self.is_running = True
            return

        if self.is_running and self.stream is not None:
            return

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.block_size,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
        except Exception as e:
            print(f"[AudioEngine] Warning: Could not open audio output stream ({e}). Continuing silently.")
            self.is_running = False

    def stop(self):
        """Stop the audio stream and release resources."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def set_instrument(self, name: str):
        """Switch current instrument timbre."""
        name = name.lower()
        if name == 'flute':
            name = 'violin'
        if name in self.INSTRUMENTS:
            self.instrument = name
            return True
        return False

    def cycle_instrument(self) -> str:
        """Cycle to the next instrument timbre and return its name."""
        idx = (self.INSTRUMENTS.index(self.instrument) + 1) % len(self.INSTRUMENTS)
        self.instrument = self.INSTRUMENTS[idx]
        return self.instrument

    def set_master_volume(self, volume: float):
        """Set master audio output volume (0.0 to 1.0)."""
        self.master_volume = float(np.clip(volume, 0.0, 1.0))

    def play_note(
        self,
        note: str,
        velocity: float = 0.85,
        duration: float = 1.0,
        instrument: Optional[str] = None,
        pitch_shift_semitones: int = 0
    ):
        """Trigger a note by name (e.g., 'E4', 'C4', 'G4'), optionally transposed by semitones."""
        if not note or note == 'REST':
            return
        effective_note = shift_note_semitones(note, pitch_shift_semitones) if pitch_shift_semitones != 0 else note
        if not effective_note or effective_note == 'REST':
            return
        freq = note_to_freq(effective_note)
        if freq <= 0.0:
            return
        self.play_freq(freq, velocity=velocity, duration=duration, instrument=instrument)

    def play_chord(
        self,
        notes: List[str],
        velocity: float = 0.80,
        duration: float = 1.6,
        instrument: Optional[str] = None
    ):
        """Trigger a polyphonic chord consisting of multiple note strings simultaneously."""
        for n in notes:
            self.play_note(n, velocity=velocity, duration=duration, instrument=instrument)

    def play_freq(self, freq: float, velocity: float = 0.85, duration: float = 1.0, instrument: Optional[str] = None):
        """Trigger a note by frequency in Hz."""
        inst = instrument if instrument is not None else self.instrument
        voice = Voice(freq=freq, instrument=inst, velocity=velocity, duration=duration)
        with self.lock:
            # Polyphony limit to prevent CPU spikes
            while len(self.voices) >= 16:
                self.voices.pop(0)
            self.voices.append(voice)

    def update_theremin(self, freq: float, volume: float, vibrato: float = 0.0, enabled: bool = True):
        """Update continuous Theremin controller parameters."""
        self.theremin.update(freq, volume, vibrato, enabled)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Real-time sounddevice output callback."""
        mix = np.zeros(frames, dtype=np.float32)

        # 1. Render Theremin
        theremin_out = self.theremin.render(self.sample_rate, frames)
        mix += theremin_out

        # 2. Render active voices
        with self.lock:
            active_voices = []
            for voice in self.voices:
                if voice.active:
                    voice_samples = voice.render(self.sample_rate, frames)
                    mix += voice_samples
                    if voice.active:
                        active_voices.append(voice)
            self.voices = active_voices

        # Apply master volume and smooth soft limiter (tanh) to avoid harsh clipping
        mix = np.tanh(mix * self.master_volume)

        # Store preview snippet for GUI oscilloscope
        preview_step = max(1, frames // 128)
        self.waveform_preview = mix[::preview_step][:128].copy()

        # Write stereo channels
        outdata[:, 0] = mix
        outdata[:, 1] = mix

    def get_waveform_preview(self) -> np.ndarray:
        """Get 128-sample waveform for oscilloscope visualization."""
        return self.waveform_preview.copy()
