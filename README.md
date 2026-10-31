# 🎵 Hand Gesture Musical Instrument

A real-time, camera-based musical instrument that translates hand gestures, finger kinematics, and wrist orientation into polyphonic musical notes and expressive sound synthesis.

Built using **OpenCV**, **MediaPipe HandLandmarker**, an anatomical **Random Forest Gesture Classifier**, and a low-latency **Polyphonic Audio Synthesizer**.

---

## 🌟 Key Features

- **Real-Time Hand Tracking & Kinematics**:
  - Sub-millimeter tracking of 21 3D hand landmarks per hand with adaptive velocity-sensitive temporal smoothing.
  - Cybernetic vector HUD rendering: geometric palm polygons, center reticles, fingertip targeting rings, and dynamic tap ripples.
  - Automatic ordinary camera framing (unzoomed, unstretched) with optional top crop adjustment (`[` / `]`).

- **13-Class ML Gesture Classifier**:
  - Custom model trained on 9,100 3D augmented hand poses (`gesture_model.pkl`).
  - Scale, translation, and 3D rotation invariant feature vector using normalized inter-landmark distances, bone angles, and palm normal projections.
  - 100% test accuracy across all 13 canonical gestures with temporal debounce and geometric heuristics.

- **Dual-Hand Spatial Controls (Fist Clutch Rotary System)**:
  - **Left Hand (Volume)**: Form a **Closed Fist** and rotate your wrist to dial master volume (0% mute at -60° to 100% loudness at +60°). Open your hand to **lock** the volume in place—you can then return your hand to center without altering the volume.
  - **Right Hand (Pitch Shift)**: Form a **Closed Fist** and rotate your wrist to transpose pitch (-12 to +12 semitones). Open your hand to **lock** the pitch shift and play notes freely.
  - **Left-Handed Mode**: Instant role swap via hotkey (`H`) or CLI flag (`--lefty`).

- **4 Performance Modes**:
  1. **Free Instrument Mode (`[2]` or `[F]`)**: Direct, unguided expressive performance. Plays whatever gesture is detected with pitch shift, chord accompaniment, and persistent note display.
  2. **Song Guide Mode (`[1]`)**: Interactive play-along tutorial with target note badge, scoring, and release/tap validation.
  3. **Air Piano Mode (`[3]`)**: Virtual on-screen keyboard. Features strict **single-finger pointer precision** (index fingertip `pts[8]`), downward laser targeting beam, floating note aim badge, hover highlighting, and key depression animations.
  4. **Air Theremin Mode (`[4]`)**: Continuous spatial synthesis. Horizontal hand position controls pitch frequency, vertical position controls volume amplitude, with natural pitch vibrato and animated audio oscilloscope.

- **Polyphonic Audio Engine & Instrument Voices**:
  - **🔔 Bell**: Modal metallic chime synthesis with shimmering inharmonic decays (ideal for *Jingle Bells*).
  - **🎹 Piano**: Dynamic multi-harmonic acoustic string model with natural decay.
  - **🎻 Violin**: Classical bowed string synthesis featuring a 50ms Helmholtz bow-strike friction profile, sawtooth harmonics (1st–7th), wooden body resonance, and natural 5.5 Hz vibrato.
  - **🎛️ Synth**: Rich analog dual-sawtooth lead with low-pass filtering.
  - Low-latency multi-voice polyphony with soft saturation limiting (`tanh`).

- **Built-in Song Repertoire**:
  - 🔔 *Jingle Bells*
  - ⭐ *Twinkle Twinkle Little Star*
  - 🎶 *Beethoven's Ode to Joy*
  - 🎂 *Happy Birthday*
  - 🐑 *Mary Had a Little Lamb*
  - 🎼 *Für Elise*

- **Synthetic Simulator & Fallback**:
  - Automatically connects to physical camera (`/dev/video0`).
  - If no webcam is attached or running in headless CI environments, automatically activates the built-in synthetic hand simulator stream for testing and automated demonstrations.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A working webcam (optional; simulator activates automatically if unavailable)
- Audio output device

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/idriss-hamza/hand-gesture-instrument.git
   cd hand-gesture-instrument
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run

### 1. Launch in Free Instrument Mode
```bash
python3 run_instrument.py --free
```

### 2. Launch Guided Song Tutorial
```bash
python3 run_instrument.py
```

### 3. Launch Air Piano with Violin
```bash
python3 run_instrument.py --mode piano --instrument violin
```

### 4. Run Demonstration with Synthetic Simulator
```bash
python3 run_instrument.py --simulate
```

### Command-Line Options

```
usage: run_instrument.py [-h] [--camera CAMERA] [--simulate] [--free]
                         [--mode {guided,free,piano,theremin}]
                         [--song SONG] [--instrument {bell,piano,violin,synth}]
                         [--lefty] [--clean-hud] [--crop-top CROP_TOP]
                         [--headless] [--frames FRAMES]
```

---

## 🕹️ Keyboard Controls

| Key | Action |
|:---:|:---|
| `1` | Switch to **Song Guide Mode** (Interactive tutorial) |
| `2` / `F` | Switch to **Free Instrument Mode** (Unguided gesture performance) |
| `3` | Switch to **Air Piano Mode** (Single-finger laser pointer keyboard) |
| `4` | Switch to **Air Theremin Mode** (Continuous pitch & volume) |
| `I` | Cycle instrument timbre (**Bell** → **Piano** → **Violin** → **Synth**) |
| `S` | Cycle to next song in the catalog |
| `L` | Print songs library and gesture notes in terminal |
| `P` | Toggle **Autoplay Demo** on/off |
| `Space` | Tap to re-strike currently aimed/held note |
| `R` | Reset current song & unlock pitch/volume to default |
| `H` | Swap Melody and Control hand roles (Right/Left) |
| `C` | Toggle Clean Camera HUD mode |
| `[` / `-` | Decrease camera top crop (-5%) |
| `]` / `+` | Increase camera top crop (+5%) |
| `Q` / `ESC` | Quit application |

---

## 🖐️ Gesture-to-Note Mapping

| Hand Gesture | Note | Symbol On Screen | Musical Description |
|:---|:---:|:---|:---|
| **Closed Fist** | `C4` | Closed Fist | 0 fingers, tight curled fist *(Also engages rotary dial)* |
| **Index Point** | `D4` | 1 Finger (Index) | 1 finger, index pointing upward |
| **Peace / Victory** | `E4` | 2 Fingers (Peace) | 2 fingers (Index + Middle) in "V" |
| **Three Fingers** | `F4` | 3 Fingers | 3 fingers (Index + Middle + Ring) |
| **Four Fingers** | `G4` | 4 Fingers | 4 fingers (Thumb tucked in) |
| **Open Palm** | `A4` | Open Palm (5 Fingers) | All 5 fingers extended wide |
| **Thumbs Up** | `B4` | Thumbs Up | Thumb extended up, fingers curled |
| **Pinch / Tip Touch**| `C5` | Pinch / Tip Touch | Thumb tip touching index tip (curled) |
| **Rock On** | `D5` | Rock On (Index + Pinky)| Index + Pinky extended |
| **Hang Loose** | `E5` | Hang Loose / Shaka | Thumb + Pinky extended outward |
| **Gun** | `F5` | Gun / L-Shape | Thumb + Index at 90° angle |
| **Pinky Up** | `G5` | Pinky Only | Only pinky finger extended |
| **OK Sign** | `A5` | OK Sign (Circle) | Thumb + Index circle, 3 fingers extended |

---

## 🧪 Automated Testing

The project includes an automated test suite covering audio synthesis, computer vision landmark normalization, gesture classification robustness, UI layout constraints, and mode state machines:

```bash
pytest -v
```

All 84 tests execute in under 25 seconds without requiring a physical camera or audio hardware.

---

## 📁 Project Structure

```
├── app.py                      # Main application orchestrator & video loop
├── audio_engine.py             # Polyphonic synthesizer (Bell, Piano, Violin, Synth, Theremin)
├── cybernetic_hud.py           # HUD engine (pitch ladder, chord catalog, oscilloscope)
├── gesture_features.py         # 21-landmark geometric extraction & normalization
├── hand_detector.py            # MediaPipe HandLandmarker wrapper & spatial tracking
├── instrument_modes.py         # Modes: SongGuide, FreeInstrument, AirPiano, Theremin
├── model.py                    # Trained RandomForest classifier & temporal debounce
├── run_instrument.py           # CLI entry point
├── simulator.py                # Synthetic hand landmark animator & mock camera
├── song_library.py             # Song catalog, notation, and gesture mapping guide
├── trainer.py                  # Synthetic dataset generator & model trainer
├── hand_landmarker.task        # MediaPipe hand landmarker model asset
├── gesture_model.pkl           # Pre-trained 13-class gesture classifier
├── requirements.txt            # Python dependencies
└── tests/                      # 84 automated unit and integration tests
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
