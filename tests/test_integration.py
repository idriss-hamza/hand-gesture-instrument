"""
End-to-end integration test for Hand Gesture Instrument Application.
"""

import numpy as np
import pytest
from app import HandGestureInstrumentApp
from simulator import HandSimulator, MockVideoCapture


def test_app_pipeline_full_run():
    app = HandGestureInstrumentApp(
        camera_id=0,
        model_path='gesture_model.pkl',
        initial_song='jingle_bells',
        initial_instrument='bell',
        simulate=True,
        headless=True
    )

    # Step through 20 frames in Song Guide mode
    for _ in range(20):
        ok, frame = app.step()
        assert ok is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    # Switch to Mode 2 (Gesture Scale)
    app.current_mode = app.MODE_GESTURE_SCALE
    for _ in range(5):
        ok, frame = app.step()
        assert ok is True

    # Switch to Mode 3 (Air Piano)
    app.current_mode = app.MODE_AIR_PIANO
    for _ in range(5):
        ok, frame = app.step()
        assert ok is True

    # Switch to Mode 4 (Theremin)
    app.current_mode = app.MODE_THEREMIN
    for _ in range(5):
        ok, frame = app.step()
        assert ok is True

    # Cycle song and instrument
    app.cycle_song()
    new_inst = app.audio.cycle_instrument()
    assert new_inst in ['piano', 'bell', 'violin', 'synth', 'flute']

    # Verify simulated hand detection produces valid hands and scores
    sim_hands = app.cap.get_simulated_hands()
    assert len(sim_hands) == 1
    assert sim_hands[0].score > 0.90
    gest, conf = app.gesture_model.predict(sim_hands[0].smoothed_landmarks)
    assert conf >= 0.70

    # Clean release
    app.close()
    assert app.cap.opened is False


def test_hand_simulator_frames():
    sim = HandSimulator(640, 480)
    frame, lms, gest = sim.render_frame()
    assert frame.shape == (480, 640, 3)
    assert lms.shape == (21, 3)
    assert len(gest) > 0
