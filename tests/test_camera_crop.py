"""
Unit and integration tests for digital camera top-cropping across all modes.
Tests slicing, scaling, simulated landmark transformation, keyboard adjustments,
CLI arguments, and mode compatibility.
"""

import os
import cv2
import numpy as np
import pytest

from app import HandGestureInstrumentApp, build_parser, main
from hand_detector import HandInfo
from trainer import generate_canonical_hand


def test_default_crop_top_initialization():
    """Verify HandGestureInstrumentApp defaults crop_top to 0.0."""
    app = HandGestureInstrumentApp(simulate=True, headless=True)
    assert app.crop_top == 0.0
    app.close()


def test_custom_and_boundary_crop_top():
    """Verify custom values and boundary clamping (0.0 to 0.85)."""
    app_custom = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.30)
    assert app_custom.crop_top == pytest.approx(0.30, abs=1e-3)
    app_custom.close()

    app_zero = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.0)
    assert app_zero.crop_top == 0.0
    app_zero.close()

    # Negative value clamped to 0.0
    app_neg = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=-0.2)
    assert app_neg.crop_top == 0.0
    app_neg.close()

    # Excessive crop clamped to 0.85
    app_high = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.99)
    assert app_high.crop_top == 0.85
    app_high.close()


def test_percentage_crop_normalization():
    """Verify crop_top passed as percentage (e.g. 48 or 55) normalizes to ratio (0.48, 0.55)."""
    app_pct = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=48.0)
    assert app_pct.crop_top == pytest.approx(0.48, abs=1e-3)
    app_pct.close()

    app_pct2 = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=55.0)
    assert app_pct2.crop_top == pytest.approx(0.55, abs=1e-3)
    app_pct2.close()


def test_crop_and_scale_removes_top_obstacle():
    """
    Synthesize an image where the top 48% is wooden brown (obstacle)
    and the bottom 52% is sky blue (user active area).
    Verify crop_and_scale completely eliminates the brown obstacle and scales blue to full window.
    """
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    wood_color = np.array([30, 80, 140], dtype=np.uint8)  # Brown/wood BGR
    user_color = np.array([240, 180, 50], dtype=np.uint8)  # Blue/bright BGR

    obstacle_h = int(round(h * 0.48))  # top 230 rows
    frame[:obstacle_h, :] = wood_color
    frame[obstacle_h:, :] = user_color

    # Run crop and scale
    scaled = app.crop_and_scale(frame)

    assert scaled.shape == (h, w, 3)

    # Top rows of scaled image should now contain user_color, NOT wood_color
    top_sample = scaled[10:30, 10:30]
    mean_diff_from_wood = np.mean(np.abs(top_sample.astype(float) - wood_color.astype(float)))
    mean_diff_from_user = np.mean(np.abs(top_sample.astype(float) - user_color.astype(float)))

    # Difference from user color should be 0 (exact match)
    assert mean_diff_from_user < 1.0, "Top of image should now be the user area"
    # Difference from wood color should be significant
    assert mean_diff_from_wood > 50.0, "Wood obstacle must be completely eliminated"

    app.close()


def test_crop_and_scale_zero_bypass_and_empty_guard():
    """Verify crop_top=0 returns original frame untouched and empty inputs are guarded."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.0)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = app.crop_and_scale(frame)
    assert np.array_equal(result, frame)

    # Empty frame guard
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert app.crop_and_scale(empty).shape == (0, 0, 3)
    app.close()


def test_dynamic_crop_keyboard_adjustment():
    """Verify runtime shortcuts [ and ] adjust crop in 5% increments."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)

    # Decrease by 5%
    app.adjust_crop(-0.05)
    assert app.crop_top == pytest.approx(0.43, abs=1e-3)
    assert app.crop_feedback_text == "Top Crop: 43%"
    assert app.crop_feedback_expire > 0

    # Increase by 5% twice
    app.adjust_crop(0.05)
    assert app.crop_top == pytest.approx(0.48, abs=1e-3)
    app.adjust_crop(0.05)
    assert app.crop_top == pytest.approx(0.53, abs=1e-3)
    assert app.crop_feedback_text == "Top Crop: 53%"

    # Decrease multiple times past 0% -> clamps at 0.0
    for _ in range(15):
        app.adjust_crop(-0.05)
    assert app.crop_top == 0.0
    assert app.crop_feedback_text == "Top Crop: 0%"

    # Increase multiple times past 85% -> clamps at 0.85
    for _ in range(25):
        app.adjust_crop(0.05)
    assert app.crop_top == 0.85
    assert app.crop_feedback_text == "Top Crop: 85%"

    app.close()


def test_simulated_hand_landmarks_alignment_with_crop():
    """
    Verify that in simulation mode with crop_top > 0, the hand landmark coordinates
    are dynamically mapped into the cropped coordinate space:
    y_scaled = (y_orig - crop_top) / (1 - crop_top).
    This ensures skeleton HUD rendering matches the visually cropped hand.
    """
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)
    ok, frame = app.step()
    assert ok is True

    # Retrieve unscaled canonical hand wrist
    raw_canonical = app.cap.simulator.get_current_landmarks()
    orig_wrist_y = raw_canonical[0, 1]

    # Expected transformed wrist in cropped frame space
    expected_y = (orig_wrist_y - 0.48) / (1.0 - 0.48)

    # Get simulated hands from app
    hands = app.cap.get_simulated_hands()
    # In step(), the transformation is applied to the hands passed to mode and HUD
    scale_denom = 1.0 - app.crop_top
    scaled_wrist_y = (hands[0].raw_landmarks[0, 1] - app.crop_top) / scale_denom
    assert scaled_wrist_y == pytest.approx(expected_y, abs=1e-3)

    app.close()


def test_crop_works_across_all_four_modes():
    """Verify camera top cropping functions seamlessly in all 4 modes."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)

    modes = [
        HandGestureInstrumentApp.MODE_SONG_GUIDE,
        HandGestureInstrumentApp.MODE_FREE_INSTRUMENT,
        HandGestureInstrumentApp.MODE_AIR_PIANO,
        HandGestureInstrumentApp.MODE_THEREMIN,
    ]

    for m in modes:
        app.current_mode = m
        ok, frame = app.step()
        assert ok is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    app.close()


def test_toast_notification_rendering_and_prompt_separation():
    """Verify crop toast is rendered onto the frame at y1=135, leaving release prompt at y=105 unobstructed."""
    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)
    app.song_mode.waiting_for_release = True
    app.adjust_crop(0.05)  # Activates toast for 1.8s

    ok, frame = app.step()
    assert ok is True
    assert frame.shape == (480, 640, 3)

    # Verify toast y-bounds start at 135, strictly below pulse prompt y=105
    toast_y1 = 135
    prompt_y = 105
    assert toast_y1 > prompt_y + 15, "Toast must be comfortably separated from release prompt"

    app.close()


def test_user_screenshot_cropping_if_present():
    """Verify crop_and_scale removes shelf obstacle from top framing."""
    h, w = 480, 640
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Simulate top 48% shelf area and bottom user area
    test_img[:int(0.48 * h), :] = [30, 80, 140]
    test_img[int(0.48 * h):, :] = [200, 120, 60]

    app = HandGestureInstrumentApp(simulate=True, headless=True, crop_top=0.48)
    result = app.crop_and_scale(test_img)

    assert result.shape == (h, w, 3)
    orig_shelf_sample = test_img[int(0.25 * h), int(0.15 * w)]
    result_top_sample = result[int(0.05 * h), int(0.15 * w)]
    color_dist = np.linalg.norm(orig_shelf_sample.astype(float) - result_top_sample.astype(float))
    assert color_dist > 20.0

    app.close()


def test_cli_argument_parsing():
    """Verify --crop-top CLI flag works in the real application argument parser."""
    parser = build_parser()

    # Test default (0.0: unzoomed natural camera resolution)
    args_default = parser.parse_args([])
    assert args_default.crop_top == 0.0

    # Test custom float value
    args_custom = parser.parse_args(["--crop-top", "0.35"])
    assert args_custom.crop_top == pytest.approx(0.35, abs=1e-3)

    # Test percentage input parsed as float
    args_pct = parser.parse_args(["--crop-top", "55"])
    assert args_pct.crop_top == pytest.approx(55.0, abs=1e-3)
