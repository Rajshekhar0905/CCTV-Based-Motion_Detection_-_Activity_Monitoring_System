"""
tests/test_motion_detector.py

Unit tests for the classical frame-difference motion detection algorithm in
motion_detector.py. These tests use small synthetic NumPy frames instead of
real video so they run instantly and deterministically.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_detector import MotionDetector  # noqa: E402


@pytest.fixture
def detector():
    return MotionDetector(
        resize_width=100,
        blur_kernel_size=5,
        pixel_diff_threshold=25,
        min_contour_area=20,
        motion_area_percent_threshold=0.5,
        morph_kernel_size=3,
        morph_dilate_iterations=1,
    )


def make_frame(width=100, height=100, box=None, color=255, bg=30):
    """Create a synthetic BGR frame, optionally with a filled rectangle."""
    frame = np.full((height, width, 3), bg, dtype=np.uint8)
    if box is not None:
        x, y, w, h = box
        frame[y:y + h, x:x + w] = color
    return frame


def test_no_movement_between_identical_frames(detector):
    """Two identical frames should never be reported as movement."""
    frame_a = make_frame(box=(10, 10, 20, 20))
    frame_b = frame_a.copy()

    gray_a = detector.preprocess(frame_a)
    gray_b = detector.preprocess(frame_b)

    result = detector.detect(gray_a, gray_b)

    assert result.motion_detected is False
    assert result.motion_percent == 0.0
    assert result.boxes == []


def test_movement_detected_between_different_frames(detector):
    """A large object appearing between two frames should trigger detection."""
    frame_a = make_frame()
    frame_b = make_frame(box=(20, 20, 40, 40))  # large bright block appears

    gray_a = detector.preprocess(frame_a)
    gray_b = detector.preprocess(frame_b)

    result = detector.detect(gray_a, gray_b)

    assert result.motion_detected is True
    assert result.motion_percent > 0.0
    assert len(result.boxes) >= 1


def test_small_noise_is_filtered_by_contour_area_threshold():
    """
    A tiny change (a couple of pixels) should NOT be reported as movement
    when min_contour_area is set high enough to filter noise.
    """
    strict_detector = MotionDetector(
        resize_width=100,
        blur_kernel_size=1,  # disable blur so the tiny box survives distinctly
        pixel_diff_threshold=25,
        min_contour_area=5000,  # much larger than the tiny box below
        motion_area_percent_threshold=0.5,
        morph_kernel_size=1,
        morph_dilate_iterations=0,
    )

    frame_a = make_frame()
    frame_b = make_frame(box=(50, 50, 2, 2))  # 2x2 pixel change = noise

    gray_a = strict_detector.preprocess(frame_a)
    gray_b = strict_detector.preprocess(frame_b)

    result = strict_detector.detect(gray_a, gray_b)

    assert result.motion_detected is False


def test_threshold_behavior_pixel_diff_threshold():
    """
    Raising pixel_diff_threshold above the actual intensity difference should
    suppress detection of a change that would otherwise be detected.
    """
    frame_a = make_frame(bg=100)
    frame_b = make_frame(bg=100, box=(20, 20, 30, 30), color=110)  # small intensity delta of 10

    lenient = MotionDetector(resize_width=100, blur_kernel_size=1, pixel_diff_threshold=5,
                              min_contour_area=20, motion_area_percent_threshold=0.1,
                              morph_kernel_size=1, morph_dilate_iterations=0)
    strict = MotionDetector(resize_width=100, blur_kernel_size=1, pixel_diff_threshold=50,
                             min_contour_area=20, motion_area_percent_threshold=0.1,
                             morph_kernel_size=1, morph_dilate_iterations=0)

    gray_a_lenient = lenient.preprocess(frame_a)
    gray_b_lenient = lenient.preprocess(frame_b)
    result_lenient = lenient.detect(gray_a_lenient, gray_b_lenient)

    gray_a_strict = strict.preprocess(frame_a)
    gray_b_strict = strict.preprocess(frame_b)
    result_strict = strict.detect(gray_a_strict, gray_b_strict)

    assert result_lenient.motion_detected is True
    assert result_strict.motion_detected is False


def test_motion_percentage_increases_with_object_size(detector):
    """A larger moving object should produce a higher motion percentage."""
    frame_a = make_frame()
    small_box_frame = make_frame(box=(10, 10, 10, 10))
    large_box_frame = make_frame(box=(10, 10, 60, 60))

    gray_a = detector.preprocess(frame_a)
    gray_small = detector.preprocess(small_box_frame)
    gray_large = detector.preprocess(large_box_frame)

    result_small = detector.detect(gray_a, gray_small)
    result_large = detector.detect(gray_a, gray_large)

    assert result_large.motion_percent > result_small.motion_percent


def test_resize_preserves_aspect_ratio():
    detector = MotionDetector(resize_width=200)
    frame = make_frame(width=400, height=200)  # 2:1 aspect ratio
    resized = detector.resize_frame(frame)
    assert resized.shape[1] == 200
    assert resized.shape[0] == 100  # preserves 2:1 ratio
