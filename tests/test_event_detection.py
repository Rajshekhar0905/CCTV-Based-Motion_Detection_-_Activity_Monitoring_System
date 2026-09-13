"""
tests/test_event_detection.py

Unit tests for the temporal event-tracking state machine (EventTracker) in
motion_detector.py, plus a test for invalid video path handling in
video_processor.py.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_detector import EventTracker  # noqa: E402
from video_processor import InvalidVideoError, VideoProcessor  # noqa: E402


@pytest.fixture
def tracker():
    return EventTracker(event_gap_seconds=1.0, min_event_duration_seconds=0.1)


def test_single_continuous_event_start_and_end(tracker):
    """A continuous run of motion frames should form exactly one event."""
    tracker.update(1.0, True, 2.0)
    tracker.update(1.5, True, 3.5)
    tracker.update(2.0, True, 2.5)
    tracker.update(2.5, False, 0.0)  # motion stops
    tracker.finalize()

    assert len(tracker.completed_events) == 1
    event = tracker.completed_events[0]
    assert event.start_time == 1.0
    assert event.end_time == 2.0  # last frame where motion was TRUE
    assert event.max_motion_percentage == 3.5


def test_event_start_returns_new_event_object(tracker):
    """update() should return a MotionEvent only on the frame that STARTS a new event."""
    result_start = tracker.update(0.5, True, 1.0)
    assert result_start is not None
    assert result_start.event_id == 1

    result_continue = tracker.update(1.0, True, 1.5)
    assert result_continue is None  # not a new event, just continuing


def test_brief_flicker_does_not_split_event(tracker):
    """
    A short gap of "no motion" that is smaller than event_gap_seconds should
    NOT create two separate events - it should remain one continuous event.
    """
    tracker.update(0.0, True, 2.0)
    tracker.update(0.5, True, 2.0)
    tracker.update(0.8, False, 0.0)   # brief flicker off (gap = 0.3s < 1.0s)
    tracker.update(1.1, True, 2.0)    # motion resumes
    tracker.update(1.5, False, 0.0)
    tracker.finalize()

    assert len(tracker.completed_events) == 1
    assert tracker.completed_events[0].start_time == 0.0
    assert tracker.completed_events[0].end_time == 1.1


def test_large_gap_creates_two_separate_events(tracker):
    """A gap longer than event_gap_seconds should split motion into two events."""
    tracker.update(0.0, True, 2.0)
    tracker.update(0.5, True, 2.0)
    tracker.update(1.0, False, 0.0)
    tracker.update(3.0, True, 2.0)   # gap of 2.0s > event_gap_seconds (1.0s)
    tracker.update(3.5, True, 2.0)
    tracker.update(4.0, False, 0.0)
    tracker.finalize()

    assert len(tracker.completed_events) == 2
    assert tracker.completed_events[0].event_id == 1
    assert tracker.completed_events[1].event_id == 2
    assert tracker.completed_events[0].end_time == 0.5
    assert tracker.completed_events[1].start_time == 3.0


def test_events_below_min_duration_are_discarded():
    """Events shorter than min_event_duration_seconds should be dropped."""
    short_tracker = EventTracker(event_gap_seconds=1.0, min_event_duration_seconds=1.0)
    short_tracker.update(0.0, True, 2.0)
    short_tracker.update(0.2, False, 0.0)  # duration only 0.2s < 1.0s minimum
    short_tracker.finalize()

    assert len(short_tracker.completed_events) == 0


def test_no_movement_produces_zero_events(tracker):
    """If motion is never detected, no events should be recorded."""
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        tracker.update(t, False, 0.0)
    tracker.finalize()

    assert len(tracker.completed_events) == 0


def test_invalid_video_path_raises_error():
    """Processing a nonexistent video file should raise InvalidVideoError."""
    processor = VideoProcessor()
    with pytest.raises(InvalidVideoError):
        processor.process("this_file_does_not_exist_12345.mp4")
