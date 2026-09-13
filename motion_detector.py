from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class FrameResult:
    """Result of running motion detection on a single frame."""
    motion_detected: bool
    motion_percent: float
    boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h) on the resized frame
    thresh_mask: Optional[np.ndarray] = None


class MotionDetector:
    """
    Stateless-per-call frame-difference motion detector.

    Usage:
        detector = MotionDetector()
        gray1 = detector.preprocess(frame1)
        gray2 = detector.preprocess(frame2)
        result = detector.detect(gray1, gray2)
    """

    def __init__(
        self,
        resize_width: int = config.RESIZE_WIDTH,
        blur_kernel_size: int = config.BLUR_KERNEL_SIZE,
        pixel_diff_threshold: int = config.PIXEL_DIFF_THRESHOLD,
        min_contour_area: int = config.MIN_CONTOUR_AREA,
        motion_area_percent_threshold: float = config.MOTION_AREA_PERCENT_THRESHOLD,
        morph_kernel_size: int = config.MORPH_KERNEL_SIZE,
        morph_dilate_iterations: int = config.MORPH_DILATE_ITERATIONS,
    ):
        self.resize_width = resize_width
        # Ensure the blur kernel is odd, as required by cv2.GaussianBlur.
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.pixel_diff_threshold = pixel_diff_threshold
        self.min_contour_area = min_contour_area
        self.motion_area_percent_threshold = motion_area_percent_threshold
        self.morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.morph_dilate_iterations = morph_dilate_iterations

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to a fixed width, preserving aspect ratio."""
        h, w = frame.shape[:2]
        if w == 0:
            return frame
        scale = self.resize_width / float(w)
        new_size = (self.resize_width, max(1, int(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a raw BGR frame into the resized, grayscale, blurred
        representation used for frame differencing.
        """
        resized = self.resize_frame(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        return blurred

    def detect(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FrameResult:
        """
        Compare two preprocessed (resized+gray+blurred) frames and determine
        whether significant movement occurred between them.
        """
        if prev_gray.shape != curr_gray.shape:
            curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        frame_delta = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_delta, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(
            thresh, self.morph_kernel, iterations=self.morph_dilate_iterations
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[int, int, int, int]] = []
        motion_pixel_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            motion_pixel_area += area
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        frame_area = thresh.shape[0] * thresh.shape[1]
        motion_percent = (motion_pixel_area / frame_area) * 100.0 if frame_area else 0.0

        motion_detected = (
            len(boxes) > 0 and motion_percent >= self.motion_area_percent_threshold
        )

        return FrameResult(
            motion_detected=motion_detected,
            motion_percent=round(motion_percent, 2),
            boxes=boxes,
            thresh_mask=thresh,
        )


@dataclass
class MotionEvent:
    """A single continuous motion event."""
    event_id: int
    start_time: float
    end_time: float = 0.0
    max_motion_percentage: float = 0.0
    snapshot_taken: bool = False

    @property
    def duration(self) -> float:
        return round(max(0.0, self.end_time - self.start_time), 2)

    def to_row(self) -> dict:
        return {
            "event_id": self.event_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": self.duration,
            "max_motion_percentage": round(self.max_motion_percentage, 2),
        }


class EventTracker:
    """
    Converts a stream of per-frame (timestamp, motion_detected, motion_percent)
    observations into discrete MotionEvent records.

    An ongoing event is only closed once "no movement" has been observed
    continuously for longer than `event_gap_seconds`. This means brief
    flicker (e.g. one or two frames without detected motion in the middle of
    someone walking across a room) does not fragment a single real event
    into several tiny ones.
    """

    def __init__(
        self,
        event_gap_seconds: float = config.EVENT_GAP_SECONDS,
        min_event_duration_seconds: float = config.MIN_EVENT_DURATION_SECONDS,
    ):
        self.event_gap_seconds = event_gap_seconds
        self.min_event_duration_seconds = min_event_duration_seconds
        self._next_event_id = 1
        self._active_event: Optional[MotionEvent] = None
        self._last_motion_time: Optional[float] = None
        self.completed_events: List[MotionEvent] = []

    @property
    def active_event(self) -> Optional[MotionEvent]:
        return self._active_event

    def update(self, timestamp: float, motion_detected: bool, motion_percent: float) -> Optional[MotionEvent]:
        """
        Feed one frame's observation into the tracker.

        Returns a newly *started* MotionEvent when this frame begins a new
        event (useful for triggering a snapshot), otherwise None.
        """
        new_event_started: Optional[MotionEvent] = None

        if motion_detected:
            if self._active_event is not None:
                # An event is open, but the silent gap since the last real
                # motion frame may already have exceeded the allowed gap
                # (this can happen if motion resumes only on this exact
                # frame). Close the stale event first so it doesn't get
                # incorrectly merged with this new burst of motion.
                gap = timestamp - self._last_motion_time
                if gap > self.event_gap_seconds:
                    self._finalize_active_event()

            if self._active_event is None:
                # Start a brand new event.
                self._active_event = MotionEvent(
                    event_id=self._next_event_id,
                    start_time=timestamp,
                    end_time=timestamp,
                    max_motion_percentage=motion_percent,
                )
                self._next_event_id += 1
                new_event_started = self._active_event
            else:
                # Extend the existing event.
                self._active_event.end_time = timestamp
                self._active_event.max_motion_percentage = max(
                    self._active_event.max_motion_percentage, motion_percent
                )
            self._last_motion_time = timestamp
        else:
            if self._active_event is not None:
                gap = timestamp - self._last_motion_time
                if gap > self.event_gap_seconds:
                    # Gap too large -> close out the event. end_time stays at
                    # the last timestamp where real motion was observed (it
                    # is NOT extended into the silent gap).
                    self._finalize_active_event()
                # else: still within the allowed gap; keep the event open
                # without moving its end_time, so a short flicker doesn't
                # split one continuous event into two, but the recorded
                # end_time still reflects the last real motion frame.

        return new_event_started

    def finalize(self) -> None:
        """Call at the end of the video to close any still-open event."""
        self._finalize_active_event()

    def _finalize_active_event(self) -> None:
        if self._active_event is None:
            return
        if self._active_event.duration >= self.min_event_duration_seconds:
            self.completed_events.append(self._active_event)
        self._active_event = None
        self._last_motion_time = None
