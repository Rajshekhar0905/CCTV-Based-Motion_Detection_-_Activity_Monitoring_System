

import csv
import os
import threading
from typing import Callable, Optional

import cv2
import numpy as np

import config
from motion_detector import EventTracker, MotionDetector


class InvalidVideoError(Exception):
    """Raised when the input video file cannot be opened or read."""


class VideoProcessor:
    """
    Orchestrates reading a video, running motion detection, writing an
    annotated output video, and generating the CSV report + snapshots.
    """

    def __init__(
        self,
        detector: Optional[MotionDetector] = None,
        results_dir: str = config.RESULTS_DIR,
        snapshots_dir: str = config.SNAPSHOTS_DIR,
        output_video_path: str = config.OUTPUT_VIDEO_PATH,
        csv_report_path: str = config.CSV_REPORT_PATH,
    ):
        self.detector = detector or MotionDetector()
        self.results_dir = results_dir
        self.snapshots_dir = snapshots_dir
        self.output_video_path = output_video_path
        self.csv_report_path = csv_report_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        video_path: str,
        frame_callback: Optional[Callable[[np.ndarray, str, float, str, int], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> dict:
       
        if not video_path or not os.path.isfile(video_path):
            raise InvalidVideoError(f"Video file not found: {video_path}")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-2 or np.isnan(fps):
            fps = 20.0  # sensible fallback for malformed metadata

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            raise InvalidVideoError(f"Video file has no readable frames: {video_path}")

        resized_first = self.detector.resize_frame(first_frame)
        out_h, out_w = resized_first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
        writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (out_w, out_h))

        tracker = EventTracker()
        prev_gray = self.detector.preprocess(first_frame)

        frame_index = 1
        self._write_annotated_frame(
            writer, resized_first, motion_detected=False, motion_percent=0.0,
            boxes=[], timestamp=0.0,
        )
        if frame_callback:
            frame_callback(
                self._draw_overlay(resized_first.copy(), False, 0.0, [], 0.0),
                "NO MOVEMENT", 0.0, self._format_timestamp(0.0), 0,
            )
        if progress_callback:
            progress_callback(frame_index, total_frames)

        stopped_early = False

        while True:
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index += 1
            timestamp = frame_index / fps

            curr_gray = self.detector.preprocess(frame)
            result = self.detector.detect(prev_gray, curr_gray)
            prev_gray = curr_gray

            new_event = tracker.update(timestamp, result.motion_detected, result.motion_percent)

            resized_frame = self.detector.resize_frame(frame)
            annotated = self._draw_overlay(
                resized_frame.copy(), result.motion_detected, result.motion_percent,
                result.boxes, timestamp,
            )
            writer.write(annotated)

            if new_event is not None:
                self._save_snapshot(annotated, new_event.event_id)

            status_text = "MOVEMENT DETECTED" if result.motion_detected else "NO MOVEMENT"
            active_count = len(tracker.completed_events) + (1 if tracker.active_event else 0)

            if frame_callback:
                frame_callback(
                    annotated, status_text, result.motion_percent,
                    self._format_timestamp(timestamp), active_count,
                )
            if progress_callback:
                progress_callback(frame_index, total_frames)

        tracker.finalize()
        cap.release()
        writer.release()

        rows = [event.to_row() for event in tracker.completed_events]
        self._write_csv(rows)

        return {
            "events": rows,
            "event_count": len(rows),
            "output_video_path": self.output_video_path,
            "csv_report_path": self.csv_report_path,
            "snapshots_dir": self.snapshots_dir,
            "frames_processed": frame_index,
            "stopped_early": stopped_early,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_overlay(
        self,
        frame: np.ndarray,
        motion_detected: bool,
        motion_percent: float,
        boxes,
        timestamp: float,
    ) -> np.ndarray:
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), config.BOUNDING_BOX_COLOR, 2)

        if motion_detected:
            status_text = "MOVEMENT DETECTED"
            color = config.TEXT_COLOR_MOTION
        else:
            status_text = "NO MOVEMENT"
            color = config.TEXT_COLOR_NO_MOTION

        cv2.putText(
            frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE, color, config.FONT_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Time: {self._format_timestamp(timestamp)}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color,
            config.FONT_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Motion: {motion_percent:.2f}%", (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color,
            config.FONT_THICKNESS, cv2.LINE_AA,
        )
        return frame

    def _write_annotated_frame(self, writer, frame, motion_detected, motion_percent, boxes, timestamp):
        annotated = self._draw_overlay(frame.copy(), motion_detected, motion_percent, boxes, timestamp)
        writer.write(annotated)

    def _save_snapshot(self, frame: np.ndarray, event_id: int) -> str:
        filename = f"event_{event_id:03d}.jpg"
        path = os.path.join(self.snapshots_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def _write_csv(self, rows) -> None:
        fieldnames = ["event_id", "start_time", "end_time", "duration", "max_motion_percentage"]
        with open(self.csv_report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
