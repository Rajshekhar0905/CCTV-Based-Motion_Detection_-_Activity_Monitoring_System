import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
from video_processor import InvalidVideoError, VideoProcessor


class MotionDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(config.GUI_TITLE)
        self.root.resizable(False, False)

        config.ensure_directories()

        self.video_path: str = ""
        self.processing_thread: threading.Thread = None
        self.stop_event = threading.Event()
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.is_processing = False

        self._build_layout()
        self._poll_queue()


    def _build_layout(self):
        title_frame = tk.Frame(self.root, bg="#1f2937", pady=10)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame, text="CCTV MOTION DETECTION SYSTEM",
            fg="white", bg="#1f2937", font=("Helvetica", 16, "bold"),
        ).pack()

        # Video preview
        preview_frame = tk.Frame(self.root, bg="black")
        preview_frame.pack(padx=10, pady=10)
        self.preview_label = tk.Label(
            preview_frame, bg="black",
            width=config.GUI_PREVIEW_WIDTH, height=config.GUI_PREVIEW_HEIGHT,
        )
        self.preview_label.pack()
        self._show_placeholder("Select a video to begin")

        # Controls
        controls_frame = tk.Frame(self.root, pady=5)
        controls_frame.pack(fill="x", padx=10)

        self.select_btn = tk.Button(
            controls_frame, text="Select Video", width=15, command=self.select_video
        )
        self.select_btn.pack(side="left", padx=5)

        self.start_btn = tk.Button(
            controls_frame, text="Start Detection", width=15,
            command=self.start_detection, state="disabled",
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(
            controls_frame, text="Stop", width=10,
            command=self.stop_detection, state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=620, mode="determinate")
        self.progress.pack(padx=10, pady=(0, 5))

        # Status panel
        status_frame = tk.LabelFrame(self.root, text="Status", padx=10, pady=8)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Status: IDLE")
        self.motion_var = tk.StringVar(value="Motion Area: 0.00%")
        self.time_var = tk.StringVar(value="Time: 00:00:00")

        tk.Label(status_frame, textvariable=self.status_var, font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(status_frame, textvariable=self.motion_var).pack(anchor="w")
        tk.Label(status_frame, textvariable=self.time_var).pack(anchor="w")

        # Results panel
        results_frame = tk.LabelFrame(self.root, text="Results", padx=10, pady=8)
        results_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.events_var = tk.StringVar(value="Events Detected: 0")
        tk.Label(results_frame, textvariable=self.events_var, font=("Helvetica", 10)).pack(side="left")

        self.open_folder_btn = tk.Button(
            results_frame, text="Open Results Folder", command=self.open_results_folder
        )
        self.open_folder_btn.pack(side="right")

        # Selected file label
        self.file_var = tk.StringVar(value="No file selected")
        tk.Label(self.root, textvariable=self.file_var, fg="#555").pack(pady=(0, 8))

    def _show_placeholder(self, text: str):
        placeholder = Image.new(
            "RGB", (config.GUI_PREVIEW_WIDTH, config.GUI_PREVIEW_HEIGHT), color=(30, 30, 30)
        )
        tk_image = ImageTk.PhotoImage(placeholder)
        self.preview_label.configure(image=tk_image, text=text, compound="center", fg="white")
        self.preview_label.image = tk_image  # keep a reference


    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select CCTV video",
            filetypes=[("MP4 video", "*.mp4"), ("All video files", "*.avi *.mov *.mkv *.mp4"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.file_var.set(f"Selected: {os.path.basename(path)}")
        self.start_btn.configure(state="normal")
        self.status_var.set("Status: READY")

    def start_detection(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please select a video file first.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Status: PROCESSING...")
        self.progress["value"] = 0

        self.processing_thread = threading.Thread(target=self._run_processing, daemon=True)
        self.processing_thread.start()

    def stop_detection(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled")

    def open_results_folder(self):
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(results_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", results_dir])
            else:
                subprocess.Popen(["xdg-open", results_dir])
        except Exception as exc:
            messagebox.showinfo("Results folder", f"Results are saved at:\n{results_dir}\n\n({exc})")

    # ------------------------------------------------------------------
    # Background processing
    # ------------------------------------------------------------------
    def _run_processing(self):
        processor = VideoProcessor()

        def frame_callback(frame_bgr, status_text, motion_percent, timestamp_str, event_count):
            payload = {
                "type": "frame",
                "frame": frame_bgr,
                "status": status_text,
                "motion_percent": motion_percent,
                "timestamp": timestamp_str,
                "event_count": event_count,
            }
            self._push_to_queue(payload)

        def progress_callback(current, total):
            self._push_to_queue({"type": "progress", "current": current, "total": total})

        try:
            summary = processor.process(
                self.video_path,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                stop_event=self.stop_event,
            )
            self._push_to_queue({"type": "done", "summary": summary})
        except InvalidVideoError as exc:
            self._push_to_queue({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface any unexpected error to the GUI
            self._push_to_queue({"type": "error", "message": f"Unexpected error: {exc}"})

    def _push_to_queue(self, payload: dict):
        # Keep the queue small so the GUI always shows roughly the latest
        # frame instead of lagging behind on old ones during heavy load.
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(payload)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Main-thread queue polling (safe place to touch Tkinter widgets)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                payload = self.frame_queue.get_nowait()
                self._handle_payload(payload)
        except queue.Empty:
            pass
        self.root.after(config.GUI_UPDATE_INTERVAL_MS, self._poll_queue)

    def _handle_payload(self, payload: dict):
        kind = payload["type"]

        if kind == "frame":
            self._update_preview(payload["frame"])
            self.status_var.set(f"Status: {payload['status']}")
            self.motion_var.set(f"Motion Area: {payload['motion_percent']:.2f}%")
            self.time_var.set(f"Time: {payload['timestamp']}")
            self.events_var.set(f"Events Detected: {payload['event_count']}")

        elif kind == "progress":
            total = payload["total"]
            if total and total > 0:
                percent = min(100, (payload["current"] / total) * 100)
                self.progress["value"] = percent

        elif kind == "done":
            self.is_processing = False
            self.start_btn.configure(state="normal")
            self.select_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            summary = payload["summary"]
            self.progress["value"] = 100
            self.events_var.set(f"Events Detected: {summary['event_count']}")
            if summary.get("stopped_early"):
                self.status_var.set("Status: STOPPED")
                messagebox.showinfo("Stopped", "Processing stopped early. Partial results were saved.")
            else:
                self.status_var.set("Status: COMPLETE")
                messagebox.showinfo(
                    "Done",
                    f"Processing complete.\n\n"
                    f"Events detected: {summary['event_count']}\n"
                    f"Output video: {summary['output_video_path']}\n"
                    f"CSV report: {summary['csv_report_path']}",
                )

        elif kind == "error":
            self.is_processing = False
            self.start_btn.configure(state="normal")
            self.select_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Status: ERROR")
            messagebox.showerror("Error", payload["message"])

    def _update_preview(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).resize(
            (config.GUI_PREVIEW_WIDTH, config.GUI_PREVIEW_HEIGHT), Image.LANCZOS
        )
        tk_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=tk_image, text="")
        self.preview_label.image = tk_image  # keep a reference to avoid garbage collection


def launch_app():
    root = tk.Tk()
    MotionDetectionApp(root)
    root.mainloop()
