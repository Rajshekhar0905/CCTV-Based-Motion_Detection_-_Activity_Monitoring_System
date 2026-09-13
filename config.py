
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SNAPSHOTS_DIR = os.path.join(RESULTS_DIR, "snapshots")
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "output_video.mp4")
CSV_REPORT_PATH = os.path.join(RESULTS_DIR, "motion_events.csv")

# ---------------------------------------------------------------------------
# Frame pre-processing
# ---------------------------------------------------------------------------
# Width (in pixels) that every frame is resized to before processing.
# Height is scaled automatically to preserve aspect ratio.
RESIZE_WIDTH = 640

# Gaussian blur kernel size (must be an odd, positive integer).
BLUR_KERNEL_SIZE = 21

# ---------------------------------------------------------------------------
# Motion detection thresholds
# ---------------------------------------------------------------------------
# Pixel intensity threshold used when binarizing the frame difference image.
# Higher = less sensitive to small pixel changes.
PIXEL_DIFF_THRESHOLD = 25

# Minimum contour area (in pixels, on the resized frame) to be considered
# real movement instead of noise.
MIN_CONTOUR_AREA = 500

# Percentage of the frame's total pixel area that must be covered by motion
# contours for the frame to be flagged as "movement detected".
MOTION_AREA_PERCENT_THRESHOLD = 0.5  # percent (e.g. 0.5 = 0.5%)

# Kernel size for morphological dilation, used to fill holes in the
# thresholded difference mask so nearby motion blobs merge together.
MORPH_KERNEL_SIZE = 5
MORPH_DILATE_ITERATIONS = 2

# ---------------------------------------------------------------------------
# Temporal / event logic
# ---------------------------------------------------------------------------
# Maximum gap (in seconds) of "no movement" frames allowed inside a single
# ongoing motion event before the event is considered finished. This avoids
# splitting one continuous event into many tiny ones due to single-frame
# flicker.
EVENT_GAP_SECONDS = 1.0

# Minimum duration (in seconds) for an event to be recorded. Filters out
# extremely brief spurious detections.
MIN_EVENT_DURATION_SECONDS = 0.3

# ---------------------------------------------------------------------------
# Output video
# ---------------------------------------------------------------------------
OUTPUT_FOURCC = "mp4v"
BOUNDING_BOX_COLOR = (0, 0, 255)      # BGR - red
TEXT_COLOR_MOTION = (0, 0, 255)       # BGR - red
TEXT_COLOR_NO_MOTION = (0, 200, 0)    # BGR - green
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
GUI_TITLE = "CCTV Motion Detection System"
GUI_PREVIEW_WIDTH = 640
GUI_PREVIEW_HEIGHT = 360
GUI_UPDATE_INTERVAL_MS = 15  # how often the GUI polls the frame queue


def ensure_directories():
    """Create all directories the application needs, if they don't exist."""
    for path in (DATA_INPUT_DIR, RESULTS_DIR, SNAPSHOTS_DIR):
        os.makedirs(path, exist_ok=True)
