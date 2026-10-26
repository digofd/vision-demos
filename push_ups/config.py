"""Params for the push-ups run: pose, reps, concentric time, depth, lockout and body line.

Edit values here, then run `python main.py`.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# A clip, or a folder whose top-level videos run in turn (subfolders skipped).
# Clips or folders on the command line replace it; `python main.py -h` lists flags.
INPUT = DATA_DIR / "input"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted, since OpenCV ignores source rotation. Poses are normalized,
# so changing EXPORT_HEIGHT needs no new gateway call (ViTPose caps at 2048px).
INFERENCE_HEIGHT = 1080   # what the model sees; changing it re-runs pose
EXPORT_HEIGHT = 1080      # the output's height in px; None keeps the source resolution
EXPORT_SQUARE = True      # the output is EXPORT_HEIGHT wide too: the clip fills the width, the panel the rest
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23           # x264 quality for the cached conversions; lower is better
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# ── HDR ──────────────────────────────────────────────────────────────────────
# iPhone HDR (HLG) looks washed out in OpenCV. "auto" tone-maps HDR-tagged
# sources, "off" never, "force" always. Exported video only.
TONEMAP = "auto"

# `bt.2446a` is the safe default; `spline` keeps midtones brighter, `hable` is filmic.
TONEMAP_ALGORITHM = "bt.2446a"

# True also feeds the model tone-mapped frames (re-runs pose).
TONEMAP_INFERENCE = False

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# True reuses a cached pose reply when its inputs match; any changed setting misses.
REUSE_POSES = True

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
EVERY_FRAME = True       # detect on every frame; costs wall clock, not money
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)
MAX_PERSONS = 1          # one athlete
PERSON_IOU_THRESHOLD = 0.6  # box overlap above which two detections are one person

DRAW_FACE = False        # face keypoints read as a scribble over the face
DRAW_BBOX = False        # draw the person's bounding box
BBOX_COLOR = (255, 160, 40)   # BGR, blue
LINE_THICKNESS = 3       # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4         # keypoint dot radius, px at a 720px-wide frame

# Skeleton color per limb, BGR; the torso stays neutral so the limbs carry color.
SKELETON_COLORS = {
    "left_arm":  (238, 211, 34),    # cyan     #22D3EE
    "right_arm": (241, 102, 99),    # indigo   #6366F1
    "left_leg":  (249, 121, 232),   # magenta  #E879F9
    "right_leg": (133, 113, 251),   # rose     #FB7185
    "torso":     (225, 213, 203),   # slate    #CBD5E1
}

# What is drawn on the athlete.
#   "near": only what is measured, on the side facing the camera: the arm, and the
#           body line (shoulder to hip to ankle) in place of the torso and leg,
#           green while straight, red past BACK_TOLERANCE_DEGREES
#   "full": the whole ViTPose skeleton, both sides, with the body line over it
# Either way, a dashed straight line from shoulder to ankle sits under the body
# line as the reference it's measured against.
OVERLAY = "near"

# ── Which side ───────────────────────────────────────────────────────────────
# Filmed side-on, the far arm and leg hide behind the body and ViTPose guesses
# them. "auto" measures with whichever side scores higher over the clip;
# "left" or "right" forces one.
SIDE = "auto"

# ── Reps ─────────────────────────────────────────────────────────────────────
# The signal is the shoulder's height above the wrist, in arm lengths: the hands
# stay planted, so it ignores the camera, and dividing by the arm cancels distance.
# 1.0 is a vertical, straight arm; 0 is the shoulder level with the wrist.
SMOOTH_MEDIAN_FRAMES = 5          # median filter, to kill isolated spikes
SMOOTH_MEAN_FRAMES = 5            # then a centered moving average, to kill jitter
TOP_PERCENTILE = 90               # relative zero: the plank at the top, not the maximum
REP_PEAK_FRACTION = 0.50          # descend past this share of rep depth to confirm a rep
REP_RESET_FRACTION = 0.20         # come back within this share of the top to close it

# Where the concentric starts and ends: walking out from the fastest point of
# the push until the speed falls under this share of it. So a pause with the
# chest on the floor, or a held plank at the top, is not timed.
CONCENTRIC_VELOCITY_FRACTION = 0.10

# Which span the CONCENTRIC column, the average and plots.png report.
#   "moving": leaving the bottom to lockout, excluding any pause on the floor
#   "ascent": the deepest point to lockout, pause included
REP_METRIC = "moving"

# ── Range of motion ──────────────────────────────────────────────────────────
# Each rep is graded on its own deepest point and its own lockout, both read off
# the rep signal (shoulder above wrist, in arm lengths). See how-it-works.md.
GRADE_ROM = True

# Down: the shoulder's height above the wrist at the deepest point, in arm
# lengths. The wrist is on the floor, so this is how close the chest got to it.
# The shoulder joint can't reach the floor (the chest is under it): chest-down
# reps on the sample clip read 0.22-0.34. A 90-degree elbow reads about 0.5.
ROM_DOWN_HEIGHT = 0.40

# Up: the shoulder's height at lockout, as a share of the plank at the top.
ROM_UP_FRACTION = 0.90

# Frames in the median window around each extreme, so one bad pose can't decide a rep.
ROM_SAMPLE_FRAMES = 5

# ── Body line ────────────────────────────────────────────────────────────────
# The hip's angle off the shoulder-ankle line: 0 is a straight plank, negative
# is the hips dropped toward the floor, positive is the hips raised up. A rep
# keeps a straight back when every frame from its descent to its lockout stays
# within this many degrees (after BACK_SAMPLE_FRAMES of median smoothing).
BACK_TOLERANCE_DEGREES = 15.0
BACK_SAMPLE_FRAMES = 5

# ── Bottom panel ─────────────────────────────────────────────────────────────
# The clip on top, the panel under it, same width. A landscape clip leaves no
# room beside it for a readable panel, so the plots go below.
PANEL_HEIGHT = 0.62        # the panel's height, as a share of the video's (unused when EXPORT_SQUARE)
PANEL_HEADER_SIZE = 40     # px at a 1280px-wide panel; scales with the output
PANEL_LIVE_SIZE = 28       # the tallies and the live body line
PANEL_NOTE_SIZE = 18       # small labels
# Display-only Gaussian smoothing (sigma, seconds) of the graphs and live values;
# grades use the raw series. 0 disables it.
PANEL_SMOOTH_SECONDS = 0.08
PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

# Credit lines, bottom-right of the panel: ("LABEL", "value") or a plain string.
#   CREDIT = ["Jeremy Park"]
#   CREDIT = [("LI", "Jeremy Park, PhD"), ("X", "@jeremyparkphd")]
# Ships blank; fill it in or pass --li, --x, --ig per run.
CREDIT = []
CREDIT_SIZE = 20                      # px at a 1280px-wide panel; scales with the output
CREDIT_FOOTER = 56                    # the credit's strip under the graphs, px at a 1280px-wide panel
CREDIT_SEPARATION = 28                # px between the credits on that one line
CREDIT_MARGIN = 22                    # inset from the right and bottom edges
CREDIT_COLOR = (180, 180, 180)        # BGR
CREDIT_LABEL_COLOR = (120, 120, 120)  # the "LI:" / "X:" part
OUTPUT_CRF = 20                       # x264 quality of the rendered video

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
