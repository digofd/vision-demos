"""All knobs for the dance pose run.

Edit values here, then `python main.py`. The only command-line flags set the
credit in the corner, which changes no result: `run.json` in each output
directory snapshots these values, so a result can always be traced back to its
settings.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Console output ───────────────────────────────────────────────────────────
# The terminal says which step ran and what came back; report.txt in the run
# directory carries the diagnostics behind those lines.
SAVE_REPORT = True

# ── Input ────────────────────────────────────────────────────────────────────
INPUT_VIDEO = DATA_DIR / "input" / "dance.MOV"

# ── Conversion (.MOV -> .mp4) ────────────────────────────────────────────────
# Always converted, even from H.264: this is what guarantees the pixels the
# model sees and the pixels OpenCV reads share an orientation and a frame count.
# Pose coordinates are normalized, so one inference serves any export size and
# setting both heights equal converts once. Neither ever upscales.
INFERENCE_HEIGHT = 1080   # uploaded to the model; ViTPose caps at a 2048px long edge
EXPORT_HEIGHT = 1080      # rendered, and sets the panel's height with it
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 20          # near-transparent
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# Reuse poses cached under data/cache/ instead of calling the gateway. The key
# covers the clip and every request field, so a stale hit is not possible. True
# lets you iterate on the overlay for free; False re-detects every run.
REUSE_POSES = False

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# Detect on every frame, decimating nothing. Dance is fast, limbs reverse
# direction inside a few frames, and a pose held across a gap shows as a
# stutter. Billing is per sampled frame, so this costs wall clock, not money.
EVERY_FRAME = True

VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)

# ── Render ───────────────────────────────────────────────────────────────────
# Face keypoints and the limbs touching them read as a scribble over the face,
# so the skeleton starts at the shoulders.
DRAW_FACE = False

# A frame whose brightest pixel is this dark has no image content, so any pose
# on it was invented. This clip's last frame is fully black and the model still
# returns three poses for it. Brightest pixel rather than mean, because this is
# a night clip and "dark on average" describes most of it.
SKIP_BLANK_FRAMES = True
BLANK_FRAME_MAX_LUMA = 8

# ── Detection cleanup ────────────────────────────────────────────────────────
# The model reports ten track ids for three dancers: boxes on the roof struts,
# on a raised arm, some with negative width, plus the tracker's own identity
# handoffs. src/pose.clean() reduces them to one box per subject; see its
# docstring for what each pass fixes.

MAX_PERSONS = None   # no cap: the dancers enter at different times

# A box smaller than this fraction of the frame is not a person. Absolute rather
# than relative to its own track, since the centre dancer starts far away and
# walks in. Real detections start at 0.068 and ghosts stop at 0.03.
MIN_BOX_AREA = 0.03

# Drop a box byte-identical to that track's previous frame: a tracker coasting
# on a lost target, not a subject holding still.
DROP_FROZEN_BOXES = True

# A track must live this long and hold this median area to count as a person.
# Both, not either: lifetime alone keeps a persistent arm ghost, area alone
# keeps a handoff duplicate.
MIN_TRACK_FRACTION = 0.35
MIN_TRACK_AREA = 0.04

# One dancer wearing two ids, judged on keypoint agreement rather than box
# overlap. Overlap cannot separate this case (0.49 IoU for a split dancer
# against 0.22 for a ghost on a real one) while keypoints are decisive (0.006
# against 0.03). Merged before the lifetime test, which is what would otherwise
# kill the shorter half. 0 disables the pass.
TRACK_MERGE_DISTANCE = 0.015
TRACK_MERGE_MIN_FRAMES = 5

PERSON_IOU_THRESHOLD = 0.6   # two ids on one body during a tracker handoff

# Print the per-track table every run, not only when something is dropped: it is
# the first thing to look at when a real dancer goes missing from the overlay.
REPORT_TRACKS = True

# What the limb color means. "track" gives one color per subject, which is what
# tells three crossing skeletons apart; "side" gives left / right / torso, which
# is better on a single subject.
COLOR_BY = "track"

# The overlay is the skeleton alone. These are drawing switches only:
# src/pose.clean() still reads every box, and its area and IoU tests are what
# reject the ghosts.
DRAW_BBOX = False
BBOX_COLOR = (255, 160, 40)   # BGR; used only when COLOR_BY = "side"
DRAW_TRACK_LABEL = False      # the "id 0" tag above the box
DRAW_HUD = False              # the frame/person counter in the top-left

# Tuned for this clip's 608px rendered width with three dancers in frame. A
# single subject filling a 1080p frame wants 3/4.
LINE_THICKNESS = 2
POINT_RADIUS = 3
OUTPUT_CRF = 20

# Mux the source audio back in: OpenCV's writer has no audio track, and the MP4
# sent to the model is built with -an. "copy" passes an existing AAC track
# through untouched; name a real encoder ("aac") for a codec MP4 cannot hold.
KEEP_AUDIO = True
AUDIO_CODEC = "copy"

# ── Similarity panel ─────────────────────────────────────────────────────────
# A side panel, equal width to the clip, carrying how alike the dancers' poses
# are right now. See src/similarity.py for the metric, src/panel.py for the
# drawing, and similarity-metric-explained.md for the whole algorithm.
SIDE_PANEL = True

# ── The scored window ────────────────────────────────────────────────────────
# Only this stretch is scored, in seconds as (start, end), because a routine
# opens and closes with freestyle. This bounds the null anchor too, which is the
# part that matters: freestyle is more varied, so letting it into the baseline
# would inflate every score inside the window. Either bound may be None.
SCORE_WINDOW = (4.0, 17.4)

# What a difference is measured on. "angles" compares each segment's direction
# inside the dancer's own torso frame, which is immune to limb proportions and
# needs no scale divisor; "positions" compares hip-centred joint coordinates and
# is kept for comparison.
SIMILARITY_METRIC = "angles"

# ── Weighting ────────────────────────────────────────────────────────────────
# Flat weights gave the eight extremity joints 88% of the score while carrying
# 4-9x the noise of the shoulders and hips, so the body is weighted core
# outward. Only the ratios matter, and similarity.json reports the share each
# tier actually contributed.

# How the weights are applied. "share" divides each segment by how much it
# typically disagrees before weighting, so a 0.35 weight buys a 0.35 share;
# needed because the forearms typically differ by 40° and the torso by 5.5°.
# "raw" weights the degrees directly.
ANGLE_WEIGHT_MODE = "share"

ANGLE_WEIGHTS = {
    "torso": 1.0,           # spine lean off vertical
    "thigh": 1.0,           # hip -> knee
    "neck": 1.0,            # mid-shoulder -> mid-ear
    "shoulder_line": 0.8,   # upper-body twist
    "head_twist": 0.6,      # head turn and roll
    "upper_arm": 0.45,      # shoulder -> elbow
    "forearm": 0.2,         # elbow -> wrist
    "shin": 0.2,            # knee -> ankle
}

# The same tiering for SIMILARITY_METRIC = "positions".
POSITION_WEIGHTS = {
    "hip": 1.0, "shoulder": 1.0, "knee": 1.0,
    "elbow": 0.6, "wrist": 0.35, "ankle": 0.35,
}

# ── Timing ───────────────────────────────────────────────────────────────────
# Measured separately from posture, because a per-frame pose comparison
# conflates the two: a 3-frame lag reads as a huge difference during a fast move
# and as none at all during a hold. The signal is per-segment angular speed,
# magnitude only, since direction of change is choreography. Smoothed before
# differentiating, or a ~1.8° noise floor yields ~54 deg/s of pure noise.
TIMING_SMOOTH_FRAMES = 5
TIMING_TOLERANCE = None    # None = the measured speed noise floor (~8 deg/s)

# ── Pictures ─────────────────────────────────────────────────────────────────
# A "picture" is what dancers call the shape you land in, and choreography is a
# sequence of them. Each one is found rather than defined, as a local minimum of
# that dancer's own speed envelope. Each dancer's shape is read at their own
# arrival, so landing late in the right shape costs timing, not shape.
PICTURES = True

# Pictures inform and weight the per-frame metric; they do not replace it. Near
# a landing the timing component becomes the picture's arrival spread rather
# than the speed-envelope proxy, crossfaded by the weight below, and the
# average, ranking and bands are computed with those weights. The per-frame
# series stays continuous, which keeps the trace readable against a timestamp.
PICTURE_WEIGHTED = True

# A Gaussian bump on each picture, over a floor so the travelling still counts.
# At 0.15 a transition frame has about a seventh of a landing's say.
PICTURE_WEIGHT_SIGMA_FRAMES = 4.0
PICTURE_WEIGHT_FLOOR = 0.15

# The two questions a dancer would ask separately.
PICTURE_WEIGHTS = {"shape": 0.5, "timing": 0.5}

# Timing anchors for a picture, in milliseconds. The tolerance is the
# measurement limit, not a taste: an arrival is a discrete frame, so a three-way
# spread carries about 47 ms of rounding. The null defaults to the cluster
# window, past which a spread cannot occur by construction.
PICTURE_TIMING_TOLERANCE_MS = 50.0
PICTURE_TIMING_NULL_MS = None

# An arrival must be the lowest point within this many frames either side.
# 6 frames = 200 ms, which keeps neighbouring pictures from swallowing each
# other at ~2.3 arrivals a second.
PICTURE_MIN_GAP_FRAMES = 6

# ...and must sit this far below the local high, as a fraction of the clip's
# speed range. This is what separates arriving at a shape from slowing down
# mid-travel.
PICTURE_PROMINENCE = 0.12

# Two dancers' arrivals within this many frames are the same picture. 4 frames =
# 133 ms, half the 267 ms half-beat these dancers lock to, so it groups the same
# beat without reaching the next one. Widening it matches a few more pictures
# but the median spread jumps to 100-167 ms, which is the signature of pairing
# up different pictures rather than finding more.
PICTURE_CLUSTER_FRAMES = 4

# The shape is averaged this many frames either side of the arrival, since a
# picture is held rather than passed through. Worth 4 points of shape score.
PICTURE_HOLD_FRAMES = 1

# ── Bands ────────────────────────────────────────────────────────────────────
# The actionable output is not a percentage, it is "look at this bit", so the
# smoothed score is cut into bands and merged into timestamped segments. The
# cuts are percentiles of this clip, so the shares below are fixed by
# construction: every clip produces 10% "pay attention" however well it was
# danced. Right for finding a run's weakest moments, wrong for judging the run.

# Five labels, worst first, so four cuts; any number works, since the cuts are
# just len(labels) - 1 percentiles. Deliberately not quartiles, which would give
# a quarter of every clip to PAY ATTENTION so the panel structurally cannot
# compliment a good take. These shares are 10 / 15 / 15 / 22 / 38, and the 10th
# percentile lands on the four valleys a dancer would point at.
BAND_PERCENTILES = (10, 25, 40, 62)
BAND_LABELS = ("PAY ATTENTION", "OKAY", "GOOD", "NICE!", "CLEAN!")

# ── The readout gate, biased toward the good ────────────────────────────────
# Every control here exists twice, once for a rise and once for a fall, and the
# rising variant is always cheaper: the readout is quick to say the shape
# landed, slow to call a dip and quick to forgive one. Display only, since every
# average and the ranking come off the raw analysis. Set the pairs equal for a
# symmetric readout.

# How far past a cut the score must go before the label changes, in points.
# Hysteresis is the only thing that stops the word toggling on a boundary; a
# rise needs none, because nothing can fall back until the fall gate clears.
BAND_RISE_HYSTERESIS = 0.0
BAND_FALL_HYSTERESIS = 6.0

# The bottom band skips the hysteresis and reports as soon as it has held this
# many frames, because a valley is the one reading worth interrupting for and a
# two-frame dip in a smoothed series is noise. Only the bottom band: CLEAN!
# covers the top 40%, so letting it jump the queue swallowed the middle labels.
BAND_VALLEY_CONFIRM_FRAMES = 8

# ...and once the word changes it holds this long. Without a dwell the word
# swung between opposite extremes every 1.5s. The split by direction is what
# makes a dwell this long usable: symmetric, it blocked the recovery out of a
# dip as hard as the dip, costing most of the peaks' screen time.
BAND_RISE_DWELL_FRAMES = 30
BAND_FALL_DWELL_FRAMES = 75

# Marks at each landing on the x axis. Off because they sat on the axis line and
# read as a rendering artifact; pictures.json is where the landings are legible.
PANEL_PICTURE_TICKS = False

# The y axis floor: "auto" rounds down to a ten below the lowest displayed
# score, a number pins it. The trace never approaches 0, so a 0-100 axis gave
# its lower third to space nothing can occupy. The top is always 100, which
# means something now the ceiling is measured.
PANEL_Y_FLOOR = "auto"

# The headline is the band, not the number: a category is actionable and a
# percentage on a clip-relative scale is not. The percentage stays underneath.
PANEL_HEADLINE_BAND = True
PANEL_SHOW_PERCENT = True

# The trace ramp, BGR and low score first. Keyed on the score's percentile
# within the clip, the same basis as the bands, so the line and the label can
# never disagree.
GRAPH_RAMP = ((60, 60, 235), (60, 145, 245), (70, 215, 240), (110, 220, 105))

# Segments shorter than this drop out of the review list: a third of a second of
# dipping is not a moment anyone can act on.
BAND_MIN_SECONDS = 0.75

# ── The summary score ────────────────────────────────────────────────────────
# A weighted mean of the two components' percentages. Blending percentages is
# what makes it legitimate: degrees and degrees per second are not
# commensurable, but both are calibrated against their own null, so both read 0
# at chance and 100 at the measured ceiling. Equal by default, since being in
# time is its own category rather than a correction to being in position.
SUMMARY_WEIGHTS = {"posture": 0.5, "timing": 0.5}

# How far posture may look in time for a better match, in frames. At 0 a
# one-frame offset during a fast move craters the score even when the shapes
# match a moment apart, which is timing error charged to posture. The window
# applies to the null anchor too, or matched frames would get a best-of-five
# choice the baseline does not; timing gets no window by design.
POSTURE_MATCH_WINDOW = 2

# ── Tolerance ────────────────────────────────────────────────────────────────
# How far each segment may differ before it counts as differing at all, in
# degrees. Per body part, because a forearm 20° off is nothing while a torso 20°
# off is a different shape, and one global figure forgives a small average
# instead. Set against each part's measured typical disagreement, which is why
# the shins (12°) are forgiven far less than the forearms (38°).
ANGLE_TOLERANCE = {
    "torso": 3.0,
    "thigh": 5.0,
    "neck": 5.0,
    "shoulder_line": 4.0,
    "head_twist": 5.0,
    "upper_arm": 14.0,
    "forearm": 23.0,
    "shin": 7.0,
}

# No dead zone may exceed this fraction of its segment's typical disagreement.
# Past it the segment is fully forgiven and silently drops out of the score.
ANGLE_TOLERANCE_CAP = 0.65

# The composite tolerance, applied after weighting. None = the component's own
# measured noise floor; the per-segment zones above do the forgiving.
SIMILARITY_TOLERANCE = None

# The 100% anchor, as a percentile of the distances this clip produced. None
# keeps 100% at "identical", which three bodies at three angles through a 2D
# projection cannot reach. The cost is that this is a curve: the 10th-best
# percentile of any clip reads 100%, so the score cannot say one run was tighter
# than another. Every raw number survives in similarity.json.
SIMILARITY_CEILING_PERCENTILE = 10

# A segment shorter than this has no reliable direction, since a forearm
# pointing at the camera projects to almost nothing, so it is dropped from that
# frame rather than contributing a random angle.
MIN_SEGMENT_PX = 6.0

# ── Scale (SIMILARITY_METRIC = "positions" only) ─────────────────────────────
# Torso length is the divisor there, and it swings 26-36% within one dancer as
# bending forward foreshortens the spine. "smoothed" keeps a slow depth change
# while dropping the wobble; "median" freezes it per dancer, "per_frame" is the
# old behaviour.
SCALE_MODE = "smoothed"
SCALE_SMOOTH_FRAMES = 31

# Below this the torso is too short to normalize against, so the frame is
# skipped rather than divided by something near zero.
MIN_TORSO_PX = 8.0

# The 0% anchor compares moments at least this far apart, which is what makes
# them independent: adjacent frames are nearly the same pose, so a short lag
# would measure the choreography instead of the baseline.
NULL_LAG_SECONDS = 2.0
NULL_DISTANCE_FALLBACK = {"angles": 46.0, "positions": 0.46, "timing": 97.0}

# Fill gaps this short by interpolation before smoothing, so the trace reads as
# one line. Every gap here is a dancer clipping the frame edge for 1-3 frames;
# longer stretches stay a visible gap rather than an invented straight line.
SIMILARITY_INTERPOLATE_MAX_GAP = 15

# Display smoothing only: the raw series is what similarity.json, every average
# and the ranking use. 21 frames takes 3.2 points of jitter down to 0.77 while
# keeping the range; wider merges the pictures, which land a median 0.74s apart.
# Gaussian keeps more amplitude than boxcar, but boxcar is what every
# measurement in this file was taken under.
SIMILARITY_SMOOTH_KERNEL = "boxcar"
SIMILARITY_SMOOTH_FRAMES = 21

# How often the trace is sampled, in frames: the display's update rate, not the
# data's. At one vertex per frame the curve re-aimed 30 times a second and read
# as twitchy. Decimating vertices cannot merge landings the way a wider
# smoothing window does, and at 10 the peaks keep their height.
PANEL_TRACE_STEP = 10

# How often the big number may change. Smoothing alone does not fix a readout
# that moves every frame, since even a stable value ticks by a point constantly.
# The trace still advances every frame; only the number and its marker hold.
PANEL_REFRESH_FRAMES = 30

PANEL_SMOOTHED = True        # False shows the raw per-frame value

# Panel labels. At the end the headline switches from the live value to the clip
# average and says so, since a number labelled the same either way would be read
# as "right now".
PANEL_TITLE = "SYNC SCORE"
PANEL_FINAL_NUMBER_TITLE = "AVG SYNC SCORE"
PANEL_GRAPH_TITLE = "SYNC OVER TIME"
PANEL_Y_LABEL = "Sync Score"
PANEL_X_LABEL = "time (s)"

# The per-dancer ranking is a still, not a panel element. On the video it was
# three bars and three rank badges that never moved for 18 seconds, which reads
# as a ranking being withheld; the still shows the tie as three overlapping
# confidence intervals, which is the actual finding.
SAVE_RANKING_PLOT = True

# Ranking needs an error bar, or two of the three places are noise: dancers 1
# and 2 average 41.9% and 42.0% here. The spread comes from a block bootstrap,
# because consecutive frames of a dance are nowhere near independent and
# resampling them singly reports a false precision.
RANK_BOOTSTRAP_BLOCK_SECONDS = 1.0
RANK_BOOTSTRAP_SAMPLES = 4000
RANK_BOOTSTRAP_SEED = 7

# Two dancers share a rank (competition style: 1, 2, 2) unless one wins 85% of
# resamples. Under the angle metric P(dancer 2 > dancer 1) is 0.68, which is not
# a ranking, it is a lean.
RANK_TIE_BAND = 0.35

SAVE_SIMILARITY_PLOT = True    # similarity.png: the summary, raw vs smoothed
SAVE_COMPONENT_PLOT = True     # components.png: posture and timing side by side

# Panel typeface. OpenCV's built-in Hershey fonts have no real weight or
# kerning, which a 100px number makes obvious. "auto" takes the first available
# from src/text.py, "opencv" forces the fallback, or name a .ttf / .otf / .ttc.
PANEL_FONT = "auto"
PANEL_FONT_INDEX = None      # face index inside a .ttc; None uses the default

# The content block is centred vertically, then lifted this fraction of the
# panel height. Geometric centring reads low because the block is top-heavy and
# the credit anchors the bottom corner. 0 for exact centre.
PANEL_BLOCK_LIFT = 0.03

# Credit lines, bottom-right of the panel: ("LABEL", "value") or a plain string.
#   CREDIT = ["Jeremy Park"]
#   CREDIT = [("LI", "Jeremy Park, PhD"), ("X", "@jeremyparkphd")]
# Ships blank; fill it in or pass --li, --x, --ig per run.
CREDIT = []
CREDIT_SIZE = 22               # px at a 720px-wide reference; scales with the output
CREDIT_LINE_GAP = 6            # px between credit lines
CREDIT_MARGIN_X = 22           # inset from the right edge
CREDIT_MARGIN_Y = 18           # inset from the bottom edge
CREDIT_OPACITY = 0.9           # 0-1; white, dialled back so it reads as a credit
CREDIT_LABEL_OPACITY = 0.5     # the "LI:" / "X:" part, a shade dimmer

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

SAVE_JOINT_PLOT = False      # one joint's height over time, one line per track
PLOT_JOINT = "left_wrist"    # any name from src.skeleton.KPT_NAMES
