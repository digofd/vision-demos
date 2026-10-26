"""The near-side joints as geometry: shoulder height, elbow angle and the body line.

Filmed side-on, the far arm and leg are hidden behind the body and ViTPose guesses
them, so everything here is read off one side, whichever the model sees best.
See how-it-works.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def pick_side(frames: list[dict], forced: str = "auto") -> str:
    """``"left"`` or ``"right"``: the side whose arm and leg score higher over the clip."""
    if forced in ("left", "right"):
        return forced
    totals = {"left": 0.0, "right": 0.0}
    for frame in frames:
        for person in (frame.get("persons") or [])[:1]:
            scores = person.get("kpts_score") or []
            for side in totals:
                for joint in JOINTS:
                    j = _INDEX[f"{side}_{joint}"]
                    if j < len(scores):
                        totals[side] += float(scores[j])
    return max(totals, key=totals.get)


@dataclass
class Joints:
    """One side's joints per returned frame, as ``(frames, 2)`` arrays in frame heights.

    x is scaled by the aspect ratio, so angles and lengths are true shapes. NaN
    where the model returned the (0, 0) "not visible" sentinel.
    """

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray


def side_joints(frames: list[dict], side: str, aspect: float) -> Joints:
    out = {joint: np.full((len(frames), 2), np.nan) for joint in JOINTS}
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        for joint in JOINTS:
            j = _INDEX[f"{side}_{joint}"]
            if j < len(kpts) and tuple(kpts[j]) != (0, 0):
                out[joint][i] = (kpts[j][0] * aspect, kpts[j][1])
    return Joints(**out)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at *b*, degrees, per row; NaN where any point is missing."""
    u, v = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def arm_length(j: Joints) -> float:
    """Upper arm plus forearm, median over the clip, in frame heights.

    The median, because the length of a limb doesn't change and one frame's
    misplaced wrist shouldn't rescale the whole signal.
    """
    lengths = (np.linalg.norm(j.shoulder - j.elbow, axis=1)
               + np.linalg.norm(j.elbow - j.wrist, axis=1))
    lengths = lengths[np.isfinite(lengths)]
    return float(np.median(lengths)) if len(lengths) else float("nan")


def shoulder_height(j: Joints, arm: float) -> np.ndarray:
    """The shoulder's height above the wrist, in arm lengths: ~1 locked out, ~0.2 chest down."""
    return (j.wrist[:, 1] - j.shoulder[:, 1]) / arm


def body_line(j: Joints) -> np.ndarray:
    """The hip's signed angle off the shoulder-ankle line, degrees.

    0 is a straight plank. Negative is the hips below the line (too low), positive
    the hips above it (too high), whichever way the athlete faces.
    """
    bend = 180.0 - angle(j.shoulder, j.hip, j.ankle)
    dx = j.shoulder[:, 0] - j.ankle[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (j.hip[:, 0] - j.ankle[:, 0]) / dx
    line_y = j.ankle[:, 1] + t * (j.shoulder[:, 1] - j.ankle[:, 1])
    # Image y grows downward, so a hip above the line has the smaller y.
    sign = np.where(j.hip[:, 1] < line_y, 1.0, -1.0)
    return bend * sign


@dataclass
class Grade:
    """One rep's form: how far down, how far back up, how straight."""

    bottom_height: float      # shoulder above the wrist at the deepest point, arm lengths
    top_height: float         # ...at lockout
    top_required: float       # what lockout had to reach: a share of this clip's plank
    back_worst: float         # the body-line reading furthest from straight, signed degrees
    # Reported, not graded: the near elbow at its most bent and at its straightest.
    bottom_elbow: float
    top_elbow: float
    bottom_ok: bool
    top_ok: bool
    back_ok: bool

    @property
    def complete(self) -> bool:
        """Full range of motion: chest down to the floor and all the way back up."""
        return self.bottom_ok and self.top_ok

    @property
    def reason(self) -> str:
        short = [name for name, ok in (("not to the floor", self.bottom_ok),
                                       ("not back up", self.top_ok)) if not ok]
        return ", ".join(short) or "full"

    @property
    def back_label(self) -> str:
        if not np.isfinite(self.back_worst):
            return "unseen"
        if self.back_ok:
            return "straight"
        return "hips low" if self.back_worst < 0 else "hips high"

    def as_dict(self) -> dict:
        def r(v, n=3):
            return None if not np.isfinite(v) else round(float(v), n)
        return {"complete": self.complete, "reason": self.reason,
                "bottom_shoulder_height": r(self.bottom_height),
                "top_shoulder_height": r(self.top_height),
                "top_required": r(self.top_required),
                "back_worst_degrees": r(self.back_worst, 1),
                "bottom_elbow_degrees": r(self.bottom_elbow, 1),
                "top_elbow_degrees": r(self.top_elbow, 1),
                "bottom_ok": self.bottom_ok, "top_ok": self.top_ok,
                "back_ok": self.back_ok, "back": self.back_label}


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Centered running median that skips NaN; NaN only where the whole window is."""
    if k < 3:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        return np.nanmedian(windows, axis=1)



def grade(height: np.ndarray, line: np.ndarray, elbow: np.ndarray, *, plank: float,
          span: tuple[int, int], push: tuple[int, int], cfg) -> Grade:
    """Grade one rep from its per-frame series.

    *height* is the shoulder above the wrist in arm lengths, *plank* its level at
    the top of the clip. *elbow* is only reported, beside the grade. *span* is the rep's descent start to lockout, *push* its
    concentric. Every reading is an extreme of a median-filtered series, so one
    bad pose can't decide a rep.
    """
    a, b = span
    height_s = median_filter(height, cfg.ROM_SAMPLE_FRAMES)
    line_s = median_filter(line, cfg.BACK_SAMPLE_FRAMES)

    def extreme(series, lo, hi, fn):
        seg = series[lo:hi + 1]
        seg = seg[np.isfinite(seg)]
        return float(fn(seg)) if len(seg) else float("nan")

    low = extreme(height_s, a, b, np.min)
    # Lockout is the highest the shoulder gets on the way up, a few frames past the
    # measured end so a lockout reached as the speed dies away still counts.
    high = extreme(height_s, push[0], min(len(height_s) - 1, push[1] + cfg.ROM_SAMPLE_FRAMES),
                   np.max)
    required = cfg.ROM_UP_FRACTION * plank
    elbow_s = median_filter(elbow, cfg.ROM_SAMPLE_FRAMES)
    bent = extreme(elbow_s, a, b, np.min)
    straightest = extreme(elbow_s, push[0],
                          min(len(elbow_s) - 1, push[1] + cfg.ROM_SAMPLE_FRAMES), np.max)

    seg = line_s[a:b + 1]
    seg = seg[np.isfinite(seg)]
    worst = float(seg[np.argmax(np.abs(seg))]) if len(seg) else float("nan")

    return Grade(
        bottom_height=low, top_height=high, top_required=required, back_worst=worst,
        bottom_elbow=bent, top_elbow=straightest,
        # An end the model couldn't see counts as short.
        bottom_ok=bool(np.isfinite(low) and low <= cfg.ROM_DOWN_HEIGHT),
        top_ok=bool(np.isfinite(high) and high >= required),
        back_ok=bool(np.isfinite(worst) and abs(worst) <= cfg.BACK_TOLERANCE_DEGREES),
    )
