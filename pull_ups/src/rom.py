"""Grading each rep's range of motion: full or partial.

Bottom: elbow angle at the valley, judged against the clip's deepest hang, since
ViTPose puts the shoulder on the deltoid and a dead hang reads ~167°, not 180°.
Top: head above hands at the peak, in torso lengths. See how-it-works.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

# Why each grade came out the way it did, in the order the conditions are read.
REASONS = {
    ("ok", "ok"): "full range",
    ("short", "ok"): "no full hang at the bottom",
    ("ok", "short"): "head did not clear the bar",
    ("short", "short"): "short at both ends",
}


@dataclass(frozen=True)
class Measurement:
    """One rep's extremes, measured before any verdict.

    Kept apart from Grade because the bottom is judged against the set's best hang.
    """

    hang_degrees: float       # elbow angle at the valley; 180 is a straight arm
    peak_degrees: float       # elbow angle at the peak; the fully folded arm
    clearance: float          # head above hands at the peak, in torso lengths


@dataclass(frozen=True)
class Grade:
    """One rep's verdict, with the numbers and the bar it was held to."""

    complete: bool
    hang_degrees: float
    # Reported, not graded: nothing fails for pulling too far. Calibration uses it
    # for the sweep.
    peak_degrees: float
    clearance: float
    bottom_ok: bool
    top_ok: bool
    hang_required: float      # the cutoff this rep's hang was actually held to
    clearance_required: float

    @property
    def reason(self) -> str:
        return REASONS[("ok" if self.bottom_ok else "short",
                        "ok" if self.top_ok else "short")]

    def as_dict(self) -> dict:
        return {
            "complete": self.complete,
            "hang_degrees": None if np.isnan(self.hang_degrees)
            else round(float(self.hang_degrees), 1),
            "peak_degrees": None if np.isnan(self.peak_degrees)
            else round(float(self.peak_degrees), 1),
            "clearance_torso_lengths": None if np.isnan(self.clearance)
            else round(float(self.clearance), 3),
            "hang_required_degrees": round(float(self.hang_required), 1),
            "clearance_required": round(float(self.clearance_required), 3),
            "bottom_ok": self.bottom_ok,
            "top_ok": self.top_ok,
            "reason": self.reason,
        }


def _point(kpts, name: str, aspect: float):
    """One keypoint as (x, y) in units of frame height, or None if unseen.

    x is scaled by *aspect* so angles are not distorted on non-square frames.
    """
    i = _INDEX[name]
    if i >= len(kpts):
        return None
    x, y = kpts[i][0], kpts[i][1]
    if (x, y) == (0, 0):   # the model's "not visible" sentinel
        return None
    return np.array([x * aspect, y], dtype=float)


def _interior_angle(a, b, c) -> float:
    """Degrees at *b*, between b->a and b->c."""
    u, v = a - b, c - b
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1.0, 1.0))))


def arm_angles(frames: list[dict], aspect: float, arms) -> np.ndarray:
    """Elbow angle per arm per returned frame, shape ``(len(arms), len(frames))``.

    NaN where a joint is unseen. Per arm so the panel can show one arm stopping short.
    """
    out = np.full((len(arms), len(frames)), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        for a, (shoulder, elbow, wrist) in enumerate(arms):
            pts = [_point(kpts, n, aspect) for n in (shoulder, elbow, wrist)]
            if all(p is not None for p in pts):
                out[a, i] = _interior_angle(*pts)
    return out


def elbow_angles(per_arm: np.ndarray) -> np.ndarray:
    """Mean elbow angle per frame across the arms that were seen, NaN where none were.

    Averaging damps a single bad solve. This is what the grade reads.
    """
    seen = np.isfinite(per_arm)
    total = np.where(seen, per_arm, 0.0).sum(axis=0)
    count = seen.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def torso_lengths(frames: list[dict], torso_joints, head_joints) -> np.ndarray:
    """Shoulder-to-hip height per frame, the body's own ruler.

    Vertical only, like the head clearance it normalizes, so rotation does not shift it.
    """
    shoulders = tuple(j for j in torso_joints if "shoulder" in j)
    hips = tuple(j for j in torso_joints if "hip" in j)
    if not shoulders or not hips:
        # No hips: fall back to head-to-shoulder, noisier but still the subject's scale.
        shoulders, hips = head_joints, tuple(j for j in torso_joints if "shoulder" in j)

    out = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []

        def mean_y(names):
            vals = [kpts[_INDEX[n]][1] for n in names
                    if _INDEX[n] < len(kpts) and tuple(kpts[_INDEX[n]]) != (0, 0)]
            return float(np.mean(vals)) if vals else np.nan

        span = abs(mean_y(hips) - mean_y(shoulders))
        if np.isfinite(span) and span > 1e-4:
            out[i] = span
    return out


def _window_median(series: np.ndarray, center: int, width: int) -> float:
    """Median of *series* over a window centered on *center*, ignoring NaNs.

    So one bad pose at the turnaround cannot decide whether a rep counted.
    """
    if not len(series):
        return np.nan
    half = max(0, int(width) // 2)
    lo, hi = max(0, center - half), min(len(series), center + half + 1)
    window = series[lo:hi]
    window = window[np.isfinite(window)]
    return float(np.median(window)) if len(window) else np.nan


class Grader:
    """Measures each rep during counting, then grades them all together.

    Two phases because the bottom is graded against the best hang in the whole set.
    """

    def __init__(self, elbow: np.ndarray, clearance: np.ndarray, *, cfg, profile=None):
        self.elbow = elbow
        self.clearance = clearance
        self.profile = profile
        self.hang_floor = cfg.ROM_HANG_ELBOW_DEGREES
        self.hang_tolerance = cfg.ROM_HANG_TOLERANCE_DEGREES
        self.sample = cfg.ROM_SAMPLE_FRAMES

        # A profile fixes both cutoffs up front; otherwise `grade` sets the hang
        # cutoff once it has seen the whole set.
        if profile is not None:
            self.hang_required = float(profile.hang_required(cfg))
            self.min_clearance = float(profile.clearance_required(cfg))
        else:
            self.hang_required = float(cfg.ROM_HANG_ELBOW_DEGREES)
            self.min_clearance = float(cfg.ROM_CLEARANCE_TORSO_FRACTION)

    def measure(self, valley_i: int, peak_i: int) -> Measurement:
        return Measurement(
            hang_degrees=_window_median(self.elbow, valley_i, self.sample),
            peak_degrees=_window_median(self.elbow, peak_i, self.sample),
            clearance=_window_median(self.clearance, peak_i, self.sample),
        )

    def reference(self, measurements) -> float:
        """The clip's own straight arm: the deepest hang in it.

        Max, not a percentile: each value is already a window median, and sets are small.
        """
        hangs = [m.hang_degrees for m in measurements if np.isfinite(m.hang_degrees)]
        return max(hangs) if hangs else np.nan

    def grade(self, reps) -> float:
        """Replace every rep's Measurement with a Grade. Returns the cutoff used."""
        if self.profile is None:
            measured = [r.rom for r in reps if isinstance(r.rom, Measurement)]
            best = self.reference(measured)

            # No profile: the clip is its own reference, with an absolute floor for
            # a set where every rep was short.
            required = self.hang_floor
            if np.isfinite(best):
                required = max(required, best - self.hang_tolerance)
            self.hang_required = float(required)

        required = self.hang_required
        for rep in reps:
            m = rep.rom
            if not isinstance(m, Measurement):
                continue
            # Unreadable is not passing: assuming a dead hang would flatter the athlete.
            bottom_ok = bool(np.isfinite(m.hang_degrees) and m.hang_degrees >= required)
            top_ok = bool(np.isfinite(m.clearance) and m.clearance >= self.min_clearance)
            rep.rom = Grade(complete=bottom_ok and top_ok,
                            hang_degrees=m.hang_degrees, peak_degrees=m.peak_degrees,
                            clearance=m.clearance,
                            bottom_ok=bottom_ok, top_ok=top_ok,
                            hang_required=self.hang_required,
                            clearance_required=self.min_clearance)
        return self.hang_required


def config(cfg, profile=None) -> dict:
    """The thresholds a grade was reached under, snapshotted into reps.json."""
    return {
        "calibrated": profile is not None,
        "profile": profile.as_dict() if profile is not None else None,
        "hang_tolerance_degrees": cfg.ROM_HANG_TOLERANCE_DEGREES,
        "clearance_fraction_of_reference": cfg.ROM_CLEARANCE_FRACTION,
        "uncalibrated_hang_elbow_floor_degrees": cfg.ROM_HANG_ELBOW_DEGREES,
        "uncalibrated_clearance_torso_fraction": cfg.ROM_CLEARANCE_TORSO_FRACTION,
        "sample_frames": cfg.ROM_SAMPLE_FRAMES,
        "arm_joints": [list(a) for a in cfg.ARM_JOINTS],
    }


def tally(reps) -> tuple[int, int]:
    """``(complete, total)`` across graded reps."""
    graded = [r for r in reps if isinstance(r.rom, Grade)]
    return sum(1 for r in graded if r.rom.complete), len(graded)


def shortfall_note(reps) -> str:
    """One line naming (not just counting) which reps were cut short, and why."""
    graded = [r for r in reps if isinstance(r.rom, Grade)]
    if not graded:
        return ""
    short = [r for r in graded if not r.rom.complete]
    if not short:
        return "every rep hit full range"

    # Group by cause, so a set that failed one way reads as one failure.
    by_reason: dict[str, list[int]] = {}
    for rep in short:
        by_reason.setdefault(rep.rom.reason, []).append(rep.number)
    if len(by_reason) == 1:
        reason, numbers = next(iter(by_reason.items()))
        label = "rep" if len(numbers) == 1 else "reps"
        return f"{label} {', '.join(map(str, numbers))}: {reason}"
    return f"{len(short)} of {len(graded)} reps cut short"
