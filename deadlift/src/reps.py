"""Rep counting, off the plate track, with the hip hinge as a second signal.

**Bar height** is the primary signal: the plate's center above where it rests
on the floor, in centimeters. A plate is a known size (45 cm for an Olympic
plate, the `PLATE_DIAMETER_CM` assumption), so its own height in the frame is
the ruler: no calibration, and the number survives the camera being nearer or
further away. The ruler is only exact at the plate's own depth, which is where
the bar moves, so it measures bar travel and nothing else. It would under-read
the lifter by ~20% here: the near plate hangs ~65 cm closer to the camera than
the body's midline, so it looks bigger than a 45 cm object where the lifter is.

**The hip hinge** is the second: the angle at the hip between the shoulder and
the knee, from ViTPose. Bent over the bar the three points make a triangle;
at lockout it collapses into a line and the angle opens toward 180 degrees.
Every rep opens and closes it, bar or no bar, so it is computed on every clip:
as a cross-check on the plate's reps, and as the rep signal itself when the
plate track comes back empty or finds nothing that looks like a rep. What it
cannot tell apart is a rep and simply standing up, which the plate can.

Both are counted by the same hysteresis state machine as chin_ups: armed at
the bottom, confirmed once the signal rises past a share of the typical rep,
closed when it comes back down. The back verdict for each rep is then read off
the System One signal over that rep's own first pull.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .panel import display_smooth

from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}
_SIDES = {side: tuple(_INDEX[f"{side}_{j}"] for j in ("shoulder", "hip", "knee"))
          for side in ("left", "right")}


@dataclass
class Rep:
    number: int
    onset_frame: int      # the pull starts
    peak_frame: int       # lockout: the bar first reaches the top
    end_frame: int        # back down
    fps: float
    top_frame: int = 0    # the signal's maximum, somewhere in the hold at the top
    height_cm: float = float("nan")   # bar travel; unknown when counted off the hinge
    knee_frame: int = 0   # end of the first pull: the window the back is judged on
    hip_deg: float = float("nan")     # fullest hip extension in the rep
    back_p: float = float("nan")      # P(rounded) over the first pull
    back_label: str = ""

    @property
    def lift_seconds(self) -> float:
        return (self.peak_frame - self.onset_frame) / self.fps

    @property
    def lower_seconds(self) -> float:
        return (self.end_frame - self.peak_frame) / self.fps

    @property
    def onset_time(self) -> float:
        return self.onset_frame / self.fps

    @property
    def peak_time(self) -> float:
        return self.peak_frame / self.fps

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "onset_time": round(self.onset_time, 3),
            "peak_time": round(self.peak_time, 3),
            "end_time": round(self.end_time, 3),
            "lift_seconds": round(self.lift_seconds, 3),
            "lower_seconds": round(self.lower_seconds, 3),
            "height_cm": None if np.isnan(self.height_cm) else round(self.height_cm, 1),
            "hip_deg_max": None if np.isnan(self.hip_deg) else round(self.hip_deg, 1),
            "back": {"label": self.back_label, "p": round(self.back_p, 3)},
            "frames": {"onset": self.onset_frame, "knee": self.knee_frame,
                       "lockout": self.peak_frame, "top": self.top_frame,
                       "end": self.end_frame},
        }


@dataclass
class RepAnalysis:
    fps: float
    source: str                   # "plate" or "hinge": which signal the reps came from
    raw_cm: np.ndarray            # bar height per frame, interpolated (nan without a plate)
    height_cm: np.ndarray         # smoothed bar height
    hip_deg: np.ndarray           # smoothed hip angle per frame (nan where not posed)
    amplitude: float              # typical rep size, in the source signal's units
    reps: list[Rep] = field(default_factory=list)
    plate_reps: list[Rep] = field(default_factory=list)
    hinge_reps: list[Rep] = field(default_factory=list)
    hinge_side: str = "left"      # the body side the hinge is measured on, fixed per clip
    facing: int = -1              # -1 the lifter faces screen-left, +1 screen-right
    arc_clockwise: bool = True    # which way the hip angle opens on screen, fixed per clip

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def has_plate(self) -> bool:
        return bool(np.isfinite(self.height_cm).any())

    def done_at(self, frame: int) -> list[Rep]:
        """Reps whose lockout has been reached by *frame*, for the progressive panel."""
        return [r for r in self.reps if r.peak_frame <= frame]

    def agreement(self, min_overlap: float) -> dict:
        """How the two signals' reps line up.

        Matched on the overlap of their spans, start of pull to back down, not on
        lockout: at the top the lifter holds still, both signals sit on a plateau,
        and where each one's maximum lands inside it is noise (the hip keeps
        creeping open for half a second after the bar stops). The span's ends are
        where both move fast, so they agree.
        """
        counts = {"plate_reps": len(self.plate_reps), "hinge_reps": len(self.hinge_reps)}
        if not self.plate_reps or not self.hinge_reps:
            return {**counts, "matched": 0}

        def overlap(a: Rep, b: Rep) -> float:
            inter = min(a.end_frame, b.end_frame) - max(a.onset_frame, b.onset_frame)
            union = max(a.end_frame, b.end_frame) - min(a.onset_frame, b.onset_frame)
            return max(0, inter) / union if union > 0 else 0.0

        pairs, used = [], set()
        for p in self.plate_reps:
            best = max((h for h in self.hinge_reps if h.number not in used),
                       key=lambda h: overlap(p, h), default=None)
            if best is not None and overlap(p, best) >= min_overlap:
                used.add(best.number)
                pairs.append((p, best))
        return {**counts, "matched": len(pairs),
                "mean_onset_offset_s": round(float(np.mean(
                    [h.onset_time - p.onset_time for p, h in pairs])), 3) if pairs else None,
                "mean_end_offset_s": round(float(np.mean(
                    [h.end_time - p.end_time for p, h in pairs])), 3) if pairs else None}

    def summary(self) -> dict:
        base = {"source": self.source, "reps": self.count}
        if not self.reps:
            return base
        lifts = [r.lift_seconds for r in self.reps]
        return {
            **base,
            "typical_rep_size": round(self.amplitude, 1),
            "typical_rep_size_units": "cm of bar travel" if self.source == "plate"
                                      else "degrees of hip extension",
            "mean_lift_seconds": round(float(np.mean(lifts)), 3),
            "fastest_lift_seconds": round(float(np.min(lifts)), 3),
            "slowest_lift_seconds": round(float(np.max(lifts)), 3),
            "reps_detail": [r.as_dict() for r in self.reps],
            "plate_reps": [r.as_dict() for r in self.plate_reps],
            "hinge_reps": [r.as_dict() for r in self.hinge_reps],
        }


# ── signals ──────────────────────────────────────────────────────────────────

def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average, edge-padded so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes, a joint that jumped for one frame, without rounding peaks."""
    if k < 3:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])


def _arrays(person: dict | None):
    if not person:
        return None
    kpts = np.asarray(person.get("kpts_xy") or [], dtype=float)
    scores = np.asarray(person.get("kpts_score") or [1.0] * len(kpts), dtype=float)
    return (kpts, scores) if len(kpts) >= len(KPT_NAMES) else None


def orientation(by_index: dict[int, list[dict]]) -> tuple[str, int]:
    """``(side, facing)`` for the whole clip: which side faces the camera, which way the lifter faces.

    Decided once per clip, not per frame. Side-on, one side of the body is
    toward the camera and the other is hidden behind it, and ViTPose's
    confidence on the two sides is close enough that a per-frame choice flips
    back and forth, moving the measurement between two different hips. The
    side taken is the one ViTPose is surer of across the clip.

    Facing is where the nose and knees sit relative to the hips: both are in
    front of them for a lifter bent over a bar.
    """
    nearer = {"left": [], "right": []}
    ahead = []
    for persons in by_index.values():
        arr = _arrays(persons[0] if persons else None)
        if arr is None:
            continue
        kpts, scores = arr
        for side, idx in _SIDES.items():
            nearer[side].append(float(scores[list(idx)].min()))
        hip_x = float(np.mean([kpts[_INDEX["left_hip"]][0], kpts[_INDEX["right_hip"]][0]]))
        forward = [kpts[_INDEX["nose"]][0] - hip_x,
                   np.mean([kpts[_INDEX["left_knee"]][0], kpts[_INDEX["right_knee"]][0]]) - hip_x]
        ahead.append(float(np.sum(forward)))
    side = max(nearer, key=lambda s: float(np.mean(nearer[s])) if nearer[s] else 0.0)
    facing = 1 if ahead and float(np.median(ahead)) > 0 else -1
    return side, facing


def hinge_points(person: dict | None, *, side: str, width: int, height: int, min_score: float):
    """``(shoulder, hip, knee)`` in pixels on *side*, or None below *min_score*.

    Coordinates are normalized to width and height separately, so they are put
    back into pixels first: on a portrait clip an angle measured in normalized
    units would be squashed by the aspect ratio.
    """
    arr = _arrays(person)
    if arr is None:
        return None
    kpts, scores = arr
    idx = _SIDES[side]
    if float(scores[list(idx)].min()) < min_score:
        return None
    scale = np.array([width, height], dtype=float)
    return tuple(kpts[j] * scale for j in idx)


def screen_angle(v: np.ndarray) -> float:
    """Direction of *v* in degrees, the way OpenCV's ellipse measures it (y down)."""
    return float(np.degrees(np.arctan2(v[1], v[0])))


def arc_direction(by_index: dict[int, list[dict]], *, side: str, width: int, height: int,
                  min_score: float, bent_below: float = 150.0) -> bool:
    """Whether the hip angle opens clockwise from the knee to the shoulder, on screen.

    Read off the frames where the lifter is clearly bent, where there is no
    doubt which way it opens. Standing tall the angle is ~180 degrees and the
    knee wobbles a few pixels either side of the shoulder-hip line, so "the
    short way round" would flip between front and back from one frame to the
    next; fixing the direction for the clip is what keeps the arc on one side.
    """
    votes = []
    for persons in by_index.values():
        pts = hinge_points(persons[0] if persons else None, side=side, width=width,
                           height=height, min_score=min_score)
        if pts is None:
            continue
        shoulder, hip, knee = pts
        sweep = (screen_angle(shoulder - hip) - screen_angle(knee - hip)) % 360
        if min(sweep, 360 - sweep) < bent_below:
            votes.append(sweep < 180)
    return bool(np.mean(votes) >= 0.5) if votes else True


def hip_angles(by_index: dict[int, list[dict]], n_frames: int, *, side: str, width: int,
               height: int, min_score: float) -> np.ndarray:
    """The angle at the hip between shoulder and knee on *side*, per frame, in degrees."""
    out = np.full(n_frames, np.nan)
    for i in range(n_frames):
        persons = by_index.get(i) or []
        pts = hinge_points(persons[0] if persons else None, side=side, width=width,
                           height=height, min_score=min_score)
        if pts is None:
            continue
        shoulder, hip, knee = pts
        a, b = shoulder - hip, knee - hip
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0:
            out[i] = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))
    return out


def _amplitude(signal: np.ndarray) -> float:
    """Typical rep size: the median of the local maxima that are clearly reps.

    Not the max, so one bad sample cannot move every threshold below.
    """
    top = float(np.nanmax(signal)) if np.isfinite(signal).any() else 0.0
    peaks = [signal[i] for i in range(1, len(signal) - 1)
             if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > 0.5 * top]
    return float(np.median(peaks)) if peaks else top


def _first_reach(signal: np.ndarray, start: int, peak: int, level: float) -> int:
    above = np.where(signal[start:peak + 1] >= level)[0]
    return start + int(above[0]) if len(above) else peak


def _count(signal: np.ndarray, fps: float, *, amplitude: float, onset_level: float,
           cfg) -> list[Rep]:
    """Hysteresis over a signal that is ~0 at the bottom of a rep and peaks at lockout.

    Armed at the bottom, confirmed past `REP_PEAK_FRACTION` of the typical rep,
    closed under `REP_RESET_FRACTION`. The rep ends when the signal is back
    under `REP_FLOOR_FRACTION`, and the pull starts at the last frame before
    lockout at or under *onset_level*: a level rather than a velocity, because a
    deadlift can break off the floor slowly and a grinding start is still the
    start.

    Lockout is the first frame at `REP_LOCKOUT_FRACTION` of the rep's maximum,
    not the maximum itself. A deadlift holds at the top: the signal sits on a
    plateau there, and where its highest frame lands inside it is noise, up to
    half a second into the hold. Timed to the maximum, that hold would be
    counted as pulling.
    """
    high = cfg.REP_PEAK_FRACTION * amplitude
    low = cfg.REP_RESET_FRACTION * amplitude
    ground = cfg.REP_FLOOR_FRACTION * amplitude

    reps: list[Rep] = []
    state, valley, peak, peak_v = "down", 0, 0, -np.inf
    for i, d in enumerate(signal):
        if state == "down":
            if d <= signal[valley] or d <= ground:
                valley = i
            if d > high:
                state, peak, peak_v = "up", i, d
        else:
            if d > peak_v:
                peak, peak_v = i, d
            if d < low or i == len(signal) - 1:
                end = i
                while end < len(signal) - 1 and signal[end] > ground:
                    end += 1
                grounded = np.where(signal[valley:peak + 1] <= onset_level)[0]
                onset = valley + int(grounded[-1]) if len(grounded) else valley
                knee = _first_reach(signal, onset, peak, cfg.BACK_WINDOW_FRACTION * peak_v)
                lockout = _first_reach(signal, onset, peak, cfg.REP_LOCKOUT_FRACTION * peak_v)
                reps.append(Rep(len(reps) + 1, onset, lockout, end, fps, top_frame=peak,
                                knee_frame=knee))
                state, valley = "down", end
    return reps


def _plate_reps(plate, fps: float, *, cfg) -> tuple[np.ndarray, np.ndarray, float, list[Rep]]:
    """``(raw_cm, height_cm, amplitude, reps)`` off the plate; empty without one."""
    n = len(plate.per_frame_cy) if plate is not None else 0
    if plate is None or not plate.n_found:
        empty = np.full(n, np.nan)
        return empty, empty, 0.0, []
    floor = float(np.nanpercentile(plate.per_frame_cy, 100 - cfg.BASELINE_PERCENTILE))
    raw = (floor - plate.per_frame_cy) / plate.median_height * cfg.PLATE_DIAMETER_CM
    height = _box_filter(raw, cfg.SMOOTH_MEAN_FRAMES)
    amplitude = _amplitude(height)
    if amplitude < cfg.REP_MIN_HEIGHT_CM:
        return raw, height, amplitude, []
    reps = _count(height, fps, amplitude=amplitude, onset_level=cfg.REP_ONSET_CM, cfg=cfg)
    for rep in reps:
        rep.height_cm = float(height[rep.top_frame])
    return raw, height, amplitude, reps


def _hinge_reps(hip: np.ndarray, fps: float, *, cfg) -> tuple[float, list[Rep]]:
    """``(amplitude_deg, reps)`` off the hip angle, counted on its extension from the bottom."""
    if not np.isfinite(hip).any():
        return 0.0, []
    extension = hip - float(np.nanpercentile(hip, cfg.BASELINE_PERCENTILE))
    amplitude = _amplitude(extension)
    if amplitude < cfg.HINGE_MIN_DEGREES:
        return amplitude, []
    reps = _count(extension, fps, amplitude=amplitude,
                  onset_level=cfg.HINGE_ONSET_FRACTION * amplitude, cfg=cfg)
    return amplitude, reps


def analyze(plate, by_index: dict[int, list[dict]], n_frames: int, fps: float, *,
            width: int, height: int, cfg) -> RepAnalysis:
    """Both signals, both rep counts, and the one the rest of the run uses.

    `REP_SOURCE = "auto"` takes the plate's reps when it found any and falls
    back to the hinge when it did not: no plate in shot, SAM finding nothing
    for the prompt, or a plate that never moved like a rep.
    """
    raw_cm, height_cm, plate_amp, plate_reps = _plate_reps(plate, fps, cfg=cfg)
    if len(raw_cm) != n_frames:
        raw_cm = height_cm = np.full(n_frames, np.nan)

    side, facing = orientation(by_index)
    hip = hip_angles(by_index, n_frames, side=side, width=width, height=height,
                     min_score=cfg.HINGE_MIN_SCORE)
    hip = _box_filter(_median_filter(_fill_gaps(hip), cfg.HINGE_SMOOTH_FRAMES),
                      cfg.HINGE_SMOOTH_FRAMES)
    hinge_amp, hinge_reps = _hinge_reps(hip, fps, cfg=cfg)
    for rep in plate_reps + hinge_reps:
        # The fullest the hip opens in the rep, not its value at the lockout
        # frame: it keeps creeping open for a moment after the bar stops.
        span = hip[rep.onset_frame:rep.end_frame + 1]
        rep.hip_deg = float(np.nanmax(span)) if np.isfinite(span).any() else float("nan")

    if cfg.REP_SOURCE == "hinge" or (cfg.REP_SOURCE == "auto" and not plate_reps):
        source, reps, amplitude = "hinge", hinge_reps, hinge_amp
    else:
        source, reps, amplitude = "plate", plate_reps, plate_amp
    clockwise = arc_direction(by_index, side=side, width=width, height=height,
                              min_score=cfg.HINGE_MIN_SCORE)
    return RepAnalysis(fps, source, raw_cm, height_cm, hip, amplitude, reps,
                       plate_reps, hinge_reps, hinge_side=side, facing=facing,
                       arc_clockwise=clockwise)


def judge_backs(analysis: RepAnalysis, signal, *, cfg) -> list[int]:
    """Each rep's verdict: mean P(rounded) over the first pull, floor to knee.

    That is where the back carries the most load at the longest lever, and
    where it rounds if it is going to. The window stops before lockout because
    standing tall, every back reads straight; averaging that in dilutes the
    part of the lift that decides the answer. It starts as the pull does, not in
    the setup, because a crouched setup reads as rounded even with a flat back.

    Averaged after a short Gaussian (BACK_SMOOTH_SECONDS). The window is only a
    few hundred milliseconds, and on the median signal alone a burst of
    flipping reads shorter than the window could carry it past the threshold.

    Returns the numbers of reps the video would contradict: called rounded
    though the panel's smoother curve never crosses the threshold during the
    rep, or called straight though it does.
    """
    judged = display_smooth(signal.per_frame, signal.fps, cfg.BACK_SMOOTH_SECONDS)
    shown = display_smooth(signal.per_frame, signal.fps, cfg.PANEL_P_SMOOTH_SECONDS)
    negative = next(k for k in cfg.LABELS if k != cfg.POSITIVE_LABEL)
    disagree = []
    for rep in analysis.reps:
        window = judged[rep.onset_frame:rep.knee_frame + 1]
        rep.back_p = float(np.mean(window)) if len(window) else float("nan")
        rep.back_label = cfg.POSITIVE_LABEL if rep.back_p > cfg.THRESHOLD else negative
        on_screen = shown[rep.onset_frame:rep.end_frame + 1]
        if len(on_screen) and (rep.back_p > cfg.THRESHOLD) != (on_screen.max() > cfg.THRESHOLD):
            disagree.append(rep.number)
    return disagree


def summary_text(analysis: RepAnalysis, *, video_seconds: float, source: str,
                 plate_cm: float, plate_px: float | None, agreement: dict) -> str:
    lines = ["Deadlift summary", "=" * 58,
             f"{'source':<22}{source}",
             f"{'clip length':<22}{video_seconds:.2f}s",
             f"{'total reps':<22}{analysis.count}   (counted off the {analysis.source})"]
    if plate_px:
        lines.append(f"{'ruler':<22}plate {plate_px:.0f}px = {plate_cm:g} cm (assumed Olympic "
                     f"plate), {plate_px / plate_cm:.2f} px/cm")
    lines.append(f"{'cross-check':<22}plate {agreement['plate_reps']} reps, hinge "
                 f"{agreement['hinge_reps']}, {agreement['matched']} matched")
    if not analysis.reps:
        lines.append("\nNo reps detected. Check timeline.png.")
        return "\n".join(lines) + "\n"
    lifts = [r.lift_seconds for r in analysis.reps]
    if analysis.source == "plate":
        lines.append(f"{'typical bar travel':<22}{analysis.amplitude / 100:.2f} m")
    else:
        lines.append(f"{'typical hip extension':<22}{analysis.amplitude:.0f} degrees")
    lines += [
        f"{'mean lift':<22}{np.mean(lifts):.2f}s   (start of pull to lockout)",
        "",
        f"{'rep':>3} {'onset':>8} {'lockout':>8} {'lift':>7} {'lower':>7} {'travel':>7} "
        f"{'hip':>5}  back",
    ]
    for r in analysis.reps:
        travel = "    -  " if np.isnan(r.height_cm) else f"{r.height_cm / 100:>5.2f}m "
        hip = "   - " if np.isnan(r.hip_deg) else f"{r.hip_deg:>4.0f}°"
        lines.append(f"{r.number:>3} {r.onset_time:>7.2f}s {r.peak_time:>7.2f}s "
                     f"{r.lift_seconds:>6.2f}s {r.lower_seconds:>6.2f}s {travel} {hip}  "
                     f"{r.back_label} (P={r.back_p:.2f})")
    return "\n".join(lines) + "\n"
