"""Vertical displacement, rep counting, and per-rep range of motion.

Counting and grading are separate: every pull is counted, short ones included,
then graded full or partial in `src/rom.py`. See how-it-works.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rom as rom_mod
from .skeleton import KPT_NAMES

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

# Which span each cfg.REP_METRIC reports, as (Rep attribute, name, endpoints).
METRICS = {
    "ascent": ("ascent_seconds", "concentric", "valley to peak"),
    "moving": ("moving_seconds", "concentric, moving only", "onset to peak"),
    "total": ("total_seconds", "full cycle", "valley to valley"),
}


def rep_value(rep, metric: str) -> float:
    """The per-rep number the UI plots, per cfg.REP_METRIC."""
    return float(getattr(rep, METRICS[metric][0]))


def phase_label(metric: str) -> str:
    """Human name for the span *metric* measures, e.g. "concentric (valley to peak)"."""
    _, name, span = METRICS[metric]
    return f"{name} ({span})"


def extremes(values, *, tol: float = 1e-6) -> tuple[int, int]:
    """Indices of the fastest and slowest rep.

    Ties are common (frame-quantized), so take the earliest fastest and latest slowest.
    """
    fastest, slowest = min(values), max(values)
    fast_i = next(i for i, v in enumerate(values) if v <= fastest + tol)
    slow_i = next(i for i in range(len(values) - 1, -1, -1) if values[i] >= slowest - tol)
    return fast_i, slow_i


@dataclass
class Rep:
    """One valley-to-peak-to-valley cycle."""

    number: int
    valley_frame: int
    peak_frame: int
    end_frame: int
    valley_time: float
    peak_time: float
    end_time: float
    ascent_seconds: float
    descent_seconds: float
    total_seconds: float
    height: float
    onset_time: float = 0.0
    onset_frame: int = 0
    # Onset to peak: the ascent minus any dead hang at the bottom.
    moving_seconds: float = 0.0
    # Full or partial range, or None when grading is off.
    rom: "rom_mod.Grade | None" = None

    @property
    def complete(self) -> bool:
        """True when graded and passed; ungraded reads as complete (no marks)."""
        return self.rom is None or self.rom.complete

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "valley_time": round(self.valley_time, 3),
            "peak_time": round(self.peak_time, 3),
            "end_time": round(self.end_time, 3),
            "ascent_seconds": round(self.ascent_seconds, 3),
            "moving_seconds": round(self.moving_seconds, 3),
            "onset_time": round(self.onset_time, 3),
            "descent_seconds": round(self.descent_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "height": round(self.height, 4),
            "frames": {"valley": self.valley_frame, "onset": self.onset_frame,
                       "peak": self.peak_frame, "end": self.end_frame},
            "rom": self.rom.as_dict() if self.rom else None,
        }


@dataclass
class RepAnalysis:
    indices: np.ndarray           # the source frame each returned frame is
    times: np.ndarray
    raw: np.ndarray
    displacement: np.ndarray
    head_margin: np.ndarray
    arms: np.ndarray              # elbow angle per arm, shape (len(ARM_JOINTS), frames)
    elbow: np.ndarray             # their mean, which is what the grade reads
    clearance: np.ndarray
    baseline: float
    amplitude: float
    reps: list[Rep] = field(default_factory=list)
    counts_by_source_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    on_bar: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # The cutoffs this clip's reps were held to, from a profile or the clip itself.
    hang_required: float = float("nan")
    clearance_required: float = float("nan")
    calibrated: bool = False

    @property
    def off_bar_frames(self) -> int:
        return int((~self.on_bar).sum()) if len(self.on_bar) else 0

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def rom_tally(self) -> tuple[int, int]:
        """``(full-range reps, graded reps)``."""
        return rom_mod.tally(self.reps)


    def count_at(self, frame: int) -> int:
        """Reps completed by *frame*, for the progressive panel."""
        counts = self.counts_by_source_frame
        return int(counts[frame]) if frame < len(counts) else self.count

    def done_at(self, frame: int) -> list[Rep]:
        """Reps whose peak has been reached by *frame*; their grade is final there."""
        return [r for r in self.reps if r.peak_frame <= frame]

    def on_source_frames(self, series: np.ndarray, n_frames: int) -> np.ndarray:
        """*series*, indexed by returned frame, spread onto every source frame.

        The model can skip frames; gaps stay NaN rather than being bridged.
        """
        out = np.full(n_frames, np.nan)
        keep = self.indices < n_frames
        out[self.indices[keep]] = series[keep]
        return out

    def summary(self) -> dict:
        if not self.reps:
            return {"reps": 0}
        total = [r.total_seconds for r in self.reps]
        full, graded = self.rom_tally
        return {
            "reps": len(self.reps),
            "full_rom_reps": full,
            "graded_reps": graded,
            "calibrated": self.calibrated,
            "hang_required_degrees": None if not np.isfinite(self.hang_required)
            else round(float(self.hang_required), 1),
            "clearance_required": None if not np.isfinite(self.clearance_required)
            else round(float(self.clearance_required), 3),
            "baseline_signal": round(self.baseline, 4),
            "peak_displacement": round(self.amplitude, 4),
            "mean_ascent_seconds": round(float(np.mean([r.ascent_seconds for r in self.reps])), 3),
            "mean_descent_seconds": round(float(np.mean([r.descent_seconds for r in self.reps])), 3),
            "mean_rep_seconds": round(float(np.mean(total)), 3),
            "fastest_rep_seconds": round(float(np.min(total)), 3),
            "slowest_rep_seconds": round(float(np.max(total)), 3),
            "reps_detail": [r.as_dict() for r in self.reps],
        }

    def series(self) -> dict:
        """The full per-frame signal, for reps.json."""
        return {
            "time": [round(float(v), 4) for v in self.times],
            "displacement": [round(float(v), 5) for v in self.displacement],
            "head_margin": [round(float(v), 5) for v in self.head_margin],
            "elbow_degrees": [None if not np.isfinite(v) else round(float(v), 2)
                              for v in self.elbow],
            "left_elbow_degrees": [None if not np.isfinite(v) else round(float(v), 2)
                                   for v in self.arms[0]],
            "right_elbow_degrees": [None if not np.isfinite(v) else round(float(v), 2)
                                    for v in self.arms[1]],
            "clearance_torso_lengths": [None if not np.isfinite(v) else round(float(v), 4)
                                        for v in self.clearance],
        }


def signal_config(cfg, profile=None) -> dict:
    """The settings that shaped the signal, snapshotted into reps.json."""
    return {
        "torso_joints": list(cfg.TORSO_JOINTS),
        "hand_joints": list(cfg.HAND_JOINTS),
        "head_joints": list(cfg.HEAD_JOINTS),
        "referenced_to_hands": cfg.REFERENCE_TO_HANDS,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES,
        "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "baseline_percentile": cfg.BASELINE_PERCENTILE,
        "peak_fraction": cfg.REP_PEAK_FRACTION,
        "reset_fraction": cfg.REP_RESET_FRACTION,
        "require_head_above_hands": cfg.REQUIRE_HEAD_ABOVE_HANDS,
        "peak_fraction_calibrated": cfg.REP_PEAK_FRACTION_CALIBRATED,
        "min_flexion_fraction": cfg.REP_MIN_FLEXION_FRACTION,
        "range_of_motion": rom_mod.config(cfg, profile) if cfg.GRADE_ROM else None,
    }


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Kill isolated spikes without rounding off the peaks."""
    if k < 3:
        return x
    k += (k + 1) % 2   # force odd
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Interpolate frames where the joints were missing."""
    valid = np.isfinite(x)
    if valid.all():
        return x
    if not valid.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])


def joint_y(frames: list[dict], names: tuple[str, ...]) -> np.ndarray:
    """Mean y of the named joints on the first tracked person, per returned frame.

    (0, 0) is the model's "not visible" sentinel and is excluded.
    """
    out = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        vals = [
            kpts[_INDEX[n]][1]
            for n in names
            if _INDEX[n] < len(kpts) and tuple(kpts[_INDEX[n]]) != (0, 0)
        ]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def weighted_y(frames: list[dict], names: tuple[str, ...], min_score: float) -> np.ndarray:
    """Confidence-weighted mean y of the named joints, per returned frame.

    For the head: whichever points the model sees (ears from behind, face from the
    front) carry it, with no camera-direction setting.
    """
    out = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        scores = persons[0].get("kpts_score") or [1.0] * len(kpts)
        ys, ws = [], []
        for n in names:
            j = _INDEX[n]
            if j < len(kpts) and tuple(kpts[j]) != (0, 0) and scores[j] >= min_score:
                ys.append(kpts[j][1])
                ws.append(scores[j])
        if ws:
            out[i] = float(np.average(ys, weights=ws))
    return out

def _onset_index(displacement, valley_i: int, peak_i: int, *, amplitude: float, cfg) -> int:
    """Where the pull begins: the foot of the rise, not the lowest point.

    ``velocity`` walks back from the fastest point of the rise, steadier than
    ``threshold``, which reads the near-still, noisy bottom.
    """
    if peak_i <= valley_i:
        return valley_i

    if cfg.REP_ONSET_METHOD == "threshold":
        rise = np.where(
            displacement[valley_i:peak_i + 1]
            > displacement[valley_i] + cfg.REP_ONSET_FRACTION * amplitude
        )[0]
        return valley_i + int(rise[0]) if len(rise) else valley_i

    segment = displacement[valley_i:peak_i + 1]
    if len(segment) < 3:
        return valley_i
    velocity = np.gradient(segment)
    fastest = int(np.argmax(velocity))
    if velocity[fastest] <= 0:
        return valley_i

    floor = cfg.REP_ONSET_VELOCITY_FRACTION * velocity[fastest]
    i = fastest
    while i > 0 and velocity[i] > floor:
        i -= 1
    return valley_i + i


def _robust_amplitude(displacement: np.ndarray, valid: np.ndarray) -> float:
    """Typical rep height, as the median of the peaks rather than the max.

    Every threshold downstream is a fraction of this, so one bad pose must not move it.
    """
    usable = displacement[valid]
    if not len(usable):
        return 0.0
    top = float(np.max(usable))
    if top <= 0:
        return 0.0

    peaks = [
        displacement[i]
        for i in range(1, len(displacement) - 1)
        if valid[i]
        and displacement[i] >= displacement[i - 1]
        and displacement[i] > displacement[i + 1]
        and displacement[i] > 0.3 * top
    ]
    if len(peaks) >= 3:
        return float(np.median(peaks))
    return float(np.percentile(usable, 95))


def analyze(frames: list[dict], fps: float, n_source_frames: int, *, cfg,
            aspect: float = 1.0, profile=None) -> RepAnalysis:
    """Build the displacement signal, count reps off it, and grade each one.

    *aspect* (width / height) undoes the per-axis normalization for elbow angles.
    *profile* (a `calibrate.Profile`) sets the grading cutoffs and lets short reps count.
    """
    indices = np.array([f["index"] for f in frames], dtype=int)
    times = indices / fps

    torso = _fill_gaps(joint_y(frames, cfg.TORSO_JOINTS))
    hands = _fill_gaps(joint_y(frames, cfg.HAND_JOINTS))
    head = _fill_gaps(weighted_y(frames, cfg.HEAD_JOINTS, cfg.HEAD_MIN_SCORE))

    # Body height relative to the hands on the bar, falling back to screen
    # position if the wrists were never found.
    on_bar = np.ones(len(torso), dtype=bool)
    if cfg.REFERENCE_TO_HANDS and np.isfinite(hands).any():
        raw = hands - torso
        # Hands off the bar break the fixed-point assumption and spike the signal,
        # so those frames are excluded.
        on_bar = np.abs(hands - np.median(hands)) <= cfg.HAND_OFF_BAR_TOLERANCE
    else:
        raw = -torso

    smooth = _box_filter(_median_filter(raw, cfg.SMOOTH_MEDIAN_FRAMES), cfg.SMOOTH_MEAN_FRAMES)

    # Relative zero is the hanging plateau, taken as a low percentile rather
    # than the minimum so one bad frame cannot define it.
    usable = smooth[on_bar] if on_bar.any() else smooth
    baseline = float(np.percentile(usable, cfg.BASELINE_PERCENTILE))
    displacement = smooth - baseline
    amplitude = _robust_amplitude(displacement, on_bar)

    # Positive when the head is above the hands, the head-over-bar criterion.
    head_margin = hands - head

    # Elbow angle and head clearance per frame, sampled at each rep's turnaround.
    # Computed even with GRADE_ROM off so timeline.png can plot them.
    arms = rom_mod.arm_angles(frames, aspect, cfg.ARM_JOINTS)
    elbow = rom_mod.elbow_angles(arms)
    torso_span = rom_mod.torso_lengths(frames, cfg.TORSO_JOINTS, cfg.HEAD_JOINTS)
    with np.errstate(invalid="ignore", divide="ignore"):
        clearance = np.where(np.isfinite(torso_span) & (torso_span > 0),
                             head_margin / torso_span, np.nan)
    grader = (rom_mod.Grader(elbow, clearance, cfg=cfg, profile=profile)
              if cfg.GRADE_ROM else None)

    reps = _count(displacement, head_margin, times, indices, amplitude, on_bar,
                  cfg=cfg, grader=grader, elbow=elbow, profile=profile)
    # Graded only now: the bottom is judged against the deepest hang in the set,
    # so no rep can be called short until every rep has been measured.
    hang_required = grader.grade(reps) if grader else float("nan")
    clearance_required = grader.min_clearance if grader else float("nan")

    counts = np.zeros(max(n_source_frames, 1), dtype=int)
    for rep in reps:
        if rep.peak_frame < len(counts):
            counts[rep.peak_frame:] = rep.number
    return RepAnalysis(indices, times, raw, displacement, head_margin, arms, elbow,
                       clearance, baseline, amplitude, reps, counts, on_bar,
                       hang_required, clearance_required, profile is not None)


def _count(displacement, head_margin, times, indices, amplitude, on_bar, *, cfg,
           grader=None, elbow=None, profile=None) -> list[Rep]:
    """Hysteresis state machine: valley arms, peak confirms, return to baseline closes.

    A peak confirms by height alone, or (calibrated) by a lower height *and* elbow
    flexion, which catches half reps. `and`, not `or`: with the arms left bent,
    flexion alone would keep re-confirming reps.
    """
    if amplitude <= 0 or len(displacement) < 3:
        return []

    low = cfg.REP_RESET_FRACTION * amplitude
    high = cfg.REP_PEAK_FRACTION * amplitude

    # The loosened gate and its chaperone, both only when calibrated.
    lenient = min_flexion = None
    if profile is not None and elbow is not None:
        lenient = cfg.REP_PEAK_FRACTION_CALIBRATED * amplitude
        min_flexion = profile.min_flexion(cfg)
        hang_reference = profile.hang_degrees.mean

    def confirms(i: int, d: float) -> bool:
        if d > high:
            return True
        if lenient is None or d <= lenient:
            return False
        # An unreadable elbow falls back to the strict gate, so it cannot invent a rep.
        return bool(np.isfinite(elbow[i]) and hang_reference - elbow[i] >= min_flexion)

    reps: list[Rep] = []
    state, valley_i, peak_i, peak_v = "down", 0, 0, -np.inf
    last_on_bar = 0

    def close(end_i: int) -> None:
        reps.append(_make_rep(
            len(reps) + 1, valley_i, peak_i, end_i, times, indices, displacement,
            onset_i=_onset_index(displacement, valley_i, peak_i,
                                 amplitude=amplitude, cfg=cfg),
            # Measured here, while valley_i and peak_i are still array positions
            # (Rep stores source frames, which differ once the model skips one).
            rom=grader.measure(valley_i, peak_i) if grader else None))

    for i, d in enumerate(displacement):
        if not on_bar[i]:
            continue   # hands off the bar: this frame says nothing about reps
        last_on_bar = i
        gate = (not cfg.REQUIRE_HEAD_ABOVE_HANDS) or bool(head_margin[i] > 0)
        if state == "down":
            if d < displacement[valley_i]:
                valley_i = i
            if gate and confirms(i, d):
                state, peak_i, peak_v = "up", i, d
        else:
            if d > peak_v:
                peak_i, peak_v = i, d
            if d < low:
                close(i)
                state, valley_i = "down", i

    if state == "up":
        # Clip ended mid-rep: close on the last on-bar frame, not the clip's end,
        # so a release is not timed as a descent.
        close(last_on_bar)
    return reps


def _make_rep(number, valley_i, peak_i, end_i, times, indices, displacement,
              *, onset_i: int | None = None, rom=None) -> Rep:
    onset_i = valley_i if onset_i is None else onset_i
    return Rep(
        number=number,
        valley_frame=int(indices[valley_i]),
        peak_frame=int(indices[peak_i]),
        end_frame=int(indices[end_i]),
        valley_time=float(times[valley_i]),
        peak_time=float(times[peak_i]),
        end_time=float(times[end_i]),
        ascent_seconds=float(times[peak_i] - times[valley_i]),
        descent_seconds=float(times[end_i] - times[peak_i]),
        total_seconds=float(times[end_i] - times[valley_i]),
        height=float(displacement[peak_i]),
        onset_time=float(times[onset_i]),
        onset_frame=int(indices[onset_i]),
        moving_seconds=float(times[peak_i] - times[onset_i]),
        rom=rom,
    )


def summary_text(analysis: RepAnalysis, *, video_seconds: float, source: str = "",
                 metric: str = "ascent") -> str:
    """Human-readable rep report, saved alongside the machine-readable JSON."""
    lines = ["Pull-up summary", "=" * 58]
    if source:
        lines.append(f"{'source':<22}{source}")
    lines += [
        f"{'clip length':<22}{video_seconds:.2f}s",
        f"{'total reps':<22}{analysis.count}",
    ]

    full, graded = analysis.rom_tally
    if graded:
        lines.append(f"{'full range of motion':<22}{full} of {graded}"
                     f"   ({full / graded * 100:.0f}%)")
        lines.append(f"{'graded against':<22}"
                     + ("a calibration profile" if analysis.calibrated
                        else "config.py defaults (run calibrate.py)"))

    if not analysis.reps:
        lines.append("\nNo reps detected. Check timeline.png.")
        return "\n".join(lines) + "\n"

    s = analysis.summary()
    values = [rep_value(r, metric) for r in analysis.reps]
    totals = [r.total_seconds for r in analysis.reps]
    work = float(np.sum(totals))
    fast_i, slow_i = extremes(values)

    lines += [
        f"{'reported phase':<22}{phase_label(metric)}",
        f"{'mean':<22}{float(np.mean(values)):.2f}s",
        f"{'fastest':<22}{min(values):.2f}s   (rep {fast_i + 1})",
        f"{'slowest':<22}{max(values):.2f}s   (rep {slow_i + 1})",
        f"{'spread':<22}{max(values) - min(values):.2f}s",
        "",
        f"{'mean ascent':<22}{s['mean_ascent_seconds']:.2f}s   (valley to peak)",
        f"{'mean moving':<22}"
        f"{float(np.mean([r.moving_seconds for r in analysis.reps])):.2f}s   (onset to peak)",
        f"{'mean descent':<22}{s['mean_descent_seconds']:.2f}s",
        f"{'mean full cycle':<22}{float(np.mean(totals)):.2f}s",
        f"{'time under tension':<22}{work:.2f}s of {video_seconds:.2f}s "
        f"({work / video_seconds * 100:.0f}%)",
        f"{'pace':<22}{analysis.count / video_seconds * 60:.1f} reps/min",
        "",
        "Per rep",
        "-" * 58,
        f"{'rep':>3} {'valley':>8} {'onset':>8} {'peak':>8} {'up':>7} {'moving':>7} "
        f"{'down':>7} {'total':>7}  {'hang':>6} {'top':>6} {'clear':>6}  range",
    ]
    for r in analysis.reps:
        row = (
            f"{r.number:>3} {r.valley_time:>7.2f}s {r.onset_time:>7.2f}s {r.peak_time:>7.2f}s "
            f"{r.ascent_seconds:>6.2f}s {r.moving_seconds:>6.2f}s "
            f"{r.descent_seconds:>6.2f}s {r.total_seconds:>6.2f}s"
        )
        if r.rom is not None:
            hang = "  --  " if np.isnan(r.rom.hang_degrees) else f"{r.rom.hang_degrees:>5.0f}d"
            top = "  --  " if np.isnan(r.rom.peak_degrees) else f"{r.rom.peak_degrees:>5.0f}d"
            clear = "  --  " if np.isnan(r.rom.clearance) else f"{r.rom.clearance:>6.2f}"
            row += f"  {hang} {top} {clear}  {'full' if r.rom.complete else r.rom.reason}"
        lines.append(row)

    if graded:
        needed = []
        if np.isfinite(analysis.hang_required):
            needed.append(f"hang >= {analysis.hang_required:.1f}d")
        if np.isfinite(analysis.clearance_required):
            needed.append(f"clear >= {analysis.clearance_required:.2f}")
        lines += [
            "",
            "Range of motion",
            "-" * 58,
            f"  {rom_mod.shortfall_note(analysis.reps)}",
            f"  held to: {', '.join(needed)}" if needed else "",
            "",
            "  `hang` is the elbow angle at that rep's lowest point and `top` the",
            "  angle at its highest, so the pair is the arc the arm travelled.",
            "  `clear` is how far the head cleared the hands at the top, in torso",
            "  lengths. `top` is reported, not graded: no rep fails for pulling",
            "  too far. All three are read at that rep's own valley and peak.",
        ]

    rests = [
        analysis.reps[i + 1].valley_time - analysis.reps[i].end_time
        for i in range(len(analysis.reps) - 1)
    ]
    if rests:
        lines += [
            "",
            "Rest between reps",
            "-" * 58,
            "  " + "  ".join(f"{r:.2f}s" for r in rests)
            + f"   (mean {float(np.mean(rests)):.2f}s)",
        ]

    lines += [
        "",
        "`up` is valley to peak; `moving` excludes the dead hang before the pull",
        "starts. Ascent lengthening while range of motion holds is fatigue;",
        "range of motion collapsing is where the set actually ended.",
    ]
    return "\n".join(lines) + "\n"
