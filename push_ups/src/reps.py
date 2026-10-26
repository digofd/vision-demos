"""The rep signal, rep counting, each rep's phases, and its form grade.

A push-up starts at the top, so the signal counted on is *depth*: how far the
shoulder has come down from the plank, in arm lengths. Counting and grading are
separate: every rep is counted, shallow ones included, then graded in
`src/form.py`. See how-it-works.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import form

# Which span each cfg.REP_METRIC reports, as (Rep attribute, name, endpoints).
METRICS = {
    "moving": ("concentric_seconds", "concentric, moving only", "leaving the bottom to lockout"),
    "ascent": ("ascent_seconds", "concentric", "deepest point to lockout"),
}


def rep_value(rep, metric: str) -> float:
    """The per-rep concentric time the UI reports, per cfg.REP_METRIC."""
    return float(getattr(rep, METRICS[metric][0]))


def phase_label(metric: str) -> str:
    _, name, span = METRICS[metric]
    return f"{name} ({span})"


@dataclass
class Rep:
    """One top-to-bottom-to-top cycle. Frames are source frames, times seconds."""

    number: int
    descent_frame: int        # the eccentric starts
    bottom_frame: int         # the deepest point
    push_frame: int           # the concentric starts: leaving the bottom
    lockout_frame: int        # the concentric ends: arms straight
    descent_time: float
    bottom_time: float
    push_time: float
    lockout_time: float
    depth: float              # arm lengths below the top
    grade: "form.Grade | None" = None

    @property
    def eccentric_seconds(self) -> float:
        return self.bottom_time - self.descent_time

    @property
    def pause_seconds(self) -> float:
        """Held at the bottom, between the descent dying away and the push starting."""
        return max(0.0, self.push_time - self.bottom_time)

    @property
    def concentric_seconds(self) -> float:
        return self.lockout_time - self.push_time

    @property
    def ascent_seconds(self) -> float:
        return self.lockout_time - self.bottom_time

    @property
    def complete(self) -> bool:
        """Graded and full range; ungraded reads as complete (no marks)."""
        return self.grade is None or self.grade.complete

    @property
    def straight(self) -> bool:
        return self.grade is None or self.grade.back_ok

    def as_dict(self) -> dict:
        def r(v):
            return round(float(v), 3)
        return {
            "number": self.number,
            "descent_time": r(self.descent_time), "bottom_time": r(self.bottom_time),
            "push_time": r(self.push_time), "lockout_time": r(self.lockout_time),
            "eccentric_seconds": r(self.eccentric_seconds),
            "pause_seconds": r(self.pause_seconds),
            "concentric_seconds": r(self.concentric_seconds),
            "ascent_seconds": r(self.ascent_seconds),
            "depth_arm_lengths": r(self.depth),
            "frames": {"descent": self.descent_frame, "bottom": self.bottom_frame,
                       "push": self.push_frame, "lockout": self.lockout_frame},
            "grade": self.grade.as_dict() if self.grade else None,
        }


@dataclass
class RepAnalysis:
    indices: np.ndarray        # the source frame each returned frame is
    times: np.ndarray
    side: str
    arm: float                 # arm length, frame heights
    raw: np.ndarray            # shoulder height above the wrist, arm lengths
    height: np.ndarray         # ...smoothed
    top: float                 # the plank: relative zero for depth
    line: np.ndarray           # body line, signed degrees
    elbow: np.ndarray          # near elbow, degrees; reported beside the grade, not graded
    reps: list[Rep] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def tally(self) -> tuple[int, int, int]:
        """``(full range, straight back, graded)``."""
        graded = [r for r in self.reps if r.grade is not None]
        return (sum(r.grade.complete for r in graded), sum(r.grade.back_ok for r in graded),
                len(graded))

    def done_at(self, frame: int) -> list[Rep]:
        """Reps locked out by *frame*; their grade and time are final there."""
        return [r for r in self.reps if r.lockout_frame <= frame]

    def on_source_frames(self, series: np.ndarray, n_frames: int) -> np.ndarray:
        """*series*, indexed by returned frame, spread onto every source frame (gaps NaN)."""
        out = np.full(n_frames, np.nan)
        keep = self.indices < n_frames
        out[self.indices[keep]] = series[keep]
        return out

    def summary(self, metric: str) -> dict:
        if not self.reps:
            return {"reps": 0}
        full, straight, graded = self.tally
        values = [rep_value(r, metric) for r in self.reps]
        return {
            "reps": len(self.reps),
            "side": self.side,
            "full_rom_reps": full,
            "straight_back_reps": straight,
            "graded_reps": graded,
            "arm_length_frame_heights": round(self.arm, 4),
            "top_shoulder_height": round(self.top, 3),
            "metric": metric,
            "mean_concentric_seconds": round(float(np.mean(values)), 3),
            "fastest_concentric_seconds": round(float(np.min(values)), 3),
            "slowest_concentric_seconds": round(float(np.max(values)), 3),
            "mean_eccentric_seconds": round(float(np.mean([r.eccentric_seconds for r in self.reps])), 3),
            "mean_pause_seconds": round(float(np.mean([r.pause_seconds for r in self.reps])), 3),
            "reps_detail": [r.as_dict() for r in self.reps],
        }

    def series(self) -> dict:
        def clean(values, n):
            return [None if not np.isfinite(v) else round(float(v), n) for v in values]
        return {
            "time": [round(float(v), 4) for v in self.times],
            "shoulder_height_arm_lengths": clean(self.height, 4),
            "shoulder_height_raw": clean(self.raw, 4),
            "body_line_degrees": clean(self.line, 2),
            "elbow_degrees": clean(self.elbow, 2),
        }


def signal_config(cfg) -> dict:
    """The settings that shaped the signal and the grades, snapshotted into reps.json."""
    return {
        "side": cfg.SIDE,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES,
        "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "top_percentile": cfg.TOP_PERCENTILE,
        "peak_fraction": cfg.REP_PEAK_FRACTION,
        "reset_fraction": cfg.REP_RESET_FRACTION,
        "concentric_velocity_fraction": cfg.CONCENTRIC_VELOCITY_FRACTION,
        "metric": cfg.REP_METRIC,
        "down_height": cfg.ROM_DOWN_HEIGHT,
        "up_fraction": cfg.ROM_UP_FRACTION,
        "back_tolerance_degrees": cfg.BACK_TOLERANCE_DEGREES,
    }


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average. Centered and edge-padded, so it does not shift turnaround times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    if valid.all() or not valid.any():
        return np.where(valid, x, 0.0)
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])


def _robust_depth(depth: np.ndarray) -> float:
    """Typical rep depth, as the median of the bottoms rather than the max."""
    top = float(np.max(depth)) if len(depth) else 0.0
    if top <= 0:
        return 0.0
    bottoms = [depth[i] for i in range(1, len(depth) - 1)
               if depth[i] >= depth[i - 1] and depth[i] > depth[i + 1] and depth[i] > 0.3 * top]
    return float(np.median(bottoms)) if len(bottoms) >= 3 else float(np.percentile(depth, 95))


def _cycles(depth: np.ndarray, amplitude: float, cfg) -> list[tuple[int, int]]:
    """``(start, bottom)`` array positions per rep, by hysteresis.

    Armed near the top, confirmed once the shoulder has come down past a share of
    the typical depth, closed when it comes back up near the top.
    """
    if amplitude <= 0:
        return []
    high, low = cfg.REP_PEAK_FRACTION * amplitude, cfg.REP_RESET_FRACTION * amplitude
    out, state, start, bottom = [], "up", 0, 0
    for i, d in enumerate(depth):
        if state == "up":
            if d < depth[start]:
                start = i
            if d > high:
                state, bottom = "down", i
        else:
            if d > depth[bottom]:
                bottom = i
            if d < low:
                out.append((start, bottom))
                state, start = "up", i
    if state == "down":
        out.append((start, bottom))   # the clip ended mid-rep; closed where it stops
    return out


def _walk(speed: np.ndarray, peak: int, lo: int, hi: int, step: int, fraction: float) -> int:
    """From *peak*, step to the first frame whose speed is under *fraction* of *peak*'s.

    Stops at *lo* / *hi* if the speed never falls that far inside the window.
    """
    floor = fraction * speed[peak]
    i = peak
    while lo <= i + step <= hi:
        i += step
        if speed[i] <= floor:
            break
    return i


def analyze(frames: list[dict], fps: float, *, cfg, aspect: float) -> RepAnalysis:
    """Build the depth signal, count reps off it, time each phase and grade each rep."""
    indices = np.array([f["index"] for f in frames], dtype=int)
    times = indices / fps

    side = form.pick_side(frames, cfg.SIDE)
    joints = form.side_joints(frames, side, aspect)
    arm = form.arm_length(joints)
    raw = form.shoulder_height(joints, arm)
    line = form.body_line(joints)
    elbow = form.angle(joints.shoulder, joints.elbow, joints.wrist)

    height = _box_filter(form.median_filter(_fill_gaps(raw), cfg.SMOOTH_MEDIAN_FRAMES),
                         cfg.SMOOTH_MEAN_FRAMES)
    top = float(np.percentile(height, cfg.TOP_PERCENTILE)) if len(height) else 0.0
    depth = top - height
    amplitude = _robust_depth(depth)

    # Signed vertical speed of the shoulder, arm lengths per frame: positive going up.
    rise = np.gradient(height) if len(height) > 2 else np.zeros_like(height)
    frac = cfg.CONCENTRIC_VELOCITY_FRACTION

    cycles = _cycles(depth, amplitude, cfg)
    reps: list[Rep] = []
    for n, (start, bottom) in enumerate(cycles):
        # The concentric's window runs to the next rep's bottom; its lockout is the
        # top reached before then.
        nxt = cycles[n + 1][1] if n + 1 < len(cycles) else len(height) - 1
        top_i = bottom + int(np.argmax(height[bottom:nxt + 1]))
        fastest_up = bottom + int(np.argmax(rise[bottom:top_i + 1]))
        push = _walk(rise, fastest_up, bottom, top_i, -1, frac)
        lockout = _walk(rise, fastest_up, bottom, top_i, +1, frac)

        fall = -rise
        fastest_down = start + int(np.argmax(fall[start:bottom + 1]))
        descent = _walk(fall, fastest_down, start, bottom, -1, frac)
        # The deepest point is where the descent dies away, not the noisiest low frame
        # of a pause on the floor: the pause belongs between it and the push.
        settle = _walk(fall, fastest_down, start, bottom, +1, frac)
        low = min(settle, push)

        grade = (form.grade(raw, line, elbow, plank=top, span=(descent, lockout), push=(push, lockout),
                            cfg=cfg) if cfg.GRADE_ROM else None)
        reps.append(Rep(
            number=n + 1,
            descent_frame=int(indices[descent]), bottom_frame=int(indices[low]),
            push_frame=int(indices[push]), lockout_frame=int(indices[lockout]),
            descent_time=float(times[descent]), bottom_time=float(times[low]),
            push_time=float(times[push]), lockout_time=float(times[lockout]),
            depth=float(depth[bottom]), grade=grade))

    return RepAnalysis(indices, times, side, arm, raw, height, top, line, elbow, reps)


def summary_text(analysis: RepAnalysis, *, video_seconds: float, source: str, metric: str,
                 cfg) -> str:
    """Human-readable rep report, saved alongside the machine-readable JSON."""
    lines = ["Push-up summary", "=" * 64]
    if source:
        lines.append(f"{'source':<24}{source}")
    lines += [f"{'clip length':<24}{video_seconds:.2f}s",
              f"{'total reps':<24}{analysis.count}",
              f"{'measured on':<24}the {analysis.side} side"]
    full, straight, graded = analysis.tally
    if graded:
        lines += [f"{'full range of motion':<24}{full} of {graded}",
                  f"{'straight back':<24}{straight} of {graded}"]
    if not analysis.reps:
        lines.append("\nNo reps detected. Check plots.png.")
        return "\n".join(lines) + "\n"

    values = [rep_value(r, metric) for r in analysis.reps]
    lines += [
        "",
        f"{'concentric':<24}{phase_label(metric)}",
        f"{'mean':<24}{np.mean(values):.2f}s",
        f"{'fastest':<24}{min(values):.2f}s   (rep {int(np.argmin(values)) + 1})",
        f"{'slowest':<24}{max(values):.2f}s   (rep {int(np.argmax(values)) + 1})",
        f"{'mean eccentric':<24}{np.mean([r.eccentric_seconds for r in analysis.reps]):.2f}s",
        f"{'mean pause at bottom':<24}{np.mean([r.pause_seconds for r in analysis.reps]):.2f}s",
        f"{'pace':<24}{analysis.count / video_seconds * 60:.1f} reps/min",
        "",
        "Per rep",
        "-" * 64,
        f"{'rep':>3} {'down':>6} {'pause':>6} {'up':>6}  {'bottom':>6} {'top':>6} "
        f"{'back':>6}  grade",
    ]
    for r in analysis.reps:
        g = r.grade

        def num(v, fmt):
            return "    --" if not np.isfinite(v) else format(v, fmt)
        row = (f"{r.number:>3} {r.eccentric_seconds:>5.2f}s {r.pause_seconds:>5.2f}s "
               f"{rep_value(r, metric):>5.2f}s  ")
        if g is not None:
            row += (f"{num(g.bottom_height, '>6.2f')} {num(g.top_height, '>6.2f')} "
                    f"{num(g.back_worst, '>+5.0f')}d  {g.reason}, back {g.back_label}")
        lines.append(row)
    lines += [
        "",
        f"Held to: bottom <= {cfg.ROM_DOWN_HEIGHT:.2f}, "
        f"top >= {cfg.ROM_UP_FRACTION * analysis.top:.2f} "
        f"({cfg.ROM_UP_FRACTION:.0%} of the {analysis.top:.2f} plank), "
        f"body line within +/-{cfg.BACK_TOLERANCE_DEGREES:.0f}d.",
        "`bottom` and `top` are the shoulder's height above the wrist, in arm lengths",
        "(1 is a vertical straight arm; the wrist is on the floor). `back` is the",
        "hip's worst angle off the shoulder-ankle line: negative is hips low, positive high.",
    ]
    return "\n".join(lines) + "\n"
