"""The two CSVs a person opens in a spreadsheet, in the vision-demos app's format.

    <clip>_summary.csv   one row per rep: grades, phase times, heights, angles
    <clip>_frames.csv    one row per video frame, every measurement the run took,
                         with the rep and phase it falls in

Same conventions as `vision-demos/exports.py`: `frame, time_s, rep` lead
the frame rows, numbers are rounded, and a missing reading is a blank cell
rather than NaN.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .reps import RepAnalysis


def _num(value, digits: int = 3):
    """A number for a CSV cell: rounded, and blank rather than NaN or None."""
    if value is None or isinstance(value, bool):
        return value
    value = float(value)
    return "" if math.isnan(value) else round(value, digits)


def summary_rows(analysis: RepAnalysis) -> list[dict]:
    rows = []
    for r in analysis.reps:
        g = r.grade
        rows.append({
            "rep": r.number,
            "full_range_of_motion": "" if g is None else ("yes" if g.complete else "no"),
            "straight_back": "" if g is None else ("yes" if g.back_ok else "no"),
            "note": "" if g is None else ", ".join(
                n for n in (g.reason if not g.complete else "",
                            g.back_label if not g.back_ok else "") if n),
            "descent_starts_s": _num(r.descent_time),
            "bottom_s": _num(r.bottom_time),
            "push_up_starts_s": _num(r.push_time),
            "lockout_s": _num(r.lockout_time),
            "down_s": _num(r.eccentric_seconds),
            "pause_at_bottom_s": _num(r.pause_seconds),
            "push_up_s": _num(r.concentric_seconds),
            "bottom_to_lockout_s": _num(r.ascent_seconds),
            "bottom_shoulder_height_arm_lengths": _num(g and g.bottom_height),
            "top_shoulder_height_arm_lengths": _num(g and g.top_height),
            "bottom_elbow_deg": _num(g and g.bottom_elbow, 1),
            "top_elbow_deg": _num(g and g.top_elbow, 1),
            "body_line_worst_deg": _num(g and g.back_worst, 1),
        })
    return rows


def frame_rows(analysis: RepAnalysis) -> list[dict]:
    """One row per returned frame. `phase` is down, pause, push-up or blank (at the top)."""
    spans = []
    for r in analysis.reps:
        spans += [(r.number, "down", r.descent_frame, r.bottom_frame),
                  (r.number, "pause", r.bottom_frame, r.push_frame),
                  (r.number, "push-up", r.push_frame, r.lockout_frame)]

    def where(frame: int):
        # Half-open, so a boundary frame belongs to the phase it starts.
        for number, phase, a, b in spans:
            if a <= frame < b:
                return number, phase
        return "", ""

    rows = []
    for i, frame in enumerate(analysis.indices):
        rep, phase = where(int(frame))
        rows.append({
            "frame": int(frame),
            "time_s": _num(analysis.times[i], 4),
            "rep": rep,
            "phase": phase,
            "shoulder_height_arm_lengths": _num(analysis.height[i], 5),
            "shoulder_height_raw_arm_lengths": _num(analysis.raw[i], 5),
            "elbow_deg": _num(analysis.elbow[i], 2),
            "body_line_deg": _num(analysis.line[i], 2),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> Path | None:
    if not rows:
        return None
    # Union of keys, in first-seen order.
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write(analysis: RepAnalysis, run_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
    """Both CSVs into *run_dir*, named after the clip."""
    return (write_csv(summary_rows(analysis), run_dir / f"{stem}_summary.csv"),
            write_csv(frame_rows(analysis), run_dir / f"{stem}_frames.csv"))
