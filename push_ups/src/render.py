"""The export video (clip on top, panel below), the body-line overlay, and plots.png."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .panel import Panel, credit_lines
from .skeleton import KPT_NAMES, draw_person, visible_parts
from .video import VideoInfo

# Verdict colors, shared by every axis in plots.png so green means the same everywhere.
GOOD_C = "#2f9e44"
BAD_C = "#e03131"
GOOD_BGR = (102, 207, 81)    # #51cf66, the panel's FULL
BAD_BGR = (107, 107, 255)    # #ff6b6b, the panel's PARTIAL

_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"),
    TimeElapsedColumn(),
)


def _point(kpts, name: str, w: int, h: int):
    j = _INDEX[name]
    if j >= len(kpts) or tuple(kpts[j]) == (0, 0):
        return None
    return np.array(kpts[j], float) * [w, h]


def _draw_body_line(frame, person, *, side: str, straight: bool, thick: int) -> None:
    """Shoulder to hip to ankle, green while straight and red once it bends past tolerance,
    over a dashed straight shoulder-to-ankle reference.

    The gap between the two is how far the hips are off straight, as the panel reports.
    """
    h, w = frame.shape[:2]
    kpts = person.get("kpts_xy") or []
    s, hip, a = (_point(kpts, f"{side}_{j}", w, h) for j in ("shoulder", "hip", "ankle"))
    if s is None or hip is None or a is None:
        return
    color = GOOD_BGR if straight else BAD_BGR
    p = lambda v: (int(v[0]), int(v[1]))  # noqa: E731
    n = 24
    dash = max(2, thick * 2 // 3)
    for k in range(0, n, 2):
        u, v = s + (a - s) * k / n, s + (a - s) * (k + 1) / n
        # A dark edge under the white, so it reads on a light wall as well as a dark floor.
        cv2.line(frame, p(u), p(v), (30, 30, 30), dash + 3, cv2.LINE_AA)
        cv2.line(frame, p(u), p(v), (245, 245, 245), dash, cv2.LINE_AA)
    cv2.line(frame, p(s), p(hip), color, thick, cv2.LINE_AA)
    cv2.line(frame, p(hip), p(a), color, thick, cv2.LINE_AA)


def _draw_near_side(frame, person, *, side: str, straight: bool, thick: int, radius: int,
                    arm_color) -> None:
    """Only what is measured: the near arm, and the body line in place of torso and leg.

    The far side is hidden behind the body and ViTPose guesses it, and nothing reads it.
    """
    h, w = frame.shape[:2]
    kpts = person.get("kpts_xy") or []
    pts = {j: _point(kpts, f"{side}_{j}", w, h)
           for j in ("shoulder", "elbow", "wrist", "hip", "ankle")}
    p = lambda v: (int(v[0]), int(v[1]))  # noqa: E731
    _draw_body_line(frame, person, side=side, straight=straight, thick=thick)
    for u, v in (("shoulder", "elbow"), ("elbow", "wrist")):
        if pts[u] is not None and pts[v] is not None:
            cv2.line(frame, p(pts[u]), p(pts[v]), arm_color, thick, cv2.LINE_AA)
    # White joints with a dark ring, as the full skeleton draws them.
    for v in pts.values():
        if v is not None:
            cv2.circle(frame, p(v), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p(v), radius + 1, (20, 22, 26), 1, cv2.LINE_AA)


def _person_at(by_index: dict, i: int, hold: int):
    """This frame's athlete, or the last one seen within *hold* frames."""
    for j in range(i, max(-1, i - hold - 1), -1):
        persons = by_index.get(j)
        if persons:
            return persons[0]
    return None


def render(info: VideoInfo, *, by_index, analysis, n_frames: int, fps: float, dst: Path,
           cfg, console) -> dict:
    """Write the overlaid clip with its panel beneath it, at the same width.

    *n_frames* and *fps* are the inference clip's; frames are matched by index.
    """
    if cfg.EXPORT_SQUARE and cfg.EXPORT_HEIGHT:
        panel_h = info.width - info.height   # whatever the clip leaves of the square
    else:
        panel_h = int(round(info.height * cfg.PANEL_HEIGHT / 2)) * 2   # even, for H.264
    # A credit gets a strip of its own under the graphs, so they keep their size.
    footer = int(round(cfg.CREDIT_FOOTER * info.width / 1280 / 2)) * 2 if credit_lines(cfg) else 0
    if not (cfg.EXPORT_SQUARE and cfg.EXPORT_HEIGHT):
        panel_h += footer
    panel = Panel(analysis, n_frames, fps, info.width, panel_h, cfg=cfg, footer=footer)
    out_h = info.height + panel_h

    edges, points = visible_parts(cfg.DRAW_FACE)
    scale = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * scale))
    radius = max(1, round(cfg.POINT_RADIUS * scale))

    # The body line's verdict per frame, from the same series the panel draws.
    line = analysis.on_source_frames(analysis.line, n_frames)
    tol = cfg.BACK_TOLERANCE_DEGREES

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
                             (info.width, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    limit = min(info.n_frames, n_frames)
    written = posed = 0
    try:
        with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
            task = progress.add_task("render", total=limit)
            for i in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]

                person = _person_at(by_index, i, hold=2)
                if person is not None:
                    straight = not (np.isfinite(line[i]) and abs(line[i]) > tol)
                    if cfg.OVERLAY == "full":
                        draw_person(frame, person, w, h, edges=edges, points=points,
                                    thickness=thickness, radius=radius,
                                    draw_bbox=cfg.DRAW_BBOX, bbox_color=cfg.BBOX_COLOR,
                                    colors=cfg.SKELETON_COLORS)
                        _draw_body_line(frame, person, side=analysis.side, straight=straight,
                                        thick=max(2, thickness))
                    else:
                        _draw_near_side(frame, person, side=analysis.side, straight=straight,
                                        thick=max(2, thickness), radius=radius,
                                        arm_color=cfg.SKELETON_COLORS[f"{analysis.side}_arm"])
                    posed += 1

                writer.write(np.vstack([frame, panel.render(i)]))
                written += 1
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()
    return {"frames_written": written, "frames_with_pose": posed,
            "output": f"{info.width}x{out_h}"}


def plot_timeline(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Shoulder height with every rep's phases and its down and up cutoffs, the elbow
    angle on the same phases for comparison, the body line with its band, and the
    concentric time of every rep in order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .reps import METRICS, rep_value

    metric = cfg.REP_METRIC
    t = analysis.times
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.5, 1.2, 1.4], hspace=0.45)
    ax = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1], sharex=ax)
    ax3 = fig.add_subplot(grid[2], sharex=ax)
    ax4 = fig.add_subplot(grid[3])

    def at(time: float) -> int:
        return int(np.argmin(np.abs(t - time)))

    def legend(axis):
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=8.5,
                    frameon=False, borderpad=0, borderaxespad=0.15)

    # Shoulder height: the rep signal, each phase shaded, each rep's ends graded.
    up = cfg.ROM_UP_FRACTION * analysis.top
    ax.plot(t, analysis.raw, color="#c9ced6", lw=1, label="raw")
    ax.plot(t, analysis.height, color="#2f6df6", lw=2, label="smoothed")
    ax.axhline(up, color="#1864ab", ls="--", lw=1.1,
               label=f"back up (\u2265 {up:.2f}, {cfg.ROM_UP_FRACTION:.0%} of the plank)")
    ax.axhline(cfg.ROM_DOWN_HEIGHT, color="#c92a2a", ls="--", lw=1.1,
               label=f"down to the floor (\u2264 {cfg.ROM_DOWN_HEIGHT:.2f})")
    for i, rep in enumerate(analysis.reps):
        ax.axvspan(rep.descent_time, rep.bottom_time, color="#e8590c", alpha=0.10, lw=0,
                   label="eccentric" if i == 0 else None)
        ax.axvspan(rep.push_time, rep.lockout_time, color="#2f6df6", alpha=0.18, lw=0,
                   label="push (timed)" if i == 0 else None)
        g = rep.grade
        if g is not None:
            ax.plot(rep.bottom_time, g.bottom_height, "o", ms=6, zorder=4,
                    color=GOOD_C if g.bottom_ok else BAD_C)
            ax.plot(rep.lockout_time, g.top_height, "o", ms=6, zorder=4,
                    color=GOOD_C if g.top_ok else BAD_C)
        ax.annotate(str(rep.number), (rep.bottom_time, analysis.height[at(rep.bottom_time)]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9,
                    fontweight="bold", color=GOOD_C if rep.complete else BAD_C)
    lo = float(np.nanmin(analysis.height)) if len(analysis.height) else 0.0
    ax.set_ylim(bottom=lo - 0.15)   # room for the rep numbers under each bottom
    ax.set_ylabel("shoulder above wrist\n(arm lengths)")
    ax.set_title(title or f"{analysis.count} reps", fontsize=13, fontweight="bold", pad=24)
    ax.grid(alpha=0.25)
    legend(ax)

    # Elbow angle, on the same phases, to compare against the shoulder height above.
    # Reported only: the grade reads the shoulder. Its dots are each rep's most bent
    # and straightest, so their timing can be set against the two dots above.
    ax2.plot(t, analysis.elbow, color="#5f3dc4", lw=1.4, label=f"{analysis.side} elbow")
    ax2.axhline(90, color="#868e96", ls=":", lw=1.1, label="90\u00b0 (the usual elbow rule)")
    for rep in analysis.reps:
        ax2.axvspan(rep.descent_time, rep.bottom_time, color="#e8590c", alpha=0.10, lw=0)
        ax2.axvspan(rep.push_time, rep.lockout_time, color="#2f6df6", alpha=0.18, lw=0)
        g = rep.grade
        if g is None:
            continue
        # Each searched for in the window it was measured over, as in form.grade.
        reach = cfg.ROM_SAMPLE_FRAMES
        windows = ((g.bottom_elbow, slice(at(rep.descent_time), at(rep.lockout_time) + 1)),
                   (g.top_elbow, slice(at(rep.push_time), at(rep.lockout_time) + reach + 1)))
        for value, seg in windows:
            vals = analysis.elbow[seg]
            if np.isfinite(value) and np.isfinite(vals).any():
                j = int(np.nanargmin(np.abs(vals - value)))
                ax2.plot(t[seg][j], value, "o", ms=5, zorder=4, color="#5f3dc4")
    ax2.set_ylabel("elbow angle\n(degrees)")
    ax2.grid(alpha=0.25)
    legend(ax2)

    # Body line: each rep's worst reading dotted.
    tol = cfg.BACK_TOLERANCE_DEGREES
    ax3.axhspan(-tol, tol, color="#2b8a3e", alpha=0.10, lw=0, label=f"straight (\u00b1{tol:.0f}\u00b0)")
    ax3.plot(t, analysis.line, color="#495057", lw=1.2, label="hip off the straight shoulder\u2013ankle line (+ high, \u2212 low)")
    ax3.axhline(0, color="#adb5bd", ls=":", lw=1.0)
    for rep in analysis.reps:
        g = rep.grade
        if g is None or not np.isfinite(g.back_worst):
            continue
        tw = t[at(rep.descent_time):at(rep.lockout_time) + 1]
        seg = analysis.line[at(rep.descent_time):at(rep.lockout_time) + 1]
        if len(seg) and np.isfinite(seg).any():
            j = int(np.nanargmin(np.abs(seg - g.back_worst)))
            ax3.plot(tw[j], g.back_worst, "o", ms=6, zorder=4,
                     color=GOOD_C if g.back_ok else BAD_C)
    ax3.set_ylim(-2.2 * tol, 1.6 * tol)
    ax3.set_ylabel("body line vs.\nstraight line (\u00b0)")
    ax3.set_xlabel("time (s)")
    ax3.grid(alpha=0.25)
    legend(ax3)

    # Concentric time per rep, in order, so a slowing set reads left to right.
    values = [rep_value(r, metric) for r in analysis.reps]
    nums = [r.number for r in analysis.reps]
    if values:
        ax4.plot(nums, values, color="#2f6df6", lw=1.6, alpha=0.6, zorder=2)
        ax4.scatter(nums, values, s=60, zorder=3,
                    c=["#2f6df6" if r.straight else BAD_C for r in analysis.reps],
                    edgecolors=["white" if r.complete else "black" for r in analysis.reps])
        mean = float(np.mean(values))
        ax4.axhline(mean, color="#495057", ls="--", lw=1.1, label=f"mean {mean:.2f}s")
        for x, v in zip(nums, values):
            ax4.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=7.5, color="#495057")
        span = max(max(values) - min(values), 0.3)
        ax4.set_ylim(max(0, min(values) - span * 0.5), max(values) + span * 0.5)
        ax4.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax4.set_xticks(nums)
    ax4.set_xlabel("rep  (red: the back left the straight band)")
    ax4.set_ylabel(f"push-up time (s)\n{METRICS[metric][2]}")
    ax4.grid(alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.06)
    fig.savefig(dst, dpi=150)
    plt.close(fig)
