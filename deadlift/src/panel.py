"""The side panel: live back verdict, bar height and P(rounded) over time, and a row per rep.

Every answer is back before the first frame is written, so everything that does
not move is drawn once: both graphs' axes, the column headings and the credit are
one image, reused for every frame. Per frame only the moving parts are drawn on a
copy. The axes are fixed to the whole clip up front, so the curves fill in from
left to right and nothing rescales under them.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .text import resolve_font

# One palette, shared with the other vision-demos panels.
BG = "#14161a"
FG = "#f2f4f7"
MUTED = "#8b9199"
GRID = "#2b3038"
TRACK = "#23272e"
ACCENT = "#28a0ff"
POSITIVE = "#ff6b6b"   # rounded
NEGATIVE = "#51cf66"   # straight


def _rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)``."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _bgr(color: str) -> tuple[int, int, int]:
    return _rgb(color)[::-1]


@lru_cache(maxsize=32)
def _face(path: str, index: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, px, index=index)


def _text(img: np.ndarray, items, font) -> np.ndarray:
    """Draw ``(text, (x, y), px, rgb, anchor)`` items; ``anchor`` is PIL's (e.g. "lm", "rm")."""
    if not items:
        return img
    if font is None:
        for text, (x, y), px, rgb, anchor in items:
            fs = px / 34
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            x -= {"l": 0, "m": tw // 2, "r": tw}[anchor[0]]
            y += {"t": th, "m": th // 2, "b": 0, "s": 0}[anchor[1]]
            cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        rgb[::-1], 2, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    for text, xy, px, rgb, anchor in items:
        draw.text(xy, text, font=_face(font[0], font[1], px), fill=rgb, anchor=anchor)
    img[:] = np.array(pil)[..., ::-1]
    return img


def _text_width(text: str, px: int, font) -> int:
    """Rendered width of *text* at *px*, in the same face `_text` draws with."""
    if font is None:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 2)[0][0]
    return int(_face(font[0], font[1], px).getlength(text))


def _axes_image(width: int, height: int, *, duration: float, ylim, yticks, yfmt,
                title: str, xlabel: str, ylabel: str, scale: float, font, margin: int,
                threshold=None):
    """Static axes as BGR, plus the axes' pixel box ``(x0, y0, x1, y1)``, top-down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fp = None
    if font is not None:
        try:
            from matplotlib import font_manager
            fp = font_manager.FontProperties(fname=font[0])
        except Exception:
            fp = None

    def f(size: float) -> dict:
        # A FontProperties carries its own size, applied after `fontsize=`; a
        # sized copy is the only way both survive.
        if fp is None:
            return {"fontsize": size}
        sized = fp.copy()
        sized.set_size(size)
        return {"fontproperties": sized}

    dpi = max(30.0, 100.0 * scale)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_subplot(111, facecolor=BG)
    ax.set_xlim(0, duration)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

    if threshold is not None:
        ax.axhline(threshold, color=MUTED, ls="--", lw=1.4, alpha=0.8)

    ax.set_title(title, color=FG, pad=12, **f(21))
    ax.set_xlabel(xlabel, color=MUTED, labelpad=4, **f(15))
    ax.set_ylabel(ylabel, color=MUTED, labelpad=6, **f(15))
    ax.tick_params(colors=MUTED, labelsize=13)
    if fp:
        ticks = fp.copy()
        ticks.set_size(13)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(ticks)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    # The right edge meets the panel margin, so the axes end where the text does.
    fig.subplots_adjust(left=0.17, right=1 - margin / width, top=0.87, bottom=0.18)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    fig_h = fig.canvas.get_width_height()[1]
    ext = ax.get_window_extent(renderer)
    box = (int(ext.x0), int(fig_h - ext.y1), int(ext.x1), int(fig_h - ext.y0))

    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    bgr = rgba[..., 2::-1].copy()
    if bgr.shape[:2] != (height, width):
        rx, ry = width / bgr.shape[1], height / bgr.shape[0]
        bgr = cv2.resize(bgr, (width, height))
        box = (int(box[0] * rx), int(box[1] * ry), int(box[2] * rx), int(box[3] * ry))
    return bgr, box


def display_smooth(values: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """A Gaussian-smoothed copy for drawing only; *seconds* is the kernel's sigma.

    Centered, so it does not lag the video: the whole clip is known before the
    first frame is drawn. Edge-padded, so the first and last frames are not
    pulled toward zero.
    """
    sigma = seconds * fps
    if sigma < 0.5 or len(values) < 3:
        return values.copy()
    half = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, half, mode="edge"), kernel, mode="valid")


class _Strip:
    """One time-series graph: its axes box and every frame's point in panel pixels."""

    def __init__(self, base: np.ndarray, top: int, height: int, values: np.ndarray, *,
                 ylim, scale: float, font, margin: int, **axes_kw):
        width = base.shape[1]
        img, (x0, y0, x1, y1) = _axes_image(width, height, ylim=ylim, scale=scale, font=font,
                                            margin=margin, **axes_kw)
        base[top:top + height] = img
        self.box = (x0, top + y0, x1, top + y1)
        x0, y0, x1, y1 = self.box
        n = len(values)
        lo, hi = ylim
        self.lo, self.hi = lo, hi
        self.px = (x0 + np.arange(n) / max(n, 1) * (x1 - x0)).astype(np.int32)
        clipped = np.clip(values, lo, hi)
        self.py = (y1 - (clipped - lo) / (hi - lo) * (y1 - y0)).astype(np.int32)

    def y_of(self, value: float) -> int:
        _, y0, _, y1 = self.box
        return int(y1 - (value - self.lo) / (self.hi - self.lo) * (y1 - y0))

    def shade(self, panel, a: int, b: int, color: str, alpha: float = 0.18) -> None:
        """Tint the frames ``[a, b]`` across the full height of the axes."""
        _, y0, _, y1 = self.box
        if b <= a:
            return
        xa = int(self.px[min(a, len(self.px) - 1)])
        xb = int(self.px[min(b, len(self.px) - 1)])
        roi = panel[y0:y1, xa:xb + 1]
        roi[:] = (roi * (1 - alpha) + np.array(_bgr(color)) * alpha).astype(np.uint8)

    def line(self, panel, k: int, colors, thick: int) -> None:
        """The curve up to frame *k*, one color per frame, drawn as same-color runs."""
        k = min(k, len(self.px))
        if k < 2:
            return
        xs, ys = self.px[:k], self.py[:k]
        start = 0
        for i in range(1, k + 1):
            if i == k or colors[i] != colors[start]:
                seg = np.stack([xs[max(0, start - 1):i], ys[max(0, start - 1):i]], 1)
                cv2.polylines(panel, [seg.reshape(-1, 1, 2)], False, _bgr(colors[start]),
                              thick, cv2.LINE_AA)
                start = i

    def playhead(self, panel, k: int, color: str, scale: float) -> None:
        _, y0, _, y1 = self.box
        i = min(max(k - 1, 0), len(self.px) - 1)
        px, py = int(self.px[i]), int(self.py[i])
        cv2.line(panel, (px, y0), (px, y1), _bgr(MUTED), max(1, int(1.5 * scale)), cv2.LINE_AA)
        cv2.circle(panel, (px, py), int(9 * scale), _bgr(BG), -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), int(7 * scale), _bgr(color), -1, cv2.LINE_AA)


class Panel:
    """Builds the panel for any frame from one static base."""

    def __init__(self, signal, analysis, width: int, height: int, *, cfg):
        self.signal, self.analysis, self.cfg = signal, analysis, cfg
        self.w, self.h = width, height
        s = self.s = width / 720
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        self.positive = cfg.POSITIVE_LABEL
        self.negative = next(k for k in cfg.LABELS if k != cfg.POSITIVE_LABEL)
        self.margin = int(48 * s)

        # Vertical rhythm, top-down, as fractions of the panel height. The rep
        # rows share out whatever is left above the credit, so a short set is
        # spaced out rather than leaving a gap, and a long one packs tighter.
        self.y_reps = int(height * 0.045)
        self.y_back = int(height * 0.098)
        self.y_rows = int(height * 0.765)
        rows_bottom = int(height * 0.915)
        n_rows = max(len(analysis.reps), 3)
        span = max(0, rows_bottom - self.y_rows)
        self.row_gap = max(1, min(int(height * 0.06), span // n_rows))
        self.room = max(1, span // self.row_gap)

        base = np.zeros((height, width, 3), np.uint8)
        base[:] = _bgr(BG)
        duration = len(signal.per_frame) / signal.fps
        # What the panel draws. Display only: rep verdicts average the
        # underlying signal, so smoothing here changes no result.
        self.p_display = display_smooth(signal.per_frame, signal.fps,
                                        cfg.PANEL_P_SMOOTH_SECONDS)

        # The rep signal's graph: bar height in meters (computed in cm off the
        # plate ruler), or the hip angle when the reps were counted off the hinge.
        strip_kw = dict(duration=duration, xlabel="Time (s)", scale=s, font=self.font,
                        margin=self.margin)
        if analysis.source == "plate":
            values = analysis.height_cm / 100
            top = max(0.1, float(np.nanmax(values)) * 1.15)
            step = 0.1 if top <= 0.6 else 0.2
            self.height_strip = _Strip(
                base, int(height * 0.135), int(height * 0.30), values,
                ylim=(min(-0.03, float(np.nanmin(values))), top),
                yticks=np.arange(0, top, step), yfmt=lambda v, _: f"{v:.1f}",
                title="Bar height", ylabel="Height (m)", **strip_kw)
        else:
            hip = analysis.hip_deg
            fill = float(np.nanmin(hip)) if np.isfinite(hip).any() else 90.0
            values = np.nan_to_num(hip, nan=fill)
            self.height_strip = _Strip(
                base, int(height * 0.135), int(height * 0.30), values,
                ylim=(max(0.0, float(values.min()) - 10), min(190.0, float(values.max()) + 10)),
                yticks=[45, 90, 135, 180], yfmt=lambda v, _: f"{v:.0f}°",
                title="Hip angle", ylabel="Angle (°)", **strip_kw)
        self.p_strip = _Strip(
            base, int(height * 0.44), int(height * 0.30), self.p_display,
            ylim=(0.0, 1.0), yticks=[0, 0.25, 0.5, 0.75, 1.0], yfmt=lambda v, _: f"{v:.0%}",
            duration=duration, title=f"P({self.positive}) over time", xlabel="Time (s)",
            ylabel=f"P({self.positive})", scale=s, font=self.font,
            margin=self.margin, threshold=cfg.THRESHOLD)

        # Rep table columns: x positions shared by the heading and every row.
        content = width - 2 * self.margin
        # REP | BACK | P(ROUNDED) | LIFT, centered at equal intervals: the verdict
        # beside its rep, and the probability under a heading that says what it
        # is the probability of.
        self.cols = tuple(int(self.margin + content * f) for f in (1 / 8, 3 / 8, 5 / 8, 7 / 8))

        # The verdict line holds a label on the left and a percentage on the
        # right. Sized to the longest either can get ("Back: STRAIGHT",
        # "P(rounded) 100%"), shrinking if needed so the two never meet.
        longest_label = max((f"Back: {k.upper()}" for k in cfg.LABELS), key=len)
        longest_value = f"P({self.positive}) 100%"
        px = int(cfg.PANEL_VERDICT_SIZE * s)
        while px > 10 and (_text_width(longest_label, px, self.font)
                           + _text_width(longest_value, int(px * 0.8), self.font)
                           + int(40 * s)) > content:
            px = int(px * 0.95)
        self.verdict_px = px
        muted, note_px = _rgb(MUTED), int(cfg.PANEL_NOTE_SIZE * s)
        y = self.y_rows
        items = [("REP", (self.cols[0], y), note_px, muted, "mm"),
                 ("BACK", (self.cols[1], y), note_px, muted, "mm"),
                 (f"P({self.positive.upper()})", (self.cols[2], y), note_px, muted, "mm"),
                 ("LIFT", (self.cols[3], y), note_px, muted, "mm")]
        _text(base, items, self.font)
        rule_y = self.y_rows + int(18 * s)
        cv2.line(base, (self.margin, rule_y), (width - self.margin, rule_y), _bgr(GRID),
                 max(1, int(s)))
        _draw_credit(base, self.font, cfg=cfg, scale=s)
        self.base = base

        above = self.p_display > cfg.THRESHOLD
        self.p_colors = [POSITIVE if a else NEGATIVE for a in above]
        self.h_colors = [ACCENT] * len(values)

    def _verdict_color(self, rep) -> str:
        return POSITIVE if rep.back_label == self.positive else NEGATIVE

    def p_at(self, frame: int) -> float:
        """The displayed P(rounded) at *frame*: the same value the graph and label show."""
        return float(self.p_display[min(frame, len(self.p_display) - 1)])

    def render(self, frame: int) -> np.ndarray:
        s, cfg = self.s, self.cfg
        panel = self.base.copy()
        p = self.p_at(frame)
        color = POSITIVE if p > cfg.THRESHOLD else NEGATIVE
        done = self.analysis.done_at(frame)
        k = frame + 1
        thick = max(2, int(3 * s))

        # Bar height: finished reps shaded in their verdict, numbered at lockout.
        items = []
        for rep in done:
            c = self._verdict_color(rep)
            self.height_strip.shade(panel, rep.onset_frame, min(rep.end_frame, frame), c)
            px = int(self.height_strip.px[rep.peak_frame])
            py = int(self.height_strip.py[rep.peak_frame]) - int(14 * s)
            items.append((str(rep.number), (px, py), int(18 * s), _rgb(c), "mb"))
        self.height_strip.line(panel, k, self.h_colors, thick)
        self.height_strip.playhead(panel, k, ACCENT, s)

        self._p_fill(panel, k)
        self.p_strip.line(panel, k, self.p_colors, thick)
        self.p_strip.playhead(panel, k, color, s)

        # Header: reps so far, and the live verdict.
        x1 = self.w - self.margin
        lifts = [r.lift_seconds for r in done]
        head_px = int(cfg.PANEL_HEADER_SIZE * s)
        items += [
            (f"Reps: {len(done)}", (self.margin, self.y_reps), head_px, _rgb(FG), "lm"),
            (f"Avg lift: {np.mean(lifts):.2f}s" if lifts else "Avg lift: —",
             (x1, self.y_reps), int(head_px * 0.7), _rgb(MUTED), "rm"),
            (f"Back: {(self.positive if p > cfg.THRESHOLD else self.negative).upper()}",
             (self.margin, self.y_back), self.verdict_px, _rgb(color), "lm"),
            (f"P({self.positive}) {p:.0%}", (x1, self.y_back), int(self.verdict_px * 0.8),
             _rgb(FG), "rm"),
        ]

        # One row per rep, once it has reached lockout. The newest rows win if a
        # long set runs out of room.
        row_px = int(cfg.PANEL_ROW_SIZE * s)
        room = self.room
        for j, rep in enumerate(done[-room:]):
            y = self.y_rows + (j + 1) * self.row_gap
            c = self._verdict_color(rep)
            items += [
                (f"{rep.number}", (self.cols[0], y), row_px, _rgb(FG), "mm"),
                (rep.back_label.upper(), (self.cols[1], y), row_px, _rgb(c), "mm"),
                (f"{rep.back_p:.0%}", (self.cols[2], y), row_px, _rgb(FG), "mm"),
                (f"{rep.lift_seconds:.2f}s", (self.cols[3], y), row_px, _rgb(FG), "mm"),
            ]
        _text(panel, items, self.font)
        return panel

    def _p_fill(self, panel: np.ndarray, k: int) -> None:
        """Area between P(rounded) and the threshold: red above it, green below.

        Drawn into a mask, then colored by which side of the line a row sits, so
        a crossing splits cleanly without solving for the intersection.
        """
        strip = self.p_strip
        x0, y0, x1, y1 = strip.box
        k = min(k, len(strip.px))
        if k < 2:
            return
        thr_y = strip.y_of(self.cfg.THRESHOLD)
        xs, ys = strip.px[:k], strip.py[:k]
        roi = panel[y0:y1 + 1, x0:x1 + 1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        poly = np.concatenate([
            np.stack([xs - x0, ys - y0], 1),
            [[xs[-1] - x0, thr_y - y0], [xs[0] - x0, thr_y - y0]],
        ]).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        tint = np.empty_like(roi)
        tint[: thr_y - y0] = _bgr(POSITIVE)
        tint[thr_y - y0:] = _bgr(NEGATIVE)
        on = mask > 0
        roi[on] = (roi[on] * 0.72 + tint[on] * 0.28).astype(np.uint8)


def credit_lines(cfg) -> list[tuple[str, str]]:
    """The credit as ``[(label, value), ...]``, top line first. Empty for none.

    Each ``CREDIT`` entry is a ``(label, value)`` pair or a plain string, as in
    rock_climbing_3d. Blank entries are skipped, so an unfilled handle leaves no gap.
    """
    lines = []
    for entry in getattr(cfg, "CREDIT", None) or []:
        if isinstance(entry, (tuple, list)):
            label, value = (entry[0], entry[1]) if len(entry) >= 2 else ("", entry[0])
        else:
            label, value = "", entry
        label, value = str(label or "").strip(), str(value or "").strip()
        if value:
            lines.append((label, value))
    return lines


def _draw_credit(panel, font, *, cfg, scale: float) -> None:
    """The credit stack in the bottom-right corner, last line on the margin.

    Each line is ``LABEL: value``, right-aligned, the label a shade dimmer. Both
    parts share one baseline, and the stack grows upward from the corner.
    """
    lines = credit_lines(cfg)
    if not lines:
        return
    h, w = panel.shape[:2]
    px = max(9, int(cfg.CREDIT_SIZE * scale))
    gap = int(cfg.CREDIT_LINE_GAP * scale)
    right = w - int(cfg.CREDIT_MARGIN * scale)
    y = h - int(cfg.CREDIT_MARGIN * scale)
    value_rgb, label_rgb = cfg.CREDIT_COLOR[::-1], cfg.CREDIT_LABEL_COLOR[::-1]

    if font is None:
        for label, value in reversed(lines):
            text = f"{label}: {value}" if label else value
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
            cv2.putText(panel, text, (right - tw, y), cv2.FONT_HERSHEY_SIMPLEX, px / 34,
                        cfg.CREDIT_COLOR, 1, cv2.LINE_AA)
            y -= th + gap
        return

    from PIL import Image, ImageDraw

    pil = Image.fromarray(panel[..., ::-1])
    draw = ImageDraw.Draw(pil)
    face = _face(font[0], font[1], px)
    for label, value in reversed(lines):
        # Baseline anchors ("rs"), so the label and value sit on one line.
        draw.text((right, y), value, font=face, fill=value_rgb, anchor="rs")
        if label:
            vw = draw.textlength(value, font=face)
            draw.text((right - vw, y), f"{label}: ", font=face, fill=label_rgb, anchor="rs")
        y -= px + gap
    panel[:] = np.array(pil)[..., ::-1]
