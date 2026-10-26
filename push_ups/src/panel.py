"""The bottom panel: live tallies, vertical displacement and body line over time,
and the concentric time of every rep.

The clip is landscape, so the panel sits under it at the same width instead of
beside it. All reps are known up front, so the static parts (axes, headings,
credit) are drawn once and copied per frame; fixed axes mean the curves and dots
fill in without rescaling.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .reps import rep_value
from .text import resolve_font

# One palette, shared with the other vision-demos panels.
BG = "#14161a"
FG = "#f2f4f7"
MUTED = "#8b9199"
GRID = "#2b3038"
FULL = "#51cf66"      # full range / straight
PARTIAL = "#ff6b6b"   # cut short / hips off straight
ACCENT = "#28a0ff"    # the displacement curve, the concentric and its dots


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


def _axes_image(width: int, height: int, *, xlim, ylim, yticks, yfmt, title: str,
                scale: float, font, pad, xlabel: str = "", xticks=None, xfmt=None,
                show_x: bool = True, hlines=(), band=None, note: str = ""):
    """Static axes as BGR, plus the axes' pixel box ``(x0, y0, x1, y1)``, top-down.

    *pad* is ``(left, right, top, bottom)`` in px, so neighbouring graphs can share
    one left edge whatever their tick labels are.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator

    fp = None
    if font is not None:
        try:
            from matplotlib import font_manager
            fp = font_manager.FontProperties(fname=font[0])
        except Exception:
            fp = None

    def f(size: float) -> dict:
        # A FontProperties overrides `fontsize=`, so pass a sized copy instead.
        if fp is None:
            return {"fontsize": size}
        sized = fp.copy()
        sized.set_size(size)
        return {"fontproperties": sized}

    dpi = max(30.0, 100.0 * scale)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BG)
    left, right, top, bottom = pad
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width,
                       1 - (top + bottom) / height], facecolor=BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    if xticks is not None:
        ax.xaxis.set_major_locator(xticks)
    else:
        span = xlim[1] - xlim[0]
        ax.xaxis.set_major_locator(MultipleLocator(10 if span > 40 else 5 if span > 12 else 1))
    ax.xaxis.set_major_formatter(FuncFormatter(xfmt or (lambda v, _: f"{v:.0f}")))
    if not show_x:
        ax.tick_params(labelbottom=False)

    if band is not None:
        ax.axhspan(*band, color=FULL, alpha=0.08, lw=0)
    for y in hlines:
        ax.axhline(y, color=MUTED, ls="--", lw=1.2, alpha=0.8)

    ax.set_title(title, color=FG, loc="left", pad=6, **f(15))
    if note:
        # Right of the title, on its line: a label beside a line would sit on the data.
        ax.annotate(note, (1, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", ha="right", va="bottom", color=MUTED, **f(11))
    if xlabel and show_x:
        ax.set_xlabel(xlabel, color=MUTED, labelpad=2, **f(11))
    ax.tick_params(colors=MUTED, labelsize=10, length=3)
    if fp:
        ticks = fp.copy()
        ticks.set_size(10)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(ticks)
    for side, spine in ax.spines.items():
        spine.set_color(GRID)
        spine.set_visible(side in ("left", "bottom"))
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


def display_smooth(values: np.ndarray, fps: float, seconds: float, *,
                   bridge_seconds: float = 0.1) -> np.ndarray:
    """Centered Gaussian smoothing for display; *seconds* is the sigma.

    NaN gaps up to *bridge_seconds* are bridged; longer ones stay gaps.
    """
    out = values.astype(float).copy()
    finite = np.isfinite(out)
    if finite.sum() < 2:
        return out
    idx = np.arange(len(out))
    filled = np.interp(idx, idx[finite], out[finite])

    keep = finite.copy()
    limit = max(1, int(round(bridge_seconds * fps)))
    i = 0
    while i < len(out):
        if finite[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not finite[j]:
            j += 1
        if i > 0 and j < len(out) and j - i <= limit:
            keep[i:j] = True
        i = j

    sigma = seconds * fps
    if sigma >= 0.5 and len(filled) >= 3:
        half = int(3 * sigma)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        filled = np.convolve(np.pad(filled, half, mode="edge"), kernel, mode="valid")
    return np.where(keep, filled, np.nan)


class _Graph:
    """One graph placed on the panel: its axes box and a mapping from data to pixels."""

    def __init__(self, base: np.ndarray, rect, *, xlim, ylim, scale: float, font, **axes_kw):
        x, y, w, h = rect
        img, (x0, y0, x1, y1) = _axes_image(w, h, xlim=xlim, ylim=ylim, scale=scale,
                                            font=font, **axes_kw)
        base[y:y + h, x:x + w] = img
        self.box = (x + x0, y + y0, x + x1, y + y1)
        self.xlim, self.ylim = xlim, ylim

    def x_of(self, value):
        x0, _, x1, _ = self.box
        lo, hi = self.xlim
        return x0 + (np.asarray(value, float) - lo) / (hi - lo) * (x1 - x0)

    def y_of(self, value):
        """Panel row of *value*, clipped to the axes; NaN stays NaN."""
        _, y0, _, y1 = self.box
        lo, hi = self.ylim
        clipped = np.clip(value, min(lo, hi), max(lo, hi))
        return y1 - (clipped - lo) / (hi - lo) * (y1 - y0)


class _Timeline(_Graph):
    """A graph whose x axis is the clip's time, one sample per source frame."""

    def __init__(self, base, rect, n: int, fps: float, **kw):
        super().__init__(base, rect, xlim=(0, n / fps), **kw)
        self.px = self.x_of(np.arange(n) / fps).astype(np.int32)

    def shade(self, panel, a: int, b: int, color: str, alpha: float = 0.18) -> None:
        """Tint the frames ``[a, b]`` across the full height of the axes."""
        _, y0, _, y1 = self.box
        if b <= a:
            return
        xa = int(self.px[min(a, len(self.px) - 1)])
        xb = int(self.px[min(b, len(self.px) - 1)])
        roi = panel[y0:y1, xa:xb + 1]
        roi[:] = (roi * (1 - alpha) + np.array(_bgr(color)) * alpha).astype(np.uint8)

    def line(self, panel, py: np.ndarray, k: int, color: str, thick: int) -> None:
        """The curve up to frame *k*, broken wherever the value is NaN."""
        k = min(k, len(self.px))
        if k < 2:
            return
        ys = py[:k]
        ok = np.isfinite(ys)
        runs, start = [], None
        for i in range(k + 1):
            if i < k and ok[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= 2:
                    runs.append(np.stack([self.px[start:i], ys[start:i].astype(np.int32)], 1)
                                .reshape(-1, 1, 2))
                start = None
        if runs:
            cv2.polylines(panel, runs, False, _bgr(color), thick, cv2.LINE_AA)

    def playhead(self, panel, k: int, scale: float) -> None:
        _, y0, _, y1 = self.box
        px = int(self.px[min(max(k - 1, 0), len(self.px) - 1)])
        cv2.line(panel, (px, y0), (px, y1), _bgr(MUTED), max(1, int(1.2 * scale)), cv2.LINE_AA)

    def dot(self, panel, frame: int, py: float, color: str, scale: float, r: float = 5) -> None:
        if not np.isfinite(py):
            return
        px = int(self.px[min(max(frame, 0), len(self.px) - 1)])
        cv2.circle(panel, (px, int(py)), int((r + 1.5) * scale), _bgr(BG), -1, cv2.LINE_AA)
        cv2.circle(panel, (px, int(py)), int(r * scale), _bgr(color), -1, cv2.LINE_AA)


class Panel:
    """Builds the panel for any frame from one static base."""

    def __init__(self, analysis, n_frames: int, fps: float, width: int, height: int, *, cfg,
                 footer: int = 0):
        self.analysis, self.cfg = analysis, cfg
        self.w, self.h = width, height
        full_h, height = height, height - footer   # the footer strip is the credit's alone
        s = self.s = width / 1280
        self.font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        self.margin = m = int(28 * s)
        self.metric = cfg.REP_METRIC

        def shown(series):
            return display_smooth(analysis.on_source_frames(series, n_frames), fps,
                                  cfg.PANEL_SMOOTH_SECONDS)

        self.height_v = shown(analysis.raw)
        self.line_deg = shown(analysis.line)

        base = np.zeros((full_h, width, 3), np.uint8)
        base[:] = _bgr(BG)

        # Header: four cells across, a small label over each value.
        self.y_label = int(height * 0.05)
        self.y_value = int(height * 0.115)
        self.cells = [int(m + (width - 2 * m) * (i + 0.5) / 4) for i in range(4)]
        note_px = int(cfg.PANEL_NOTE_SIZE * s)
        labels = ("PUSH-UPS", "AVG PUSH-UP TIME", "FULL RANGE", "STRAIGHT BACK")
        _text(base, [(lab, (x, self.y_label), note_px, _rgb(MUTED), "mm")
                     for lab, x in zip(labels, self.cells)], self.font)
        rule_y = int(height * 0.17)
        cv2.line(base, (m, rule_y), (width - m, rule_y), _bgr(GRID), max(1, int(s)))

        # Body: the two timelines stacked on the left, the concentric dots on the right.
        top = rule_y + int(6 * s)
        floor = height - int(8 * s)
        # Split so the two columns' plot areas come out the same width: the left one
        # gives more room to its tick labels ("hips high 15°") than the right ("1.0s").
        left_pad, right_pad, gap = int(112 * s), int(62 * s), int(16 * s)
        split = (width + left_pad + gap - right_pad - m) // 2
        # Split so both plot areas come out the same height; the lower one also
        # carries the time axis, so it gets that much more of the body.
        mid = (top + floor) // 2 - int(8 * s)
        common = dict(scale=s, font=self.font)

        # Shoulder above the wrist: the rep itself. The two lines are the cutoffs
        # themselves, by name, since arm lengths aren't a unit anyone reads: each
        # rep's dots land on the side of them its grade says.
        plank, down = analysis.top, cfg.ROM_DOWN_HEIGHT
        up = cfg.ROM_UP_FRACTION * plank
        marks = {up: "up", down: "floor"}
        seen = self.height_v[np.isfinite(self.height_v)]
        lo_h = min(float(seen.min()) if len(seen) else down, down)
        hi_h = max(float(seen.max()) if len(seen) else plank, up)
        pad_h = (hi_h - lo_h) * 0.07
        self.height_g = _Timeline(
            base, (0, top, split, mid - top), n_frames, fps,
            ylim=(lo_h - pad_h, hi_h + pad_h), yticks=sorted(marks),
            yfmt=lambda v, _: min(marks.items(), key=lambda kv: abs(kv[0] - v))[1],
            title="Vertical displacement", show_x=False, hlines=(up, down),
            pad=(left_pad, gap, int(26 * s), int(6 * s)), **common)
        self.height_py = self.height_g.y_of(self.height_v)

        # Fit to the set itself, so getting up off the floor afterwards doesn't
        # stretch the axis; it is clipped to the edge instead.
        tol = cfg.BACK_TOLERANCE_DEGREES
        reps_ = analysis.reps
        in_set = (self.line_deg[reps_[0].descent_frame:reps_[-1].lockout_frame + 1]
                  if reps_ else self.line_deg)
        in_set = in_set[np.isfinite(in_set)]
        lo_l = min(float(in_set.min()) if len(in_set) else -tol, -tol)
        hi_l = max(float(in_set.max()) if len(in_set) else tol, tol)
        pad_l = (hi_l - lo_l) * 0.08
        self.line_g = _Timeline(
            base, (0, mid, split, floor - mid), n_frames, fps,
            ylim=(lo_l - pad_l, hi_l + pad_l), yticks=[-tol, 0, tol],
            yfmt=lambda v, _: "straight" if abs(v) < 1e-6 else f"hips {'low' if v < 0 else 'high'} {abs(v):.0f}°",
            title="Body line vs. straight line", hlines=(0,),
            xfmt=lambda v, _: f"{v:.0f}s",
            band=(-tol, tol),
            pad=(left_pad, gap, int(26 * s), int(22 * s)), **common)
        self.line_py = self.line_g.y_of(self.line_deg)

        # Concentric time per rep, a dot each, fixed axes so nothing rescales.
        values = [rep_value(r, self.metric) for r in analysis.reps]
        n = max(len(values), 1)
        lo_v = min(values) if values else 0.0
        hi_v = max(values) if values else 1.0
        span = max(hi_v - lo_v, 0.1)
        ylim = (max(0.0, lo_v - span * 0.15), hi_v + span * 0.3)
        # About five ticks, on a round step.
        step = next(st for st in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0)
                    if (ylim[1] - ylim[0]) / st <= 6)
        from matplotlib.ticker import MaxNLocator
        self.dots = _Graph(
            base, (split, top, width - split, floor - top),
            xlim=(0.4, n + 0.6), ylim=ylim,
            yticks=list(np.arange(np.ceil(ylim[0] / step) * step, ylim[1], step)),
            yfmt=lambda v, _: f"{v:.2f}s" if step < 0.1 else f"{v:.1f}s",
            xticks=MaxNLocator(integer=True, nbins=12),
            title="Push-up time per rep",
            pad=(right_pad, m, int(26 * s), int(22 * s)), **common)

        _draw_credit(base, self.font, cfg=cfg, scale=s, footer=footer)
        self.base = base

    def render(self, frame: int) -> np.ndarray:
        s, cfg = self.s, self.cfg
        panel = self.base.copy()
        reps = self.analysis.reps
        done = self.analysis.done_at(frame)
        k = frame + 1
        thick = max(2, int(2.4 * s))
        items = []

        # Body line: red where it leaves the band, so the bend shows where it happens.
        tol = cfg.BACK_TOLERANCE_DEGREES
        bent = np.where(np.abs(self.line_deg) > tol, self.line_py, np.nan)
        self.line_g.line(panel, self.line_py, k, FG, max(1, thick - 1))
        self.line_g.line(panel, bent, k, PARTIAL, thick)
        self.height_g.line(panel, self.height_py, k, ACCENT, thick)
        for g in (self.height_g, self.line_g):
            g.playhead(panel, k, s)

        # Where each concentric is timed from and to: leaving the bottom, and lockout.
        # On the curve itself, green, or red when that end missed its cutoff, as in pull_ups.
        last = len(self.height_py) - 1
        for rep in reps:
            gr = rep.grade
            for f, ok in ((rep.push_frame, gr is None or gr.bottom_ok),
                          (rep.lockout_frame, gr is None or gr.top_ok)):
                if f <= frame:
                    self.height_g.dot(panel, f, self.height_py[min(f, last)],
                                      FULL if ok else PARTIAL, s, r=5)
        i = min(frame, len(self.height_py) - 1)
        self.height_g.dot(panel, i, self.height_py[i], ACCENT, s, r=6)
        self.line_g.dot(panel, i, self.line_py[i],
                        PARTIAL if np.isfinite(self.line_deg[i]) and abs(self.line_deg[i]) > tol
                        else FG, s, r=5)

        # One dot per rep as it locks out, joined in order; red if its back left the band.
        g = self.dots
        if done:
            pts = np.array([(g.x_of(r.number), g.y_of(rep_value(r, self.metric))) for r in done],
                           np.int32)
            if len(pts) > 1:
                cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, _bgr(ACCENT),
                              max(1, thick - 1), cv2.LINE_AA)
            avg = float(np.mean([rep_value(r, self.metric) for r in done]))
            ya = int(g.y_of(avg))
            x0, _, x1, _ = g.box
            for xa in range(x0, x1, int(12 * s)):
                cv2.line(panel, (xa, ya), (min(xa + int(6 * s), x1), ya), _bgr(MUTED),
                         max(1, int(s)), cv2.LINE_AA)
            r_dot = 7 * s
            for rep, (px, py) in zip(done, pts):
                color = ACCENT if rep.straight else PARTIAL
                cv2.circle(panel, (int(px), int(py)), int(r_dot + 2 * s), _bgr(BG), -1, cv2.LINE_AA)
                cv2.circle(panel, (int(px), int(py)), int(r_dot), _bgr(color), -1, cv2.LINE_AA)
                if not rep.complete:
                    cv2.circle(panel, (int(px), int(py)), int(r_dot + 2 * s), _bgr(FG),
                               max(1, int(s)), cv2.LINE_AA)
            last = done[-1]
            items.append((f"{rep_value(last, self.metric):.2f}s",
                          (int(pts[-1][0]), int(pts[-1][1] - r_dot - 8 * s)), int(14 * s),
                          _rgb(FG), "mb"))

        # Header values.
        graded = [r for r in done if r.grade is not None]
        full = sum(r.grade.complete for r in graded)
        straight = sum(r.grade.back_ok for r in graded)
        val_px = int(cfg.PANEL_LIVE_SIZE * s)
        conc = [rep_value(r, self.metric) for r in done]

        def tally(num):
            if not graded:
                return "—", _rgb(MUTED)
            return f"{num} / {len(graded)}", _rgb(FULL if num == len(graded) else PARTIAL)

        cells = [
            (str(len(done)), _rgb(FG), int(cfg.PANEL_HEADER_SIZE * s)),
            (f"{np.mean(conc):.2f}s" if conc else "—", _rgb(ACCENT if conc else MUTED), val_px),
            (*tally(full), val_px),
            (*tally(straight), val_px),
        ]
        items += [(text, (x, self.y_value), px, rgb, "mm")
                  for (text, rgb, px), x in zip(cells, self.cells)]
        _text(panel, items, self.font)
        return panel


def credit_lines(cfg) -> list[tuple[str, str]]:
    """``cfg.CREDIT`` as ``[(label, value), ...]``, top line first; blank entries skipped.

    Each entry is a ``(label, value)`` pair or a plain string.
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


def _draw_credit(panel, font, *, cfg, scale: float, footer: int) -> None:
    """The credit on one line, right-aligned in the footer strip under the graphs."""
    lines = credit_lines(cfg)
    if not lines or not footer:
        return
    h, w = panel.shape[:2]
    px = max(9, int(cfg.CREDIT_SIZE * scale))
    right = w - int(cfg.CREDIT_MARGIN * scale)
    y = h - footer // 2 + px // 3          # baseline, so the text sits mid-strip
    sep = int(cfg.CREDIT_SEPARATION * scale)
    value_rgb, label_rgb = cfg.CREDIT_COLOR[::-1], cfg.CREDIT_LABEL_COLOR[::-1]

    if font is None:
        for label, value in reversed(lines):
            text = f"{label}: {value}" if label else value
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 34, 1)
            cv2.putText(panel, text, (right - tw, y), cv2.FONT_HERSHEY_SIMPLEX, px / 34,
                        cfg.CREDIT_COLOR, 1, cv2.LINE_AA)
            right -= tw + sep
        return

    from PIL import Image, ImageDraw

    pil = Image.fromarray(panel[..., ::-1])
    draw = ImageDraw.Draw(pil)
    face = _face(font[0], font[1], px)
    for label, value in reversed(lines):
        # Baseline anchors ("rs"), so the label and value sit on one line.
        draw.text((right, y), value, font=face, fill=value_rgb, anchor="rs")
        vw = draw.textlength(value, font=face)
        lw = 0
        if label:
            lw = draw.textlength(f"{label}: ", font=face)
            draw.text((right - vw, y), f"{label}: ", font=face, fill=label_rgb, anchor="rs")
        right -= int(vw + lw + sep)
    panel[:] = np.array(pil)[..., ::-1]
