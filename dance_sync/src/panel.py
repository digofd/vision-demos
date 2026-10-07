"""The side panel: one similarity score, big, plus the trace behind it.

Built with OpenCV and Pillow rather than matplotlib, because this panel changes
every frame. A matplotlib figure costs tens of milliseconds to render and save;
547 of them would dominate the whole run. Instead the parts that never change —
title, axis, grid, the labelled anchor lines — are drawn once into a background
image, and each frame copies that and draws only what moves.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import text as textmod
from .skeleton import TRACK_COLORS

BG = (18, 18, 20)          # BGR, near-black so it sits beside a night clip
GRID = (40, 40, 46)        # faint: structure the trace sits on top of, not in
AXIS = (90, 90, 98)
DIM = (150, 150, 158)
INK = (245, 245, 245)
MUTED = (110, 110, 118)
TRACE = (205, 205, 210)    # neutral: the trace is the group, not any one dancer
# Colors for the three *pairs* in similarity.png. Deliberately a separate
# palette from TRACK_COLORS: a pair is not a dancer, and keying a pair's color
# on one of its members gives the two pairs containing dancer 0 the same color.
# Blending the members instead is worse — dancer 0's orange and dancer 1's blue
# average to grey. The plot legend names the members, and the panel's two dots
# carry the identity.
PAIR_PLOT_COLORS = ("#7048e8", "#0ca678", "#e64980")


def _catmull_rom(xs, ys, per_span: int):
    """Dense points along a Catmull-Rom spline through (xs, ys).

    The trace is sampled on a slow grid so the curve stops re-aiming thirty
    times a second, and joining those samples with straight chords made it
    *angular* — slower to update but sharper to look at, which is the opposite
    of the ask. A spline through the same samples keeps the slow update rate and
    gives back a smooth line: the vertices are the update, the curve is the
    rendering.

    Catmull-Rom because it passes exactly through every sample, so the drawn
    curve never invents a value the series does not have — a Bezier or a
    smoothing spline would pull the line off its own data.
    """
    n = len(xs)
    if n < 3:
        return list(zip(xs, ys))

    out = []
    for i in range(n - 1):
        p0 = (xs[max(i - 1, 0)], ys[max(i - 1, 0)])
        p1 = (xs[i], ys[i])
        p2 = (xs[i + 1], ys[i + 1])
        p3 = (xs[min(i + 2, n - 1)], ys[min(i + 2, n - 1)])
        for k in range(per_span):
            t = k / per_span
            t2, t3 = t * t, t * t * t
            # Uniform Catmull-Rom basis, halved as the standard form requires.
            b0 = -0.5 * t3 + t2 - 0.5 * t
            b1 = 1.5 * t3 - 2.5 * t2 + 1.0
            b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            b3 = 0.5 * t3 - 0.5 * t2
            out.append((p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
                        p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3))
    out.append((xs[-1], ys[-1]))
    return out


def _ramp(t: float, stops) -> tuple[int, int, int]:
    """Sample a multi-stop colour ramp at *t* in [0, 1], interpolated in BGR.

    Piecewise-linear between the stops, which is what makes a per-frame trace
    read as one continuous gradient rather than as bands of flat colour.
    """
    stops = np.asarray(stops, dtype=float)
    t = float(np.clip(t, 0.0, 1.0)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    frac = t - i
    return tuple(int(v) for v in (stops[i] + (stops[i + 1] - stops[i]) * frac))


def _segments(values: np.ndarray) -> list[list[int]]:
    """Runs of consecutive defined frames.

    The score is undefined wherever a dancer is out of shot — 13 separate gaps
    on this clip — and a single polyline over the defined points would draw a
    straight line across each one, inventing a smooth transition through a
    stretch that was never measured. Each run is drawn as its own polyline, so a
    gap reads as a gap.
    """
    runs: list[list[int]] = []
    current: list[int] = []
    for i, v in enumerate(values):
        if np.isfinite(v):
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def credit_lines(cfg) -> list[tuple[str, str]]:
    """The credit as ``[(label, value), ...]``, top line first. Empty for none.

    Each ``CREDIT`` entry is a ``(label, value)`` pair or a plain string; blank
    entries are skipped.
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


class SimilarityPanel:
    """Draws the panel for a given frame. One instance per render."""

    def __init__(self, analysis, width: int, height: int, *, cfg, font=None,
                 picture_times=()):
        self.a = analysis
        self.w, self.h = width, height
        self.cfg = cfg
        self.font = font
        self.u = width / 720.0            # everything is sized at a 720px-wide panel

        # ── Layout ───────────────────────────────────────────────────────
        # One content block — title, word, percentage, graph — measured from its
        # own type sizes and then centred vertically as a whole.
        #
        # Two blocks became one when the per-dancer leaderboard left for
        # ranking.png: it was the element that never had anything to say, since
        # the bootstrap ties all three dancers, so it printed the same three
        # bars for 18 seconds.
        #
        # Centring the *plot box* on the panel's midline was the first attempt
        # and it read wrong: the headline pins the top, so a centred box leaves
        # the whole surplus below it as one dead quarter. Centring the block
        # splits that surplus into equal margins instead.
        #
        # Horizontal placement is computed from the ink, not from the plot box.
        # The box used to sit at fixed fractions (0.19 to 0.945), which centred
        # *the box* and so pushed the graph visibly right: the left gutter
        # carries the tick numbers and the rotated y label, and none of that was
        # counted. Leftmost ink is the y label, rightmost is the axis's own end,
        # and both now sit PANEL_SIDE_MARGIN from the edge.
        self.size_axis_label = self._s(19)
        self.y_label_dx = self._s(60)     # plot edge to the rotated label's centre
        side = self._s(30)
        self.px0 = side + self.y_label_dx + self.size_axis_label // 2
        self.px1 = width - side

        # Type sizes, declared before the layout because the block's height is
        # the sum of them.
        #
        # The word is the one thing sized *down* here, and the size is set by
        # the *longest* label rather than by the look of a short one. _fit only
        # shrinks a label that would overflow the panel, and "PAY ATTENTION"
        # never did — it just took most of the width. Measured on this panel:
        #
        #     base   PAY ATTENTION   CLEAN!
        #       62            70%      32%
        #       52            59%      27%     <- default
        #       44            50%      23%
        #
        # At 52 the longest label leaves a real margin and the short ones are
        # still the panel's headline.
        self.size_title = self._s(30)
        self.size_word = self._s(52)
        self.size_percent = self._s(30)
        # The graph's own title, sized up so it reads as a label for the plot
        # rather than as a caption on it — it is what tells a first-time viewer
        # what the trace is. Kept under size_title so the panel still has one
        # headline.
        self.size_graph_title = self._s(28)

        # A square plot box. The trace was drawn into a tall rectangle, which
        # stretches every peak vertically and makes the dance look more erratic
        # than it is; at 1:1 a slope on screen is a slope in the data.
        plot_h = self.px1 - self.px0

        gap_word = self._s(38)
        gap_percent = self._s(10)
        gap_graph_title = self._s(44)     # the break between headline and graph
        gap_plot = self._s(16)
        # Tick labels and the x axis label. The label is centred y_label_dx
        # below the plot, matching the y label's offset, so the block has to
        # reserve that plus half its height.
        below_axis = self.y_label_dx + self.size_axis_label // 2

        # The centred span is the **whole visual extent**, tick labels and axis
        # label included. Leaving them out was tried and it read as a panel
        # sitting too low, for the obvious reason: they are ink on the screen,
        # and the eye centres what it can see, not the box the code thinks in.
        block = (self.size_title + gap_word + self.size_word + gap_percent
                 + self.size_percent + gap_graph_title + self.size_graph_title
                 + gap_plot + plot_h + below_axis)

        # ...then lifted slightly above true centre. Geometric centre reads low
        # here for two reasons: the block's own weight is top-heavy (a bold word
        # against a mostly empty plot), and the credit anchors the bottom
        # corner. Optical centring is the standard fix and PANEL_BLOCK_LIFT is
        # the knob — 0 for true geometric centre.
        lift = int(self.cfg.PANEL_BLOCK_LIFT * height)
        top = max(self._s(12), (height - block) // 2 - lift)

        self.y_title = top
        y = top + self.size_title + gap_word
        self.y_number = y + self.size_word // 2        # drawn centred, not topped
        y += self.size_word + gap_percent
        self.percent_dy = y - self.y_number            # the drop from word to %
        y += self.size_percent + gap_graph_title
        self.y_graph_title = y
        y += self.size_graph_title + gap_plot
        self.py0, self.py1 = y, y + plot_h

        self.series = analysis.smoothed if cfg.PANEL_SMOOTHED else analysis.series
        self.runs = _segments(self.series)
        # The number is sampled on a refresh grid and held in between; reading a
        # value that changes 30 times a second is not reading it.
        self.refresh = max(1, int(cfg.PANEL_REFRESH_FRAMES))
        # The final state starts when the scored window *ends*, not before it.
        #
        # This used to be backdated by a couple of seconds to give the label
        # screen time, and that was a lookahead bug, not a layout choice: the
        # average covers every scored frame, so showing it early displayed a
        # number derived from frames that had not played yet. The same applied
        # to the final leaderboard. Nothing about the whole-window result is
        # shown until the whole window has happened.
        end = (analysis.window_frames[1] if analysis.window_frames
               else analysis.n_frames - 1)
        self.final_from = end + 1
        # The x axis runs from the start of the scored window to its end.
        self.x0, self.x1 = (analysis.window_frames if analysis.window_frames
                            else (0, analysis.n_frames - 1))
        self.final_means = {row["dancer"]: row["mean"] for row in analysis.ranking}
        # The **true** mean of the scored frames — score_percent.mean in
        # similarity.json, unweighted.
        #
        # This was the picture-weighted mean, and that was overclaiming: a
        # headline reading AVG over a weighted number is not an average. The
        # weighting gives a landing 2.2x a valley (mean weight 0.52 in the
        # trace's top quartile against 0.23 in the bottom), which put the figure
        # at 59.7 against a true 53.1 — and drew the old average line above 70%
        # of its own curve.
        self.average = (float(np.mean(analysis.score)) if analysis.n_scored
                        else np.nan)
        self.picture_times = tuple(picture_times)

        # The y axis starts under the data, not at zero. The trace never goes
        # near 0 — the score bottoms out around the null, not at it — so a
        # 0-to-100 axis spent its lower third on space nothing can occupy, and
        # the plot box is square, so that space cost real height off the curve.
        #
        # The top stays at 100. That end is meaningful now that the ceiling is
        # measured: 100 is the best agreement these bodies reach, so the gap
        # above the trace is the headroom, not padding.
        finite_all = self.series[np.isfinite(self.series)]
        floor_cfg = cfg.PANEL_Y_FLOOR
        if floor_cfg == "auto":
            low = float(finite_all.min()) if finite_all.size else 0.0
            self.y_lo = float(max(0.0, np.floor((low - 5.0) / 10.0) * 10.0))
        else:
            self.y_lo = float(floor_cfg)
        self.y_hi = 100.0
        # Ticks on round tens, at whatever step keeps the axis to a few labels.
        span = self.y_hi - self.y_lo
        self.y_step = next(v for v in (10, 20, 25, 50) if span / v <= 5)

        # Colour is keyed on the score's percentile within this clip, the same
        # basis the bands use, so the label and the line can never disagree —
        # an absolute ramp would happily print CLEAN! over an orange trace.
        finite = np.sort(self.series[np.isfinite(self.series)])
        self._sorted = finite if finite.size else np.array([0.0, 100.0])

        self.read_value, self.band_seq = self._readout()
        self.background = self._background()

    def _held(self, frame_index: int) -> int:
        """The frame the slow readout is currently sampling."""
        return (frame_index // self.refresh) * self.refresh

    # ── geometry ─────────────────────────────────────────────────────────────
    def _sx(self, frame_index: float) -> int:
        """Frame to pixel, across the **scored window** rather than the clip.

        The axis used to span the whole video so the trace stayed aligned with
        playback, and the window's edges were drawn on as two dashed lines. That
        spent a third of the width on the freestyle intro and outro, where there
        is nothing to plot, and it forced the second labels onto a coarse step
        to stay legible. Starting the axis *at* the bounds says the same thing
        the dashed lines did, with the space given back to the part that has a
        trace in it.

        Clamped, because a 21-frame smoothing window can leave a finite value a
        few frames outside the scored range and there is no axis out there.
        """
        span = max(1, self.x1 - self.x0)
        t = (float(frame_index) - self.x0) / span
        return int(self.px0 + (self.px1 - self.px0) * float(np.clip(t, 0.0, 1.0)))

    def _sy(self, score: float) -> int:
        span = max(self.y_hi - self.y_lo, 1e-9)
        t = (float(np.clip(score, self.y_lo, self.y_hi)) - self.y_lo) / span
        return int(self.py1 - (self.py1 - self.py0) * t)

    def _s(self, px: float) -> int:
        return max(1, int(round(px * self.u)))

    def _fit(self, text: str, want: int) -> int:
        """Shrink a size until the text fits the panel width."""
        limit = self.w - self._s(40)
        size = want
        while size > self._s(14):
            wide, _ = textmod.measure(text, self.font, size)
            if wide <= limit:
                break
            size = int(size * 0.92)
        return size

    def _tint(self, score: float) -> tuple[int, int, int]:
        t = float(np.searchsorted(self._sorted, score) / max(len(self._sorted), 1))
        return _ramp(t, self.cfg.GRAPH_RAMP)

    def _band_tint(self, band: int) -> tuple[int, int, int]:
        """One colour per band, sampled at the middle of that band's range.

        The label is a discrete category, so it takes a discrete colour — under
        the continuous ramp a score just above a cut printed GOOD in warning
        orange, which reads as a contradiction. The trace stays a continuous
        gradient; only the word is quantized.
        """
        n = max(len(self.a.band_labels), 1)
        return _ramp((band + 0.5) / n, self.cfg.GRAPH_RAMP)

    def _readout(self) -> tuple[np.ndarray, np.ndarray]:
        """The value and the band to show on every frame, precomputed together.

        The gate is **asymmetric by direction**: a rise is cheap and a fall is
        expensive. Everything below exists in a rising and a falling variant,
        and the falling one is always the slower of the two.

        * **A rise clears immediately** — no hysteresis, and it may cut an
          existing label's dwell short after BAND_RISE_DWELL_FRAMES.
        * **A fall has to earn it** — clear the cut by BAND_FALL_HYSTERESIS and
          wait out the full BAND_FALL_DWELL_FRAMES since the last change.
        * **The bottom band alone skips the hysteresis**, once it has held for
          BAND_VALLEY_CONFIRM_FRAMES, because a valley is the one reading worth
          interrupting for.

        That asymmetry is the point: the readout is what a dancer watches, so it
        leans toward the moment they got it right, is slow to call a dip, and
        quick to forgive one. Nothing here touches the recorded score — the
        trace, the band segments in `similarity.json` and the averages all come
        straight off the analysis, so every dip is still logged whether or not
        the word ever said it.

        It is also what makes a long dwell usable at all. Symmetric, a 75-frame
        dwell blocked the *recovery* out of a dip as hard as the dip itself, so
        calming the word cost most of its CLEAN! screen time (188 frames to 87).
        Split by direction, the same dwell reads calmer *and* shows more of the
        peaks (256 frames).

        Value and band come from the same sample throughout, which is what stops
        the word contradicting the number beside it: the figure on screen is
        always the most recent reading that actually falls in the band on
        screen, so a lagging pair can never print CLEAN! above 36% again.
        """
        n_frames = len(self.series)
        values = np.full(n_frames, np.nan)
        bands = np.full(n_frames, -1, int)
        cuts = np.asarray(self.a.band_cuts, dtype=float)
        if not cuts.size:
            return values, bands

        hyst_up = float(self.cfg.BAND_RISE_HYSTERESIS)
        hyst_down = float(self.cfg.BAND_FALL_HYSTERESIS)
        confirm = max(1, int(self.cfg.BAND_VALLEY_CONFIRM_FRAMES))
        dwell_up = max(0, int(self.cfg.BAND_RISE_DWELL_FRAMES))
        dwell_down = max(0, int(self.cfg.BAND_FALL_DWELL_FRAMES))

        current = -1
        changed_at = -(10 ** 9)
        frozen = (np.nan, -1)

        def band_of(value: float) -> int:
            return int(np.searchsorted(cuts, value))

        def confirmed_valley(i: int) -> int:
            """The bottom band at *i*, if it holds for the confirmation window.

            Only the bottom band skips the hysteresis. It used to be both ends,
            on the reasoning that an extreme is the actionable moment and
            waiting is how it gets missed — but that was written when the bands
            were quartiles and the top one was a genuine extreme. It now covers
            the top 40% of the clip, which is the *normal* state, and letting it
            jump the queue meant it swallowed the middle two labels whole: NICE!
            went to 0 frames on screen and GOOD to 30. Reached through the
            hysteresis path instead, the readout uses all four words (36 / 30 /
            90 / 247 frames) at exactly the same change rate.

            The bottom decile is still a real excursion, so it keeps the jump.
            """
            window = self.series[i:i + confirm]
            if len(window) < confirm or not np.isfinite(window).all():
                return -1
            if all(band_of(v) == 0 for v in window):
                return 0
            return -1

        for i in range(n_frames):
            valley = confirmed_valley(i)
            if valley >= 0:
                candidate, sample = valley, self.series[i]
            else:
                sample = self.series[self._held(i)]
                candidate = current
                if np.isfinite(sample):
                    target = band_of(sample)
                    if current < 0:
                        candidate = target
                    elif target > current and sample >= cuts[current] + hyst_up:
                        candidate = target
                    elif target < current and sample <= cuts[current - 1] - hyst_down:
                        candidate = target

            gate = dwell_up if candidate > current else dwell_down
            if current < 0 and candidate >= 0:
                current, changed_at, frozen = candidate, i, (sample, candidate)
            elif candidate != current and (i - changed_at) >= gate:
                current, changed_at, frozen = candidate, i, (sample, candidate)

            # The number on screen is always the most recent reading that
            # belongs to the band on screen. A reading the label would deny is
            # skipped rather than printed — a lagging pair once put CLEAN! above
            # a displayed 36% — and because agreement resumes as soon as the
            # score re-enters the band, it never goes stale for long.
            if np.isfinite(sample) and band_of(sample) == current:
                frozen = (sample, current)
            values[i], bands[i] = frozen
        return values, bands

    # ── the parts that never change ──────────────────────────────────────────
    def _background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        f = self.font

        textmod.draw(img, self.cfg.PANEL_GRAPH_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_graph_title), color=DIM, anchor="ct")

        # No horizontal gridlines — the labels carry the scale and the lines were
        # only competing with the trace.
        first = int(np.ceil(self.y_lo / self.y_step) * self.y_step)
        for pct in range(first, int(self.y_hi) + 1, int(self.y_step)):
            textmod.draw(img, f"{pct}", f, size=self._s(15),
                         xy=(self.px0 - self._s(9), self._sy(pct)), color=MUTED, anchor="rm")

        cv2.line(img, (self.px0, self.py0), (self.px0, self.py1), AXIS, 1, cv2.LINE_AA)
        cv2.line(img, (self.px0, self.py1), (self.px1, self.py1), AXIS, 1, cv2.LINE_AA)

        # No vertical lines at all. The window bounds used to be dashed on and
        # the seconds gridded inside them; now that the axis *is* the window,
        # the bounds are its ends and the labels below carry the pace. Both were
        # only structure for the trace to sit on, and the trace reads better
        # without anything behind it.

        # The pictures, ticked on the axis: little marks at each landing, which
        # is what the average is weighted toward. Off by default — they sat on
        # the axis line itself and read as an artifact of the axis rather than
        # as data. PANEL_PICTURE_TICKS brings them back.
        if self.cfg.PANEL_PICTURE_TICKS:
            for t in self.picture_times:
                x = self._sx(t * (self.a.fps or 30.0))
                cv2.line(img, (x, self.py1 - self._s(4)), (x, self.py1),
                         (150, 190, 150), self._s(2), cv2.LINE_AA)

        # Seconds on the x axis. Without them the trace has a shape but no pace.
        # Whole seconds only, from the first one inside the window: the window
        # is 4.0-17.4s, so ticking from its edges would label the axis 4.0, 6.7,
        # 9.4 — three digits each, and none of them a number anyone reads a
        # timestamp as. The two ends are where the trace starts and stops, which
        # already says where the window is.
        fps = self.a.fps or 30.0
        t0, t1 = self.x0 / fps, self.x1 / fps
        span = max(t1 - t0, 1e-6)
        step = next(v for v in (1, 2, 5, 10, 15, 30, 60) if span / v <= 8)
        for sec in range(int(np.ceil(t0)), int(np.floor(t1)) + 1):
            if sec % step:
                continue
            # Bare numbers: the axis label underneath already says "time (s)",
            # and a unit on every tick is the same word seven times.
            textmod.draw(img, f"{sec}", f, size=self._s(14),
                         xy=(self._sx(sec * fps), self.py1 + self._s(8)),
                         color=MUTED, anchor="ct")

        # Axis labels. The y axis is the rotated one, which is why text.draw
        # grew a `rotate` argument.
        textmod.draw(img, self.cfg.PANEL_Y_LABEL, f, size=self.size_axis_label,
                     xy=(self.px0 - self.y_label_dx, (self.py0 + self.py1) // 2),
                     color=DIM, anchor="cm", rotate=90)
        # Centred on the *panel*, not on the plot box, so it sits in one column
        # with the graph title and the headline above it. The box's own centre is
        # 26px to the right — the left gutter is not part of the axis — and a
        # label visibly out of line with the title over it reads as a mistake,
        # while a label a few pixels off its own axis midpoint does not.
        #
        # Set off from the plot by the *same* y_label_dx as its partner, and
        # anchored the same way (centre, not top), so the two axis labels sit an
        # equal distance from their own edge. It used to be 19px against the y
        # label's 45 and the axis looked lopsided.
        textmod.draw(img, self.cfg.PANEL_X_LABEL, f, size=self.size_axis_label,
                     xy=(self.w // 2, self.py1 + self.y_label_dx),
                     color=DIM, anchor="cm")

        self._credit(img)
        return img

    def _credit(self, img) -> None:
        """The credit stack, bottom-right; the last line's ink sits on the margin.

        Each part is placed from a shared baseline, so "X:" and "@handle" line up.
        """
        cfg, f = self.cfg, self.font
        lines = credit_lines(cfg)
        if not lines:
            return
        size = self._s(cfg.CREDIT_SIZE)
        right = self.w - self._s(cfg.CREDIT_MARGIN_X)
        label, value = lines[-1]
        y = self.h - self._s(cfg.CREDIT_MARGIN_Y) - max(
            textmod.descent(value, f, size), textmod.descent(f"{label}:", f, size) if label else 0)
        for label, value in reversed(lines):
            textmod.draw(img, value, f, size=size,
                         xy=(right, y + textmod.descent(value, f, size)),
                         color=INK, anchor="rb", opacity=cfg.CREDIT_OPACITY)
            if label:
                vw = textmod.measure(value, f, size)[0]
                tag = f"{label}:"
                textmod.draw(img, tag, f, size=size,
                             xy=(right - vw - self._s(cfg.CREDIT_SIZE * 0.3),
                                 y + textmod.descent(tag, f, size)),
                             color=INK, anchor="rb", opacity=cfg.CREDIT_LABEL_OPACITY)
            y -= size + self._s(cfg.CREDIT_LINE_GAP)

    # ── per-frame ────────────────────────────────────────────────────────────
    def draw(self, frame_index: int) -> np.ndarray:
        img = self.background.copy()
        f = self.font
        is_final = frame_index >= self.final_from

        # At the end the headline stops being "right now" and becomes the clip
        # average, with the title changed to say so. A number that reads the
        # same either way would be taken for the live value.
        if is_final and np.isfinite(self.average):
            value, defined = self.average, True
        else:
            value = (self.read_value[frame_index]
                     if frame_index < len(self.read_value) else np.nan)
            defined = bool(np.isfinite(value))

        textmod.draw(img, self.cfg.PANEL_FINAL_NUMBER_TITLE if is_final
                     else self.cfg.PANEL_TITLE, f, size=self.size_title,
                     xy=(self.w // 2, self.y_title), color=INK, anchor="ct")

        if defined:
            color = self._tint(value)
            # In the final window the headline is the clip average, so the word
            # has to be the average's band — reading it off the live series
            # printed PAY ATTENTION above an average of 55%.
            if is_final:
                band = int(np.searchsorted(np.asarray(self.a.band_cuts), value))
            else:
                band = (int(self.band_seq[frame_index])
                        if frame_index < len(self.band_seq) else -1)
            if self.cfg.PANEL_HEADLINE_BAND and self.a.band_labels and band >= 0:
                label = self.a.band_labels[band]
                # Sized up with the leaderboard gone: the word is the panel's
                # one headline now, so it gets the room the three bars had.
                textmod.draw(img, label, f, size=self._fit(label, self.size_word),
                             xy=(self.w // 2, self.y_number),
                             color=self._band_tint(band), anchor="cm")
                if self.cfg.PANEL_SHOW_PERCENT:
                    textmod.draw(img, f"{value:.0f}%", f, size=self.size_percent,
                                 xy=(self.w // 2, self.y_number + self.percent_dy),
                                 color=MUTED, anchor="ct")
            else:
                text = f"{value:.0f}"
                size = self._s(152)
                wide, _ = textmod.measure(text, f, size)
                pct_w, _ = textmod.measure("%", f, self._s(54))
                left = self.w // 2 - (wide + self._s(6) + pct_w) // 2
                textmod.draw(img, text, f, size=size, xy=(left, self.y_number),
                             color=color, anchor="lm")
                textmod.draw(img, "%", f, size=self._s(54),
                             xy=(left + wide + self._s(6), self.y_number + self._s(26)),
                             color=color, anchor="lb")

        # The trace up to now — progressive, so the panel reads as a recording
        # being made rather than a finished chart with a cursor on it.
        # Drawn segment by segment rather than as one polyline, because each
        # segment takes the colour of its own value — that is what makes the line
        # a continuous gradient instead of blocks of flat colour.
        #
        # Sampled every PANEL_TRACE_STEP frames, not every frame. This is the
        # *display's* update rate, not the data's: at one vertex per frame the
        # curve re-aimed 30 times a second and read as twitchy, when what it has
        # to say happens on the scale of a landing. Decimating the vertices
        # cannot merge landings the way a wider smoothing window does — the
        # geometry between them is simply drawn straighter.
        #
        # The last segment still runs out to the live frame, so the trace always
        # meets the marker and the playhead rather than lagging a third of a
        # second behind them.
        thickness = self._s(2)
        step = max(1, int(self.cfg.PANEL_TRACE_STEP))
        for run in self.runs:
            visible = [i for i in run if i <= frame_index]
            if len(visible) < 2:
                continue
            points = [i for i in visible if i % step == 0]
            if points[:1] != visible[:1]:
                points.insert(0, visible[0])
            if points[-1:] != visible[-1:]:
                points.append(visible[-1])
            curve = _catmull_rom([float(i) for i in points],
                                 [float(self.series[i]) for i in points],
                                 per_span=max(2, step // 2))
            for (fa, va), (fb, vb) in zip(curve[:-1], curve[1:]):
                cv2.line(img, (self._sx(fa), self._sy(va)),
                         (self._sx(fb), self._sy(vb)),
                         self._tint(0.5 * (va + vb)), thickness, cv2.LINE_AA)

        # The clip average, marked on the trace once the headline becomes that
        # average. Back after a spell removed: it was drawn at the
        # picture-weighted mean, which sat above 70% of its own curve and read
        # as wrong. Against the true mean it lands where an average should, and
        # it is what tells you at a glance which passages were above the day's
        # own level.
        if is_final and np.isfinite(self.average):
            y = self._sy(self.average)
            for x in range(self.px0, self.px1, self._s(11)):
                cv2.line(img, (x, y), (min(x + self._s(6), self.px1), y),
                         self._tint(self.average), 1, cv2.LINE_AA)
            textmod.draw(img, "avg", f, size=self._s(14),
                         xy=(self.px1, y - self._s(5)),
                         color=self._tint(self.average), anchor="rb")

        # A playhead that travels the scored window in real time. It is what
        # makes the trace scrubbable: the x position against the second labels
        # below tells you where to look in the video.
        #
        # It used to travel the whole clip, because the axis did. Now that the
        # axis stops at the window it is drawn only inside it — _sx clamps, so
        # otherwise it would park on the axis end through the whole freestyle
        # intro and read as a stuck playhead rather than as an absent one.
        #
        # No timestamp of its own: the line's position against the second
        # labels already says where we are, and printing "0.0s" directly under
        # the axis's own "0s" was just saying it twice.
        px = self._sx(frame_index)
        if self.x0 <= frame_index <= self.x1:
            cv2.line(img, (px, self.py0), (px, self.py1), (86, 86, 94), 1, cv2.LINE_AA)

        # The score dot rides the curve, and simply is not drawn where there is
        # no score — before the window, after it, or in a gap. Nothing is
        # invented to keep it on screen.
        live = self.series[frame_index] if frame_index < len(self.series) else np.nan
        if np.isfinite(live):
            mark = (px, self._sy(live))
            cv2.circle(img, mark, self._s(5), self._tint(live), -1, cv2.LINE_AA)
            cv2.circle(img, mark, self._s(5), INK, self._s(1), cv2.LINE_AA)

        return img


def plot_ranking(analysis, dst, *, cfg, title: str = "") -> None:
    """The per-dancer ranking as a still, replacing the panel's leaderboard.

    This lives here rather than on the video because a tie is the answer on this
    clip, and a tie is something to *show*, not to state. The live leaderboard
    printed three bars and three rank badges that never moved for 18 seconds,
    which reads as a ranking that has been computed and is being withheld. Three
    overlapping confidence intervals say the real thing in one glance.

    One row per dancer per measure: the dot is the mean the ranking is built on,
    the bar is the 95% interval from a **block bootstrap over one-second
    blocks** — consecutive frames of a dance are nowhere near independent, and a
    naive bootstrap over frames would report intervals several times too tight.
    Two dancers share a rank unless one wins RANK_TIE_BAND of the resamples, so
    a lean is never printed as a ranking.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = [("overall", analysis.ranking)]
    measures += [(c.name, c.ranking) for c in analysis.components if c.ranking]
    measures = [(name, rows) for name, rows in measures if rows]
    if not measures:
        return

    fig, axes = plt.subplots(1, len(measures), figsize=(4.6 * len(measures), 2.9),
                             sharey=True)
    axes = np.atleast_1d(axes)

    slots = list(analysis.slots)
    y_of = {slot: len(slots) - 1 - i for i, slot in enumerate(slots)}

    for ax, (name, rows) in zip(axes, measures):
        for row in rows:
            slot = row["dancer"]
            y = y_of.get(slot, 0)
            color = "#%02x%02x%02x" % TRACK_COLORS[slot % len(TRACK_COLORS)][::-1]
            lo, hi = row.get("ci95", (row["mean"], row["mean"]))
            ax.plot([lo, hi], [y, y], color=color, lw=3.0, alpha=0.35,
                    solid_capstyle="round", zorder=2)
            ax.plot([row["mean"]], [y], "o", color=color, ms=9, zorder=3)
            ax.text(hi + 1.5, y, f"{row['mean']:.1f}%", va="center", ha="left",
                    fontsize=9, color="#444")
            # The rank per measure, inside the axes: a dancer can place
            # differently on posture and on timing, so it cannot live in the
            # shared y labels.
            ax.text(2.5, y, f"#{row['rank']}", va="center", ha="left",
                    fontsize=9, color="#888")
        ax.set_ylim(-0.65, len(slots) - 0.35)
        ax.set_yticks(range(len(slots)))
        ax.set_yticklabels([f"dancer {s}" for s in reversed(slots)])
        ax.set_xlim(0, 100)
        ax.set_xlabel("sync score")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        # Say the verdict in words as well, because the overlap is the point and
        # a reader should not have to measure it off the axis.
        ranks = {row["rank"] for row in rows}
        if len(ranks) == 1:
            verdict = "all tied"
        else:
            best = [r for r in rows if r["rank"] == min(ranks)]
            verdict = ("%s ahead" % ", ".join(f"dancer {r['dancer']}" for r in best)
                       if len(best) < len(rows) else "all tied")
        ax.set_title(f"{name} — {verdict}", fontsize=11, fontweight="bold")

    band = getattr(cfg, "RANK_TIE_BAND", 0.35)
    fig.suptitle(title or "Per-dancer ranking", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015,
             f"dot = mean, bar = 95% interval from a block bootstrap over "
             f"{cfg.RANK_BOOTSTRAP_BLOCK_SECONDS:.0f}s blocks "
             f"({cfg.RANK_BOOTSTRAP_SAMPLES} resamples). Two dancers share a rank "
             f"unless one wins {(0.5 + band / 2) * 100:.0f}% of resamples — "
             f"overlapping bars mean the difference is not measurable.",
             ha="center", fontsize=8.5, color="#666")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.26, wspace=0.10)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_similarity(analysis, dst, *, cfg, title: str = "") -> None:
    """The whole series as a still, for looking at after the run.

    Shows the raw per-frame score under the smoothed one — the panel displays
    the smoothed curve, and this is where you can see how much that smoothing
    is doing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(analysis.n_frames) / (analysis.fps or 30.0)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})

    ax.plot(t, analysis.series, color="#c9ced6", lw=0.9, label="per frame", zorder=2)
    ax.plot(t, analysis.smoothed, color="#2f6df6", lw=2.2, zorder=3,
            label=f"smoothed ({cfg.SIMILARITY_SMOOTH_FRAMES} frames, gaps interpolated)")
    ax.set_ylim(-2, 102)
    ax.set_ylabel("sync\n(0 = unrelated, 100 = identical)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=4, fontsize=9, frameon=False)
    ax.set_title(title or "Sync score", fontsize=13, fontweight="bold", pad=26)

    # Per dancer, not per pair: this is the series the ranking is built on, and
    # the running mean is what the panel's leaderboard shows.
    ranks = {row["dancer"]: row for row in analysis.ranking}
    for i, slot in enumerate(analysis.slots):
        color = "#%02x%02x%02x" % TRACK_COLORS[slot % len(TRACK_COLORS)][::-1]
        row = ranks.get(slot, {})
        tie = " (tied)" if row.get("tied_with") else ""
        ax2.plot(t, analysis.dancer_running[:, i], lw=2.0, color=color,
                 label=f"#{row.get('rank', '?')} dancer {slot} — "
                       f"{row.get('mean', float('nan')):.1f}%{tie}")
        ax2.plot(t, analysis.dancer_series[:, i], lw=0.7, alpha=0.35, color=color)
    ax2.set_ylim(-2, 102)
    # 0 is a floor, not a minimum: the score clamps there whenever a pair is at
    # or past the unrelated baseline, which is why the thin lines touch the axis.
    ax2.set_ylabel("per dancer\n(bold = running mean)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, ncol=3, frameon=False, loc="lower right")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.09, hspace=0.12)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_components(analysis, dst, *, cfg, title: str = "") -> None:
    """Posture and timing, separately, plus how they make the summary.

    Kept out of the video panel on purpose — three traces and two anchor sets is
    more than a viewer can read while watching people dance — but it is the plot
    to look at when the summary moves and you want to know which half moved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = {c.name: c for c in analysis.components}
    posture, timing = comps.get("posture"), comps.get("timing")
    t = np.arange(analysis.n_frames) / (analysis.fps or 30.0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2]})
    COL = {"posture": "#2f6df6", "timing": "#e8590c"}

    for ax, comp in zip(axes[:2], (posture, timing)):
        if comp is None:
            continue
        ax.plot(t, comp.series, color="#c9ced6", lw=0.8, zorder=2, label="per frame")
        ax.plot(t, comp.smoothed, color=COL[comp.name], lw=2.2, zorder=3,
                label=f"smoothed ({cfg.SIMILARITY_SMOOTH_FRAMES} frames)")
        med = float(np.median(comp.score))
        ax.axhline(med, color=COL[comp.name], ls="--", lw=1.0, alpha=0.6, zorder=1,
                   label=f"median {med:.0f}%")
        ax.set_ylim(-2, 102)
        ax.set_ylabel(f"{comp.name}\n(%)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5, ncol=3, frameon=False)
        ax.set_title(
            f"{comp.name.upper()} — "
            + ("limb directions: are we in the same shape?" if comp.name == "posture"
               else "limb speeds: are we moving and stopping together?")
            + f"    0% = {comp.null_distance:.1f} {comp.unit} (unrelated moments),"
              f"  100% = under {comp.tolerance:.1f} {comp.unit} (noise floor)",
            fontsize=10, loc="left", pad=6)

    ax = axes[2]
    if posture is not None and timing is not None:
        ax.plot(t, posture.smoothed, color=COL["posture"], lw=1.4, alpha=0.75,
                label=f"posture x {posture.weight:.2f}")
        ax.plot(t, timing.smoothed, color=COL["timing"], lw=1.4, alpha=0.75,
                label=f"timing x {timing.weight:.2f}")
    ax.plot(t, analysis.smoothed, color="#212529", lw=2.6, zorder=4,
            label="summary (what the panel shows)")
    ax.set_ylim(-2, 102)
    ax.set_ylabel("summary\n(%)")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, ncol=3, frameon=False)
    formula = " + ".join(f"{c.weight:.2f} x {c.name}" for c in analysis.components)
    r = (np.corrcoef(posture.score, timing.score)[0, 1]
         if posture is not None and timing is not None else float("nan"))
    ax.set_title(f"SUMMARY = {formula}    "
                 f"(the two components correlate at r = {r:+.2f}, so each carries "
                 f"information the other does not)",
                 fontsize=10, loc="left", pad=6)

    fig.suptitle(title or "Posture and timing", fontsize=13, fontweight="bold", x=0.09,
                 ha="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.07, hspace=0.30)
    fig.savefig(dst, dpi=150)
    plt.close(fig)
