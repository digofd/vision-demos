"""Concentric time across sets: a box per set, and every rep in order.

The pipelines' plots.png already carries a box of one clip's concentric time.
This is the same measurement over all the sets of a run, in the same style:
on the left a box per set, to compare sets; on the right every rep as a line,
set 1 rep 1 through the last rep of the last set, to see where it slows down.
One set draws the same figure with one box.

Only full reps. A rep cut short has a short concentric phase because it was
cut short, not because it was fast, and would read as the best of the set.

Drawn from the summary rows exports.py builds, so it shows nothing the
downloaded summary CSV does not also say.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# The summary column that is the concentric phase, and how to say what it is.
CONCENTRIC = {
    "pull_ups": ("pulling_s", "onset to peak"),
    "chin_ups": ("pulling_s", "onset to peak"),
    "deadlift": ("lift_s", "floor to lockout"),
}

# The pipelines' box blue and point orange for set 1, then distinct hues.
SET_COLORS = ["#1864ab", "#e8590c", "#2b8a3e", "#862e9c", "#c92a2a",
              "#0b7285", "#5c940d", "#d6336c", "#495057", "#e67700"]
INK, MUTED = "#495057", "#868e96"


def _sets(demo_key: str, summary: list[dict]) -> tuple[list[tuple[int, list[dict]]], int]:
    """[(set, full reps with a concentric time)] in set order, and how many
    partial reps were left out."""
    column, _ = CONCENTRIC[demo_key]
    by_set: dict[int, list[dict]] = {}
    partial = 0
    for row in summary:
        if row.get("full_range_of_motion") == "no":
            partial += 1
            continue
        value = row.get(column)
        if isinstance(value, (int, float)) and value > 0:
            by_set.setdefault(row.get("set", 1), []).append(
                {"rep": row.get("rep"), "seconds": float(value)})
    return sorted(by_set.items()), partial


def concentric_plot(demo_key: str, summary: list[dict], path: Path) -> Path | None:
    """The figure at *path*, or None when there is no concentric time to plot
    or fewer than two full reps in all."""
    if demo_key not in CONCENTRIC:
        return None
    sets, partial = _sets(demo_key, summary)
    if sum(len(reps) for _, reps in sets) < 2:
        return None
    _, phase = CONCENTRIC[demo_key]
    color = {n: SET_COLORS[(n - 1) % len(SET_COLORS)] for n, _ in sets}
    # Zoomed to the reps rather than from zero: a set slowing from 1.0s to
    # 1.2s is the thing to see, and from zero it is a flat line. Headroom above
    # for the set labels; never below zero.
    values = [r["seconds"] for _, reps in sets for r in reps]
    low, high = min(values), max(values)
    span = max(high - low, 0.1 * high)
    bottom, top = max(0.0, low - 0.25 * span), high + 0.4 * span

    fig = Figure(figsize=(12, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=[1 + 0.45 * len(sets), 5], wspace=0.08)
    box_ax = fig.add_subplot(grid[0])
    line_ax = fig.add_subplot(grid[1], sharey=box_ax)

    # ── a box per set ───────────────────────────────────────────────────────
    for i, (n, reps) in enumerate(sets, 1):
        values = [r["seconds"] for r in reps]
        c = color[n]
        box_ax.boxplot([values], positions=[i], widths=0.5, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=c + "26", edgecolor=c, lw=1.4),
                       medianprops=dict(color=c, lw=2.2),
                       whiskerprops=dict(color=c, lw=1.2),
                       capprops=dict(color=c, lw=1.2))
        # Spread across the box in rep order, as the pipelines do: no two
        # reps share a spot, and a rerun draws the same figure.
        xs = np.linspace(i - 0.2, i + 0.2, len(values)) if len(values) > 1 else [i]
        box_ax.scatter(xs, values, s=40, zorder=3, color=c, edgecolor="white", lw=0.8)
        for x, y, rep in zip(xs, values, (r["rep"] for r in reps)):
            box_ax.annotate(str(rep), (x, y), textcoords="offset points",
                            xytext=(6, -3), fontsize=8, fontweight="bold", color=INK)
    box_ax.set_xticks(
        range(1, len(sets) + 1),
        [f"Set {n}\nn = {len(reps)}\nmedian {np.median([r['seconds'] for r in reps]):.2f}s"
         for n, reps in sets], fontsize=9)
    box_ax.set_xlim(0.4, len(sets) + 0.6)
    box_ax.set_ylim(bottom, top)
    box_ax.set_ylabel("concentric time (s)")
    box_ax.set_title("By set", fontsize=11, fontweight="bold", loc="left")
    box_ax.grid(axis="y", alpha=0.25)

    # ── every rep, in order ─────────────────────────────────────────────────
    # A set's reps sit side by side; a one-slot gap and a band mark each set,
    # and the line breaks between sets, since the rest between them is not a rep.
    x, ticks, labels = 0, [], []
    for i, (n, reps) in enumerate(sets):
        xs = list(range(x, x + len(reps)))
        values = [r["seconds"] for r in reps]
        c = color[n]
        if i % 2:
            line_ax.axvspan(xs[0] - 0.5, xs[-1] + 0.5, color="#f1f3f5", zorder=0)
        line_ax.plot(xs, values, color=c, lw=2, marker="o", ms=6,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        line_ax.text((xs[0] + xs[-1]) / 2, 0.97, f"Set {n}", ha="center",
                     va="top", fontsize=10, fontweight="bold", color=c,
                     transform=line_ax.get_xaxis_transform())
        ticks += xs
        labels += [f"R{r['rep']}" for r in reps]
        x += len(reps) + 1
    line_ax.set_xticks(ticks, labels, fontsize=8.5)
    line_ax.set_xlim(-0.6, x - 1.4)
    line_ax.tick_params(axis="y", labelleft=False)
    line_ax.set_title("Every full rep, in order", fontsize=11, fontweight="bold",
                      loc="left")
    line_ax.grid(axis="y", alpha=0.25)

    note = f"Concentric time, {phase} · full reps only"
    if partial:
        note += f", {partial} partial rep{'s' if partial != 1 else ''} left out"
    note += " · y-axis not from zero"
    fig.suptitle(note, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2)
    fig.savefig(path, facecolor="white")
    return path
