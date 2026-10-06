"""Measure what a full-range rep looks like for one athlete, once.

    conda activate pull_ups
    python calibrate.py                          # CALIBRATION_VIDEO from config.py
    python calibrate.py data/input/full_set.mov  # or any clip

Point it at a set of full-range reps; `main.py` then grades later clips against it.
Writes `CALIBRATION_PROFILE`; delete it or use `main.py --uncalibrated` to opt out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.table import Table
from rich.panel import Panel

import config as cfg
from main import console, inference_tonemap, kv, pose_clip, prepare, rel
from src import calibrate, reps, video
from src.env import load_api_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn what a full-range rep measures for one athlete, from a clip of "
                    "a set taken through full range.")
    parser.add_argument("clip", nargs="?", type=Path, metavar="CLIP",
                        help=f"the reference set (config: {rel(cfg.CALIBRATION_VIDEO)})")
    parser.add_argument("--out", type=Path, metavar="JSON",
                        help=f"where to write the profile (config: {rel(cfg.CALIBRATION_PROFILE)})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the cached poses and call the model again")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out:
        cfg.CALIBRATION_PROFILE = args.out.expanduser().resolve()
    if args.fresh:
        cfg.REUSE_POSES = False
    cfg.GRADE_ROM = True   # the measurements ride on the grader, whatever config.py says
    console.print()
    console.rule("[bold]Calibrate: what a full rep measures[/]", align="center")

    clip = (args.clip or Path(cfg.CALIBRATION_VIDEO)).expanduser().resolve()
    if not clip.is_file():
        console.print(f"[red]Not found:[/] {clip}")
        console.print("  [dim]pass a clip of a full-range set: python calibrate.py path/to/set.mov[/]")
        return 1

    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key from [green]{source}[/]")
    console.print(f"  reference [bold]{rel(clip)}[/]\n")

    src_info = video.probe_source(clip)
    _, infer_tonemap, _ = inference_tonemap(src_info)
    mp4 = prepare(clip, cfg.INFERENCE_HEIGHT, "inference", tonemap=infer_tonemap,
                  n_frames=src_info["n_frames"])
    info = video.inspect(mp4)
    frames = pose_clip(api_key, mp4, info)["frames"]

    # Uncalibrated on purpose: the reference defines the standard, so it must not
    # be filtered by it. Detection uses the strict height gate, fine for good reps.
    analysis = reps.analyze(frames, info.fps, info.n_frames, cfg=cfg,
                            aspect=info.width / info.height, profile=None)
    if not analysis.reps:
        console.print("\n[red]No reps detected in the reference clip.[/]")
        return 1

    profile = calibrate.build(analysis, clip.name)

    table = Table(box=None, pad_edge=False)
    for col in ("rep", "hang", "peak", "sweep", "clear"):
        table.add_column(col, justify="right", style="dim" if col == "rep" else None)
    for rep in analysis.reps:
        g = rep.rom
        table.add_row(str(rep.number), f"{g.hang_degrees:.1f}°", f"{g.peak_degrees:.1f}°",
                      f"{g.hang_degrees - g.peak_degrees:.1f}°", f"{g.clearance:.2f}")
    console.print()
    console.print(Panel(table, title=f"[bold green]{len(analysis.reps)} reference reps[/]",
                        title_align="left", expand=False))

    kv(profile.summary(), title="learned")

    # A reference whose reps disagree with each other was probably the wrong clip.
    if profile.hang_degrees.spread > cfg.CALIBRATION_MAX_SPREAD_DEGREES:
        console.print(
            f"  [yellow]warning[/] the reference reps disagree about the bottom by "
            f"{profile.hang_degrees.spread:.1f}°, over the "
            f"{cfg.CALIBRATION_MAX_SPREAD_DEGREES:.0f}° this expects. "
            "Was every rep in this clip taken to a full dead hang?"
        )
    if len(analysis.reps) < cfg.CALIBRATION_MIN_REPS:
        console.print(f"  [yellow]warning[/] only {len(analysis.reps)} reps; "
                      f"{cfg.CALIBRATION_MIN_REPS}+ makes a steadier reference.")

    dst = calibrate.path(cfg)
    calibrate.save(profile, dst)
    console.print(f"\n  profile -> [dim]{rel(dst)}[/]")

    kv({
        "full hang at": f"≥ {profile.hang_required(cfg):.1f}°  "
                        f"({profile.hang_degrees.mean:.1f}° − "
                        f"{cfg.ROM_HANG_TOLERANCE_DEGREES:g}°)",
        "clears the bar at": f"≥ {profile.clearance_required(cfg):.2f}  "
                             f"({profile.clearance.mean:.2f} × "
                             f"{cfg.ROM_CLEARANCE_FRACTION:.2f})",
        "counts as a rep at": f"≥ {profile.min_flexion(cfg):.0f}° of flexion  "
                              f"({profile.sweep:.0f}° × "
                              f"{cfg.REP_MIN_FLEXION_FRACTION:.2f})",
    }, title="every later run will grade against")

    console.print("\n  [bold]python main.py[/] now grades against this.\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
