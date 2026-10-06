"""Count pull-ups, time each pull, and grade each rep's range of motion as full or partial.

    conda activate pull_ups
    python calibrate.py                   # once: what a full rep measures for you
    python main.py                        # every clip in data/input/
    python main.py data/input/set.mov     # just this one (clips or folders)
    python main.py -h                     # every flag

One model: ViTPose poses every frame, giving the rep signal, both elbow angles,
and how far the head clears the bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import calibrate, pose, render, reps, video
from src.env import load_api_key

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    """Project-relative path for console output; absolute when outside the project."""
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def prepare(clip: Path, height, label: str, *, tonemap: str | None, n_frames: int) -> Path:
    """Convert *clip* to *height* (None = source), reusing a cached copy if present."""
    path = video.cache_path(clip, cfg.CACHE_DIR, target_height=height,
                            trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                            tonemap=tonemap)
    if path.is_file() and not cfg.FORCE_RECONVERT:
        console.print(f"  [green]cache hit[/] ({label}): [dim]{rel(path)}[/]")
    else:
        console.print(f"  [yellow]converting[/] ({label})")
        video.convert(clip, path, target_height=height, trim_seconds=cfg.TRIM_SECONDS,
                      crf=cfg.CONVERT_CRF, total_frames=n_frames, console=console,
                      tonemap=tonemap)
    return path


def inference_tonemap(src_info: dict) -> tuple[str | None, str | None, str]:
    """``(export_filter, inference_filter, reason)`` for this source.

    Decided before the cache key is built, so toggling tone-mapping gives a new MP4.
    """
    tonemap, reason = video.tonemap_filter(src_info, mode=cfg.TONEMAP,
                                           algorithm=cfg.TONEMAP_ALGORITHM)
    return tonemap, tonemap if cfg.TONEMAP_INFERENCE else None, reason


def pose_clip(api_key: str, mp4: Path, info: video.VideoInfo) -> dict:
    """ViTPose on every frame of *mp4*, from the cache when it matches.

    Shared with calibrate.py, so reference and graded clips use the same request.
    """
    video_b64, extra_body = pose.build_request(
        mp4, every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES, precision=cfg.PRECISION)
    cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL, extra_body=extra_body)
    cached = pose.load_cache(cache) if cfg.REUSE_POSES else None
    seconds = None
    if cached is not None:
        payload, usage, _, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/]; no gateway call")
    else:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                        timeout=cfg.REQUEST_TIMEOUT, max_retries=1)
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; ViTPose on every frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(client, model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        seconds = time.perf_counter() - t0
        pose.save_cache(cache, payload, usage, {}, stamp=stamp_now())
        console.print(f"  [green]done in {seconds:.1f}s[/]")

    frames, _ = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    return {
        "frames": frames,
        "by_index": {f["index"]: f["persons"] for f in frames},
        "extra_body": extra_body,
        "dropped": dropped,
        "seconds": seconds,
        # What this run spent: a cached reply was paid for by the run that made it.
        "cost": 0.0 if cached is not None else float((usage or {}).get("cost") or 0),
    }


def run_clip(api_key: str, profile) -> dict:
    """The whole pipeline for ``cfg.INPUT_VIDEO``. Returns what the batch summary prints."""
    t_start = time.perf_counter()

    # ── 2. Source ────────────────────────────────────────────────────────────
    rule(2, "Source video")
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']}°",
        "color": f"{src_info['color_primaries'] or '?'} / {src_info['color_transfer'] or '?'}"
                  + ("  [yellow](HDR)[/]" if src_info["is_hdr"] else "  [dim](SDR)[/]"),
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    # ── 3. Convert (cached) ──────────────────────────────────────────────────
    rule(3, "Convert to MP4")
    tonemap, infer_tonemap, tonemap_reason = inference_tonemap(src_info)
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{tonemap_reason}"
                  + ("" if not tonemap or infer_tonemap else
                     "  [dim](export only; the model sees the plain conversion)[/]"))

    mp4 = prepare(cfg.INPUT_VIDEO, cfg.INFERENCE_HEIGHT, "inference", tonemap=infer_tonemap,
                  n_frames=src_info["n_frames"])
    info = video.inspect(mp4)
    console.print(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")
    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT and tonemap == infer_tonemap:
        export_info = info
    else:
        export_info = video.inspect(prepare(cfg.INPUT_VIDEO, cfg.EXPORT_HEIGHT, "export",
                                            tonemap=tonemap, n_frames=src_info["n_frames"]))
        console.print(f"  export    [bold]{export_info}[/]")
        if export_info.n_frames != info.n_frames:
            console.print(f"  [yellow]warning[/] frame counts differ ({info.n_frames} vs "
                          f"{export_info.n_frames}); the overlay is matched by frame index")

    # ── 4. Pose: ViTPose ─────────────────────────────────────────────────────
    rule(4, "Pose (ViTPose)")
    posed = pose_clip(api_key, mp4, info)
    frames, by_index = posed["frames"], posed["by_index"]
    n_posed = sum(1 for f in frames if f["persons"])
    kv({"model": cfg.POSE_MODEL,
        "frames with a person": f"{n_posed} of {info.n_frames}",
        "duplicates dropped": posed["dropped"],
        "cost": f"${posed['cost']:.4f}"}, title="pose")

    # ── 5. Reps and range of motion ──────────────────────────────────────────
    rule(5, "Reps and range of motion")
    # Angles need the real aspect: normalized x and y scale differently.
    analysis = reps.analyze(frames, info.fps, info.n_frames, cfg=cfg,
                            aspect=info.width / info.height, profile=profile)
    full, graded = analysis.rom_tally
    if analysis.reps:
        table = Table(box=None, pad_edge=False)
        for col in ("rep", "rom", "bottom", "top", "clear", "pull", "onset", "peak"):
            table.add_column(col, justify="left" if col == "rom" else "right",
                             style="dim" if col == "rep" else None)
        for rep in analysis.reps:
            g = rep.rom

            def num(value, fmt, ok=True, unit=""):
                if g is None or not np.isfinite(value):
                    return "—"
                text = format(value, fmt) + unit
                return text if ok else f"[red]{text}[/]"

            table.add_row(
                str(rep.number),
                "[dim]not graded[/]" if g is None else
                "[green]full[/]" if rep.complete else f"[red]partial: {g.reason}[/]",
                num(g and g.hang_degrees, ".0f", g and g.bottom_ok, "°"),
                num(g and g.peak_degrees, ".0f", g and g.top_ok, "°"),
                num(g and g.clearance, ".2f", g and g.top_ok),
                f"{reps.rep_value(rep, cfg.REP_METRIC):.2f}s",
                f"{rep.onset_time:.2f}s", f"{rep.peak_time:.2f}s")
        console.print(Panel(table, title=f"[bold green]{analysis.count} reps[/]"
                                         + (f" · [bold]{full}/{graded}[/] full range"
                                            if graded else "")
                                         + f" [dim](pull: {reps.phase_label(cfg.REP_METRIC)})[/]",
                            title_align="left", expand=False))
        if graded:
            console.print(f"  held to: full hang ≥ {analysis.hang_required:.1f}°, "
                          f"clears the bar ≥ {analysis.clearance_required:.2f} torso lengths "
                          f"[dim]({'calibration profile' if analysis.calibrated else 'this clip + config.py defaults'})[/]")
    else:
        console.print("  [yellow]no reps detected[/]. Check timeline.png")
    if analysis.off_bar_frames:
        console.print(f"  [dim]{analysis.off_bar_frames} frames excluded, hands off the bar[/]")

    # ── 6. Output ────────────────────────────────────────────────────────────
    rule(6, "Render")
    # One directory per clip; suffixed if two clips finish in the same second.
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{n}"
        n += 1
    stamp = run_dir.name
    run_dir.mkdir(parents=True)

    (run_dir / "reps.json").write_text(json.dumps({
        "fps": info.fps,
        "signal": reps.signal_config(cfg, profile),
        **analysis.summary(),
        "series": analysis.series(),
    }, indent=2))

    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(export_info, by_index=by_index, analysis=analysis,
                          n_frames=info.n_frames, fps=info.fps, dst=raw, cfg=cfg,
                          console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_pull_ups.mp4"
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)   # MPEG-4 Part 2 intermediate; no player wants it
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB, {render_seconds:.1f}s)")

    timeline = run_dir / "timeline.png"
    pulls = [reps.rep_value(r, cfg.REP_METRIC) for r in analysis.reps]
    render.plot_timeline(
        analysis, timeline, cfg=cfg,
        title=f"{analysis.count} reps"
              + (f", {np.mean(pulls):.2f}s mean pull" if pulls else "")
              + (f" · {full}/{graded} full range of motion" if graded else "")
              + f" · {reps.phase_label(cfg.REP_METRIC)}")
    console.print(f"  graph  -> [dim]{rel(timeline)}[/]")

    (run_dir / "summary.txt").write_text(reps.summary_text(
        analysis, video_seconds=info.duration, source=cfg.INPUT_VIDEO.name,
        metric=cfg.REP_METRIC))
    console.print(f"  summary-> [dim]{rel(run_dir / 'summary.txt')}[/]")

    total = time.perf_counter() - t_start
    cost = posed["cost"]
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp,
        "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info},
        "export": {"path": str(export_info.path), "width": export_info.width,
                   "height": export_info.height},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height,
                      "fps": info.fps, "n_frames": info.n_frames},
        "tonemap": {"mode": cfg.TONEMAP, "algorithm": cfg.TONEMAP_ALGORITHM,
                    "filter": tonemap, "reason": tonemap_reason,
                    "inference": bool(infer_tonemap)},
        "pose": {"model": cfg.POSE_MODEL, "extra_body": posed["extra_body"],
                 "frames_posed": n_posed},
        "calibration": profile.as_dict() if profile is not None else None,
        "reps": analysis.summary(),
        "cost_usd": {"pose": round(cost, 6), "total": round(cost, 6)},
        "gateway_seconds": {"pose": round(posed["seconds"], 2)} if posed["seconds"] else {},
        "render": {**stats, "seconds": round(render_seconds, 2), "credit": cfg.CREDIT},
    }, indent=2))

    console.print()
    kv({"pose": f"${cost:.4f}", "total": f"[bold]${cost:.4f}[/]"}, title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"reps": analysis.count, "full": full, "graded": graded,
            "grades": ["full" if r.complete else "partial" for r in analysis.reps],
            "cost": cost, "seconds": total, "output": run_dir}


def load_profile():
    """The calibration profile this run grades against, or None, said out loud either way."""
    if not cfg.GRADE_ROM:
        console.print("  [dim]range of motion not graded (GRADE_ROM = False)[/]")
        return None
    profile = calibrate.load(calibrate.path(cfg)) if cfg.USE_CALIBRATION else None
    if profile is not None:
        console.print(f"  calibrated against [green]{profile.source}[/] "
                      f"([dim]{profile.created}, {profile.reps} reps[/]): "
                      f"hang ≥ {profile.hang_required(cfg):.1f}°, "
                      f"clear ≥ {profile.clearance_required(cfg):.2f}, "
                      f"rep ≥ {profile.min_flexion(cfg):.0f}° of flexion")
    elif not cfg.USE_CALIBRATION:
        console.print("  [yellow]calibration off[/] (--uncalibrated); each clip is graded "
                      "against itself and the config.py defaults")
    else:
        console.print(f"  [yellow]no calibration profile[/] at {rel(calibrate.path(cfg))}; "
                      "each clip is graded against itself and the config.py defaults")
        console.print("  [dim]python calibrate.py path/to/a_full_set.mov measures yours[/]")
    return profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The per-run settings, as flags. Anything left out keeps its config.py value."""
    parser = argparse.ArgumentParser(
        description="Count pull-ups and grade each rep's range of motion as full or partial. "
                    "Flags override config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips, to run (config: {rel(cfg.INPUT)}); "
                             "a folder runs every video directly inside it")

    def height(text):
        return "full" if text.lower() in ("full", "source") else int(text)

    parser.add_argument("--height", type=height, metavar="PX|full",
                        help=f"output video height in px, or 'full' for the source's own; "
                             f"only ever scales down, and the model still sees "
                             f"{cfg.INFERENCE_HEIGHT}px, so no new gateway call "
                             f"(config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of each clip, for a quick test")
    parser.add_argument("--profile", type=Path, metavar="JSON",
                        help=f"the calibration profile to grade against "
                             f"(config: {rel(cfg.CALIBRATION_PROFILE)})")
    parser.add_argument("--uncalibrated", action="store_true",
                        help="ignore the calibration profile; grade each clip against "
                             "itself and the config.py defaults")
    parser.add_argument("--hang-tolerance", type=float, metavar="DEG",
                        help=f"degrees short of the reference dead hang that still count as "
                             f"full (config: {cfg.ROM_HANG_TOLERANCE_DEGREES:g})")
    parser.add_argument("--metric", choices=sorted(reps.METRICS),
                        help=f"the span the PULL time measures (config: {cfg.REP_METRIC})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the cached poses and call the model again")
    credit = parser.add_argument_group(
        "credit", "the lines in the panel's corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    if args.uncalibrated:
        cfg.USE_CALIBRATION = False
    if args.height:
        cfg.EXPORT_HEIGHT = None if args.height == "full" else args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.profile:
        cfg.CALIBRATION_PROFILE = args.profile.expanduser().resolve()
    if args.hang_tolerance is not None:
        cfg.ROM_HANG_TOLERANCE_DEGREES = args.hang_tolerance
    if args.metric:
        cfg.REP_METRIC = args.metric
    if args.fresh:
        cfg.REUSE_POSES = False

    def at(handle):
        return handle if handle.startswith("@") else f"@{handle}"

    handles = [(label, value) for label, value in
               (("LI", args.li), ("X", args.x and at(args.x)),
                ("IG", args.ig and at(args.ig))) if value]
    if args.no_credit:
        cfg.CREDIT = []
    elif handles:
        cfg.CREDIT = handles


def find_clips(inputs: list[Path]) -> list[Path]:
    """Every clip named, and every video directly inside every folder named, in order."""
    clips: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            clips += sorted(p for p in path.iterdir()
                            if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
        elif path.is_file():
            clips.append(path)
        else:
            console.print(f"[red]Not found:[/] {item}")
    return list(dict.fromkeys(clips))   # a clip named twice runs once


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_args(args)
    clips = find_clips(args.inputs or [cfg.INPUT])
    console.print()
    console.rule("[bold]Pull-ups: reps, pull time and range of motion[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a pull-up video in "
                      f"{rel(cfg.INPUT)}/, or pass one: python main.py path/to/clip.mov")
        return 1

    # ── 1. Key and calibration ───────────────────────────────────────────────
    rule(1, "API key and calibration")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key loaded from [green]{source}[/]")
    profile = load_profile()
    if len(clips) > 1:
        console.print(f"  {len(clips)} clips: " + ", ".join(c.name for c in clips))

    results: dict[str, dict | str] = {}
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            results[clip.name] = run_clip(api_key, profile)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # One bad clip does not cost the rest of the batch.
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            results[clip.name] = str(exc).splitlines()[0][:100]

    if len(clips) > 1:
        table = Table(title="batch", box=None, pad_edge=False, title_justify="left")
        for col in ("clip", "reps", "full ROM", "grades", "cost", "output"):
            table.add_column(col)
        for name, r in results.items():
            if isinstance(r, str):
                table.add_row(name, "[red]failed[/]", r, "", "", "")
            else:
                table.add_row(name, str(r["reps"]),
                              f"{r['full']}/{r['graded']}" if r["graded"] else "—",
                              ", ".join(r["grades"]), f"${r['cost']:.3f}",
                              str(rel(r["output"])))
        console.print(table)
    return 0 if all(not isinstance(r, str) for r in results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
