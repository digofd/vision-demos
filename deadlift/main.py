"""Track a deadlift: pose, the plate, the reps, and whether the back was straight or rounded.

    conda activate deadlift
    python main.py                        # every clip in data/input/
    python main.py data/input/lift.mov    # just this one (clips or folders)
    python main.py -h                     # every flag

Three models, one gateway:
    ViTPose     the lifter's pose on every frame, and the box System One reads
    SAM 3.1     the plate tracked through the clip, for bar height and reps
    System One  P(rounded) per frame, and a verdict per rep
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import classify, plate as plate_mod, pose, render, reps, video
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


def _plate_fallback() -> str:
    """What a missing plate means, which depends on how the reps were asked for."""
    if cfg.REP_SOURCE == "plate":
        return 'REP_SOURCE is "plate", so nothing is counted'
    return "reps fall back to the hip hinge"


def track_plate(client, mp4: Path, video_b64: str, plan, *, fps: float) -> tuple[dict, dict]:
    """SAM 3.1 over the clip, one call per segment, fused into one reply."""
    parts, usages = [], []
    for n, (start, length) in enumerate(plan):
        clip = mp4
        if len(plan) > 1:
            clip = cfg.CACHE_DIR / f"{mp4.stem}.seg{n}_{start}_{length}.mp4"
            if not clip.is_file():
                video.segment(mp4, clip, start_frame=start, n_frames=length, fps=fps)
        with console.status(f"[cyan]waiting on the gateway[/]; SAM is tracking the plate "
                            f"({n + 1}/{len(plan)})…", spinner="dots"):
            part, usage = plate_mod.request_track(
                client, model=cfg.PLATE_MODEL,
                video_b64=video_b64 if clip is mp4 else base64.b64encode(
                    clip.read_bytes()).decode("ascii"),
                prompt=cfg.PLATE_PROMPT, skip_frames=cfg.PLATE_TRACK_STRIDE,
                max_frames=cfg.PLATE_TRACK_MAX_FRAMES)
        # Each segment numbers its frames and tracks from scratch.
        parts.append(plate_mod.shift(part, frame_offset=start, track_offset=n * 1000))
        usages.append(usage or {})
    return plate_mod.concat(parts), {"cost": sum(u.get("cost") or 0.0 for u in usages)}


def run_clip(api_key: str) -> dict:
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
    # Decided before the cache key is built, so switching tone-mapping on or off
    # produces a different MP4 rather than silently reusing the other one.
    tonemap, tonemap_reason = video.tonemap_filter(
        src_info, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM,
    )
    infer_tonemap = tonemap if cfg.TONEMAP_INFERENCE else None
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{tonemap_reason}"
                  + ("" if not tonemap or infer_tonemap else
                     "  [dim](export only; the models see the plain conversion)[/]"))

    def prepare(height, label: str, tonemap: str | None) -> Path:
        """Convert to *height* (None = source), reusing a cached copy if present."""
        path = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=height,
                                trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                                tonemap=tonemap)
        if path.is_file() and not cfg.FORCE_RECONVERT:
            console.print(f"  [green]cache hit[/] ({label}): [dim]{rel(path)}[/]")
        else:
            console.print(f"  [yellow]converting[/] ({label})")
            video.convert(cfg.INPUT_VIDEO, path, target_height=height,
                          trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                          total_frames=src_info["n_frames"], console=console,
                          tonemap=tonemap)
        return path

    mp4 = prepare(cfg.INFERENCE_HEIGHT, "inference", infer_tonemap)
    info = video.inspect(mp4)
    console.print(f"  inference [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")
    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT and tonemap == infer_tonemap:
        export_info = info
    else:
        export_info = video.inspect(prepare(cfg.EXPORT_HEIGHT, "export", tonemap))
        console.print(f"  export    [bold]{export_info}[/]")
        if export_info.n_frames != info.n_frames:
            console.print(f"  [yellow]warning[/] frame counts differ ({info.n_frames} vs "
                          f"{export_info.n_frames}); the overlay is matched by frame index")

    from openai import OpenAI

    client = OpenAI(base_url=cfg.OPENAI_BASE_URL, api_key=api_key,
                    timeout=cfg.VIDEO_REQUEST_TIMEOUT, max_retries=1)
    costs: dict[str, float] = {}
    seconds: dict[str, float] = {}

    # ── 4. Pose: ViTPose ─────────────────────────────────────────────────────
    rule(4, "Pose (ViTPose)")
    video_b64, extra_body = pose.build_request(
        mp4, every_frame=cfg.EVERY_FRAME, fps=info.fps, n_frames=info.n_frames,
        video_fps=cfg.VIDEO_FPS, video_max_frames=cfg.VIDEO_MAX_FRAMES, precision=cfg.PRECISION)
    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL, extra_body=extra_body)
    cached = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None
    if cached is not None:
        payload, usage, _, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/]; no gateway call")
    else:
        t0 = time.perf_counter()
        with console.status("[cyan]waiting on the gateway[/]; ViTPose on every frame…",
                            spinner="dots"):
            payload, usage = pose.request_poses(client, model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        seconds["pose"] = time.perf_counter() - t0
        pose.save_cache(poses_cache, payload, usage, {}, stamp=stamp_now())
        console.print(f"  [green]done in {seconds['pose']:.1f}s[/]")
    # What this run spent: a cached reply was paid for by the run that made it.
    costs["pose"] = 0.0 if cached is not None else float((usage or {}).get("cost") or 0)

    frames, by_index = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    by_index = {f["index"]: f["persons"] for f in frames}
    n_posed = sum(1 for f in frames if f["persons"])
    kv({"model": cfg.POSE_MODEL,
        "frames with a person": f"{n_posed} of {info.n_frames}",
        "duplicates dropped": dropped,
        "cost": f"${costs['pose']:.4f}"}, title="pose")

    # ── 5. Plate: SAM 3.1 ────────────────────────────────────────────────────
    rule(5, "Plate (SAM 3.1)")
    plan = plate_mod.segment_plan(info.n_frames, cfg.PLATE_TRACK_STRIDE,
                                  max_frames=cfg.PLATE_TRACK_MAX_FRAMES)
    plate_settings = {"stride": cfg.PLATE_TRACK_STRIDE, "max_frames": cfg.PLATE_TRACK_MAX_FRAMES,
                      "segments": len(plan)}
    plate_cache = plate_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.PLATE_MODEL,
                                       prompt=cfg.PLATE_PROMPT, settings=plate_settings)
    cached = plate_mod.load_cache(plate_cache) if cfg.REUSE_PLATE else None
    payload, usage = None, None
    if cfg.REP_SOURCE == "hinge":
        console.print("  [dim]skipped: REP_SOURCE = \"hinge\"[/]")
    elif cached is not None:
        payload, usage, created = cached
        console.print(f"  [green]cache hit[/]: plate track from [dim]{created}[/]; no gateway call")
    else:
        try:
            t0 = time.perf_counter()
            payload, usage = track_plate(client, mp4, video_b64, plan, fps=info.fps)
            costs["plate"] = float((usage or {}).get("cost") or 0)
            seconds["plate"] = time.perf_counter() - t0
            plate_mod.save_cache(plate_cache, payload, usage, stamp=stamp_now())
            console.print(f"  [green]done in {seconds['plate']:.1f}s[/]")
        except Exception as exc:
            # Not fatal on "auto": the hip hinge counts the reps without it.
            # "plate" was asked for explicitly, so an empty count is the honest result.
            console.print(f"  [yellow]plate tracking failed[/] "
                          f"({str(exc).splitlines()[0][:120]}); {_plate_fallback()}")
    costs.setdefault("plate", 0.0)

    plate = None
    if payload is not None:
        try:
            plate = plate_mod.select(payload, info.n_frames, min_score=cfg.PLATE_MIN_SCORE,
                                     min_area_fraction=cfg.PLATE_MIN_AREA_FRACTION,
                                     size_tolerance=cfg.PLATE_SIZE_TOLERANCE)
        except Exception as exc:
            console.print(f"  [yellow]plate track unusable[/] "
                          f"({str(exc).splitlines()[0][:120]}); {_plate_fallback()}")
            plate = None
        else:
            if not plate.n_found:
                console.print(f"  [yellow]no plate found[/] for the prompt {cfg.PLATE_PROMPT!r}; "
                              f"{_plate_fallback()}")
                plate = None
    plate_px = plate.median_height * info.height if plate is not None else None
    if plate is not None:
        kv({"model": cfg.PLATE_MODEL,
            "prompt": cfg.PLATE_PROMPT,
            "sampled": f"every {cfg.PLATE_TRACK_STRIDE} frames ({len(plate.sampled)} looks, "
                       f"{len(plan)} call{'s' if len(plan) > 1 else ''})",
            "plate found": f"{plate.n_found} of {len(plate.sampled)} samples "
                           f"(SAM track ids {plate.track_ids})",
            "ruler": f"plate is {plate.median_height * info.height:.0f}px = {cfg.PLATE_DIAMETER_CM:g} cm "
                     f"[dim](assumed Olympic plate, PLATE_DIAMETER_CM)[/]",
            "cost": f"${costs['plate']:.4f}"}, title="plate")

    # ── 6. Reps ──────────────────────────────────────────────────────────────
    rule(6, "Reps")
    analysis = reps.analyze(plate, by_index, info.n_frames, info.fps,
                            width=info.width, height=info.height, cfg=cfg)
    agreement = analysis.agreement(cfg.HINGE_MATCH_OVERLAP)
    size = (f"typical bar travel {analysis.amplitude / 100:.2f} m" if analysis.source == "plate"
            else f"typical hip extension {analysis.amplitude:.0f}°")
    console.print(f"  [bold green]{analysis.count} reps[/] off the [bold]{analysis.source}[/], {size}")
    onset, end = agreement.get("mean_onset_offset_s"), agreement.get("mean_end_offset_s")
    console.print(f"  cross-check: plate {agreement['plate_reps']} reps, hip hinge "
                  f"{agreement['hinge_reps']}, [bold]{agreement['matched']} matched[/]"
                  + (f" [dim](hinge start {onset:+.2f}s, end {end:+.2f}s from the plate's)[/]"
                     if onset is not None else ""))

    # ── 7. Back position: System One ─────────────────────────────────────────
    rule(7, "Back position (System One)")
    indices = classify.sample_indices(info.n_frames, info.fps, cfg.SAMPLE_FPS)
    crops = classify.person_crops(by_index, indices, pad=cfg.PERSON_CROP_PAD) \
        if cfg.CROP_TO_PERSON else {}
    settings = classify.request_settings(cfg)
    kv({
        "model": cfg.READ_MODEL,
        "endpoint": f"{cfg.TYPESAFE_BASE_URL}/v1/systemone",
        "question": f"choice over {list(cfg.LABELS)}",
        "reads": f"{len(indices)} frames ({cfg.SAMPLE_FPS or info.fps:g}/s of {info.fps:.0f} fps)",
        "image": (f"ViTPose person box +{cfg.PERSON_CROP_PAD:.0%}" if crops else "whole frame")
                 + f", {cfg.IMAGE_LONG_EDGE}px long edge, detail={cfg.IMAGE_DETAIL}",
        "concurrency": cfg.CONCURRENCY,
    }, title="request")

    reads_cache = classify.cache_path(mp4, cfg.CACHE_DIR, settings=settings,
                                      crop_source=poses_cache.name if crops else None)
    read_list = classify.load_cache(reads_cache) if cfg.REUSE_READS else None
    asked: list = []      # the reads made this run, which are all it spent
    retry = [r.index for r in read_list if not r.ok] if read_list is not None else []
    if read_list is not None and not retry:
        console.print(f"  [green]cache hit[/]: [dim]{reads_cache.name}[/]; no gateway call")
    elif read_list is not None:
        # A cached failure is a gap, not an answer: ask those frames again.
        console.print(f"  [green]cache hit[/]: [dim]{reads_cache.name}[/]; "
                      f"retrying [yellow]{len(retry)}[/] failed read{'s' if len(retry) > 1 else ''}")
        images = classify.extract_images(mp4, retry, cfg=cfg, crops=crops)
        t0 = time.perf_counter()
        fresh = classify.read_frames(images, info.fps, api_key=api_key, cfg=cfg, console=console)
        asked = fresh
        seconds["reads"] = time.perf_counter() - t0
        by_frame = {r.index: r for r in read_list} | {r.index: r for r in fresh}
        read_list = [by_frame[i] for i in sorted(by_frame)]
        classify.save_cache(reads_cache, read_list, stamp=stamp_now())
    else:
        images = classify.extract_images(mp4, indices, cfg=cfg, crops=crops)
        t0 = time.perf_counter()
        read_list = classify.read_frames(images, info.fps, api_key=api_key, cfg=cfg,
                                         console=console)
        asked = read_list
        seconds["reads"] = time.perf_counter() - t0
        classify.save_cache(reads_cache, read_list, stamp=stamp_now())

    ok = [r for r in read_list if r.ok]
    failed = [r for r in read_list if not r.ok]
    latencies = np.array([r.latency for r in ok]) if ok else np.array([np.nan])
    costs["reads"] = sum(r.usage.get("cost") or 0 for r in asked if r.ok)
    kv({
        "answered": f"{len(ok)}/{len(read_list)}"
                    + (f"  [red]{len(failed)} failed[/]: {failed[0].error.splitlines()[0][:160]}" if failed else ""),
        "wall clock": f"{seconds['reads']:.1f}s" if "reads" in seconds else "cached",
        "latency": f"p50 {np.median(latencies) * 1e3:.0f} ms, "
                   f"p95 {np.percentile(latencies, 95) * 1e3:.0f} ms",
        "cost": (f"${costs['reads']:.5f} (${costs['reads'] / max(len(asked), 1):.6f} / read)"
                 if asked else "$0 (cached)"),
    }, title="response")

    signal = classify.build_signal(read_list, info.n_frames, info.fps, cfg=cfg)
    disagree = reps.judge_backs(analysis, signal, cfg=cfg)
    if disagree:
        console.print(f"  [yellow]warning[/] rep {', '.join(map(str, disagree))}: the verdict "
                      f"disagrees with the panel's curve (BACK_SMOOTH_SECONDS "
                      f"{cfg.BACK_SMOOTH_SECONDS:g}s vs PANEL_P_SMOOTH_SECONDS "
                      f"{cfg.PANEL_P_SMOOTH_SECONDS:g}s); the video will look wrong there")
    if analysis.reps:
        table = Table(box=None, pad_edge=False)
        for col in ("rep", "back", f"P({cfg.POSITIVE_LABEL})", "onset", "lockout", "lift",
                    "lower", "travel", "hip"):
            table.add_column(col, justify="right", style="dim" if col == "rep" else None)
        for r in analysis.reps:
            style = "red" if r.back_label == cfg.POSITIVE_LABEL else "green"
            table.add_row(str(r.number), f"[{style}]{r.back_label}[/]", f"{r.back_p:.0%}",
                          f"{r.onset_time:.2f}s", f"{r.peak_time:.2f}s",
                          f"{r.lift_seconds:.2f}s", f"{r.lower_seconds:.2f}s",
                          "—" if np.isnan(r.height_cm) else f"{r.height_cm / 100:.2f} m",
                          "—" if np.isnan(r.hip_deg) else f"{r.hip_deg:.0f}°")
        console.print(Panel(table, title=f"[bold green]{analysis.count} reps[/] "
                                         f"[dim](off the {analysis.source})[/]",
                            title_align="left", expand=False))
    else:
        console.print("  [yellow]no reps detected[/]. Check timeline.png")

    # ── 8. Output ────────────────────────────────────────────────────────────
    rule(8, "Render")
    # One directory per clip. A batch can finish two clips in the same second, and
    # the shared files (reps.json, summary.txt, run.json) would otherwise overwrite.
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{n}"
        n += 1
    stamp = run_dir.name
    run_dir.mkdir(parents=True)

    (run_dir / "reads.json").write_text(json.dumps({
        "fps": info.fps, "settings": settings,
        "crops": {str(k): [round(v, 4) for v in box] for k, box in crops.items()},
        "reads": [asdict(r) for r in read_list],
        "series": {"t": np.round(np.arange(info.n_frames) / info.fps, 4).tolist(),
                   f"p_{cfg.POSITIVE_LABEL}": np.round(signal.per_frame, 4).tolist()},
    }, indent=2))
    def series(x: np.ndarray, digits: int) -> list:
        """JSON has no NaN: a frame with no value is null."""
        return [None if not np.isfinite(v) else round(float(v), digits) for v in x]

    (run_dir / "reps.json").write_text(json.dumps({
        "fps": info.fps, **analysis.summary(), "agreement": agreement,
        "series": {"height_cm": series(analysis.height_cm, 2),
                   "hip_deg": series(analysis.hip_deg, 1)},
    }, indent=2))

    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(export_info, by_index=by_index, plate=plate, signal=signal, analysis=analysis,
                          crops=crops, dst=raw, cfg=cfg, console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_deadlift.mp4"
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    raw.unlink(missing_ok=True)   # MPEG-4 Part 2 intermediate; no player wants it
    console.print(f"  video  -> [dim]{rel(out_video)}[/] "
                  f"({out_video.stat().st_size / 1e6:.1f} MB, {render_seconds:.1f}s)")

    timeline = run_dir / "timeline.png"
    render.plot_timeline(signal, read_list, analysis, timeline, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(timeline)}[/]")

    summary = reps.summary_text(analysis, video_seconds=info.duration,
                                source=cfg.INPUT_VIDEO.name, plate_cm=cfg.PLATE_DIAMETER_CM,
                                plate_px=plate_px, agreement=agreement)
    (run_dir / "summary.txt").write_text(summary)
    console.print(f"  summary-> [dim]{rel(run_dir / 'summary.txt')}[/]")

    total = time.perf_counter() - t_start
    cost = sum(costs.values())
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
        "pose": {"model": cfg.POSE_MODEL, "extra_body": extra_body, "frames_posed": n_posed},
        "plate": None if plate is None else {
            "model": cfg.PLATE_MODEL, "prompt": cfg.PLATE_PROMPT, **plate_settings,
            "found": plate.n_found, "samples": len(plate.sampled),
            "track_ids": plate.track_ids,
            "ruler": {"assumed_diameter_cm": cfg.PLATE_DIAMETER_CM,
                      "plate_px": round(plate_px, 1),
                      "px_per_cm": round(plate_px / cfg.PLATE_DIAMETER_CM, 3)}},
        "rep_source": analysis.source,
        "agreement": agreement,
        "reads": {**settings, "answered": len(ok), "failed": len(failed),
                  "latency_p50_ms": round(float(np.median(latencies)) * 1e3, 1)},
        "reps": analysis.summary(),
        "cost_usd": {k: round(v, 6) for k, v in costs.items()} | {"total": round(cost, 6)},
        "gateway_seconds": {k: round(v, 2) for k, v in seconds.items()},
        "render": {**stats, "seconds": round(render_seconds, 2)},
    }, indent=2))

    console.print()
    kv({k: f"${v:.4f}" for k, v in costs.items()} | {"total": f"[bold]${cost:.4f}[/]"},
       title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"reps": analysis.count, "source": analysis.source,
            "verdicts": [f"{r.back_label} ({r.back_p:.0%})" for r in analysis.reps],
            "cost": cost, "seconds": total, "output": run_dir}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The per-run settings, as flags. Anything left out keeps its config.py value."""
    parser = argparse.ArgumentParser(
        description="Count deadlift reps and judge each rep's back, from side-on clips. "
                    "Flags override config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips, to run (config: {rel(cfg.INPUT)}); "
                             "a folder runs every video directly inside it")

    def height(text):
        return "full" if text.lower() in ("full", "source") else int(text)

    parser.add_argument("--height", type=height, metavar="PX|full",
                        help=f"output video height in px, or 'full' for the source's own; "
                             f"only ever scales down, and the models still see "
                             f"{cfg.INFERENCE_HEIGHT}px, so no new gateway call "
                             f"(config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of each clip, for a quick test")
    parser.add_argument("--plate-cm", type=float, metavar="CM",
                        help=f"plate diameter, the ruler for bar height "
                             f"(config: {cfg.PLATE_DIAMETER_CM:g}, an Olympic plate)")
    parser.add_argument("--rep-source", choices=("auto", "plate", "hinge"),
                        help=f"what the reps are counted off; 'hinge' skips the plate "
                             f"tracking (config: {cfg.REP_SOURCE})")
    parser.add_argument("--model", metavar="ID",
                        help=f"the System One model that judges the back "
                             f"(config: {cfg.READ_MODEL})")
    parser.add_argument("--reads-per-second", type=float, metavar="N",
                        help="System One reads per second of video, the main cost; frames "
                             "between reads are interpolated (config: every frame)")
    parser.add_argument("--threshold", type=float, metavar="P",
                        help=f"P({cfg.POSITIVE_LABEL}) above this calls a rep "
                             f"{cfg.POSITIVE_LABEL} (config: {cfg.THRESHOLD:g})")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore cached replies and call every model again")
    credit = parser.add_argument_group(
        "credit", "the lines in the panel's corner; any of these replaces config.py's CREDIT")
    credit.add_argument("--li", metavar="TEXT", help='LinkedIn, e.g. "Jeremy Park, PhD"')
    credit.add_argument("--x", metavar="HANDLE", help="X handle, e.g. @jeremyparkphd")
    credit.add_argument("--ig", metavar="HANDLE", help="Instagram handle")
    credit.add_argument("--no-credit", action="store_true", help="no credit at all")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Write the flags over the config, before anything reads it."""
    if args.height:
        cfg.EXPORT_HEIGHT = None if args.height == "full" else args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.plate_cm:
        cfg.PLATE_DIAMETER_CM = args.plate_cm
    if args.rep_source:
        cfg.REP_SOURCE = args.rep_source
    if args.model:
        cfg.READ_MODEL = args.model
    if args.reads_per_second:
        cfg.SAMPLE_FPS = args.reads_per_second
    if args.threshold is not None:
        cfg.THRESHOLD = args.threshold
    if args.fresh:
        cfg.REUSE_POSES = cfg.REUSE_PLATE = cfg.REUSE_READS = False

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
    console.rule("[bold]Deadlift: pose, plate, reps and back position[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a side-on deadlift video in "
                      f"{rel(cfg.INPUT)}/, or pass one: python main.py path/to/clip.mov")
        return 1

    # ── 1. Key ───────────────────────────────────────────────────────────────
    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  loaded from [green]{source}[/]")
    if len(clips) > 1:
        console.print(f"  {len(clips)} clips: " + ", ".join(c.name for c in clips))

    results: dict[str, dict | str] = {}
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            results[clip.name] = run_clip(api_key)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # One bad clip does not cost the rest of the batch.
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            results[clip.name] = str(exc).splitlines()[0][:100]

    if len(clips) > 1:
        table = Table(title="batch", box=None, pad_edge=False, title_justify="left")
        for col in ("clip", "reps", f"backs (P {cfg.POSITIVE_LABEL})", "cost", "output"):
            table.add_column(col)
        for name, r in results.items():
            if isinstance(r, str):
                table.add_row(name, "[red]failed[/]", r, "", "")
            else:
                table.add_row(name, f"{r['reps']} ({r['source']})", ", ".join(r["verdicts"]),
                              f"${r['cost']:.3f}", str(rel(r["output"])))
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
