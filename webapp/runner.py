"""Run one vendored demo pipeline in its own process.

    python webapp/runner.py <demo_dir> <overrides.json>

The demos are driven entirely by module-level constants in their own
`config.py` and have no command-line flags, by design. This bridges that to a
web UI the only way that does not require editing them: import `config`, set
attributes on the module object, and only then import `main`, which does
`import config as cfg` and so picks up the same, already-mutated module.

A separate process per run, rather than importing them into the Gradio app:

* Every demo defines modules named `config`, `main` and `src`. Importing two
  of them into one interpreter collides on every one of those names.
* The pipelines are written as a script that ends in `sys.exit()`, and they
  mutate their config module. A crash or a stray exit takes the process with it,
  not the web app.
* It gives the app a clean place to stream stdout from, and a hard kill for a
  run that overruns.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Not config constants: several clips to run one after another, each set on
# the named constant in turn. For the demos whose pipeline takes one clip per
# run; the order given is the order they run in.
CLIPS_KEY, CLIP_TARGET_KEY = "__CLIPS__", "__CLIP_KEY__"

# Keys whose values are paths. JSON has no path type, so they are carried
# separately rather than guessed at from the string.
PATH_KEYS = frozenset({
    "INPUT", "INPUT_VIDEO", "INPUT_DIR", "OUTPUT_DIR", "CACHE_DIR", "DATA_DIR",
})


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: runner.py <demo_dir> <overrides.json>", file=sys.stderr)
        return 2

    demo_dir = Path(sys.argv[1]).resolve()
    overrides = json.loads(Path(sys.argv[2]).read_text())

    if not (demo_dir / "main.py").is_file():
        print(f"error: no main.py in {demo_dir}", file=sys.stderr)
        return 2

    # The demos import `config` and `src.*` as top-level names, so the demo
    # directory has to lead sys.path. cwd follows it because a few of them
    # resolve incidental paths against it.
    sys.path.insert(0, str(demo_dir))
    os.chdir(demo_dir)

    import config as cfg

    clips = overrides.pop(CLIPS_KEY, None)
    clip_key = overrides.pop(CLIP_TARGET_KEY, "INPUT_VIDEO")

    applied = []
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            # Loud rather than silent: a renamed constant upstream should not
            # look like a setting that quietly stopped working.
            print(f"  [runner] warning: {key} is not in {demo_dir.name}/config.py, ignored",
                  file=sys.stderr)
            continue
        if key in PATH_KEYS and value is not None:
            value = Path(value)
        elif isinstance(value, list):
            # The configs use tuples throughout; JSON only has arrays.
            value = tuple(value)
        setattr(cfg, key, value)
        applied.append(key)

    print(f"  [runner] {demo_dir.name}: {len(applied)} config overrides applied")

    # Imported only now, so `import config as cfg` inside it binds the module
    # this has already rewritten.
    import main as demo_main

    # An explicit empty argv: left to itself, each pipeline's argparse would
    # read this script's own arguments, and take the demo directory and the
    # overrides file for clips to run (or refuse them as unrecognized).
    if not clips:
        return demo_main.main([])

    # One bad clip does not cost the others, but still fails the run.
    worst = 0
    for n, clip in enumerate(clips, 1):
        print(f"\n  [runner] clip {n}/{len(clips)}: {Path(clip).name}", flush=True)
        setattr(cfg, clip_key, Path(clip))
        code = demo_main.main([])
        worst = worst or code
    return worst


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[runner] interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:
        # The full traceback, not just the message: this is the only place a
        # failure inside a pipeline is visible to the app.
        traceback.print_exc()
        sys.exit(1)
