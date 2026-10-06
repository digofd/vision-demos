# pull_ups

Counts pull-up reps from a clip, times each pull, and grades every rep's range
of motion as **full** or **partial**: a dead hang at the bottom, and the head
clearing the bar at the top. Pose runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large)
through the [VLM Run Gateway](https://www.vlm.run/gateway), so there are no
weights to download. How it works: [how-it-works.md](how-it-works.md).

<p align="center">
  <img src="readme_images/pull_ups_demo_thumbnail.jpg" width="600" alt="A six-rep pull-up set with a pose overlay on the left; on the right the rep count, 3 of 6 full range of motion, vertical displacement over time against the dead hang and the bar, and each rep's grade, bottom and top elbow angle, and pull time">
</p>

## Run it

1. **Get an API key** at [app.vlm.run/sign-in](https://app.vlm.run/sign-in).

2. **Set it** in a `.env` at the repo root:

   ```bash
   cp ../.env.example ../.env
   # paste your key after VLMRUN_API_KEY=
   ```

3. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate pull_ups
   ```

4. **Add pull-up clips** to `data/input/`, filmed from in front or behind with
   the whole body and the bar in shot.

5. **Run it** from this directory:

   ```bash
   python main.py                                   # every clip in data/input/
   python main.py data/input/my_set.mov             # just this one
   python main.py --li "Your Name" --x @yourhandle  # with your credit in the corner
   ```

   Flags override their [`config.py`](config.py) value for that run only:

   | flag | what it sets |
   |---|---|
   | `CLIP` or `DIR` | clips, or folders of clips, to run (`INPUT`) |
   | `--uncalibrated` / `--profile PATH` | skip calibration, or grade against another profile |
   | `--hang-tolerance 12.5` | degrees under your dead hang that still count as full |
   | `--metric moving` | what the pull time measures: `moving`, `ascent` or `total` |
   | `--height 1080` | output height in px, or `full`; no new model call |
   | `--trim 5` | only the first 5 s of each clip, for a quick test |
   | `--fresh` | ignore cached poses and call the model again |
   | `--li` `--x` `--ig` / `--no-credit` | the credit in the panel's corner |

   `python main.py -h` lists them. Everything else stays in `config.py`.

6. **Optional: calibrate** on a set where every rep was full:

   ```bash
   python calibrate.py data/input/my_full_set.mov
   ```

   Without it, each clip is graded against its own deepest hang and a default
   bar clearance, and a pull that stops halfway may not be counted at all.
   With it, reps are graded against your own dead hang and clearance, and half
   reps are counted and marked partial. Later runs use it automatically.

## Output

One timestamped directory per clip under `data/output/`:

```
20261006-000912/
├── <clip>_pull_ups.mp4   # overlay + live panel
├── timeline.png          # displacement, elbow angle and head clearance, every rep marked
├── reps.json             # per-rep timings and grades, plus every per-frame series
├── summary.txt           # the rep report
└── run.json              # config, calibration, provenance, cost
```

The profile is saved to `data/calibration.json`. Gateway replies are cached in
`data/cache/`, so re-grading a clip costs nothing, and `data/` is gitignored.
`run.json` records what each run cost.

## License

[Apache-2.0](../LICENSE).
