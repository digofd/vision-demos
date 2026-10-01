# deadlift

Counts deadlift reps from a side-on clip, times each pull off the barbell
plate, and calls each rep's back **straight** or **rounded**, with the
probability live beside the video. Three models, one
[VLM Run Gateway](https://www.vlm.run/gateway), no weights to download:

| What | Model |
|---|---|
| Pose, hip angle, and the crop the back is read in | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| The plate: bar height and reps | [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) |
| Back position, every frame | [`gemma-4-26b-a4b-it`](https://vlm.run/gateway/models/google-gemma-4-26b-a4b-it) |

<p align="center">
  <a href="https://www.youtube.com/watch?v=917J5oOvcBM"><img src="https://img.youtube.com/vi/917J5oOvcBM/maxresdefault.jpg" width="600" alt="A four-rep deadlift set with pose and plate overlay on the left and the live back verdict, bar height and per-rep results on the right. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=917J5oOvcBM">▶ Watch on YouTube</a>
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
   conda activate deadlift
   ```

4. **Add side-on deadlift clips** to `data/input/`. Film side-on, with the whole
   lifter and the near plate in shot.

5. **Run it** from this directory:

   ```bash
   python main.py                                   # every clip in data/input/
   python main.py data/input/my_lift.mov            # just this one
   python main.py --li "Your Name" --x @yourhandle  # with your credit in the corner
   ```

   The settings that change from run to run are flags, and each overrides its
   [`config.py`](config.py) value for that run only:

   | flag | what it sets |
   |---|---|
   | `CLIP` or `DIR` | clips, or folders of clips, to run (`INPUT`; default `data/input/`) |
   | `--reads-per-second 10` | System One reads per second, the main cost (`SAMPLE_FPS`; default every frame) |
   | `--plate-cm 45` | plate diameter, the ruler for bar height (`PLATE_DIAMETER_CM`) |
   | `--rep-source hinge` | count reps off the hip angle instead of the plate (`REP_SOURCE`) |
   | `--model ID` | the System One model that judges the back (`READ_MODEL`) |
   | `--threshold 0.5` | P(rounded) above which a rep is called rounded (`THRESHOLD`) |
   | `--height 1080` | output video height in px, or `full`; the models still see `INFERENCE_HEIGHT`, so no new calls (`EXPORT_HEIGHT`) |
   | `--trim 5` | only the first 5 s of each clip, for a quick test (`TRIM_SECONDS`) |
   | `--fresh` | ignore cached replies and call every model again |
   | `--li` `--x` `--ig` / `--no-credit` | the credit in the panel's corner (`CREDIT`) |

   `python main.py -h` lists them. Everything else stays in `config.py`.

`run.json` in each output folder records what the run cost, per model; the back
reads are most of it.

## How it works

- **Bar height and reps.** SAM 3.1 tracks the plate, and its own size is the
  ruler, so height comes out in meters. This assumes an Olympic plate, 45 cm
  across (`PLATE_DIAMETER_CM`). A rep is counted with the same hysteresis as
  [chin_ups](../chin_ups).
- **Hip hinge.** The shoulder–hip–knee angle from ViTPose, drawn on the lifter.
  It cross-checks the plate's reps and takes over if the plate track fails
  (`REP_SOURCE`).
- **Back position.** Every frame, cropped to the lifter, is one System One
  `choice` question. The answer comes back as exactly two probabilities,
  `{"straight": 0.05, "rounded": 0.95}`, with no text to parse. The question
  asks about the spine's *shape*, not how far the torso leans, since a flat
  back leaning over the bar otherwise reads as rounded.
- **Each rep's verdict** is the mean P(rounded) over its first pull, from the
  bar leaving the floor to about the knee, where rounding shows.
- **HDR clips.** The displayed video is tone-mapped (`TONEMAP`, `auto` by
  default). The models see that tone-mapped video only when `TONEMAP_INFERENCE`
  is on; otherwise they see the plain conversion. Tone-mapping needs an ffmpeg
  with `libplacebo`, which the conda environment has; without it the run warns
  and carries on untoned. On macOS, libplacebo also needs a Vulkan driver
  (`conda install -c conda-forge moltenvk`); without it an HDR conversion fails.

Every setting, prompt included, is in [`config.py`](config.py).

## Output

One timestamped directory per run under `data/output/`:

```
20260929-134948/
├── <clip>_deadlift.mp4   # overlay + live panel
├── timeline.png          # each rep's timed pull on bar height, hip angle, P(rounded)
├── reps.json             # per-rep timings and verdicts
├── reads.json            # every System One read
├── summary.txt           # the rep report
└── run.json              # config, provenance, cost
```

Gateway replies are cached in `data/cache/`, so a re-render costs nothing, and
`data/` is gitignored.

## License

[Apache-2.0](../LICENSE).
