# How it works

Code: [`src/reps.py`](src/reps.py) counts, [`src/rom.py`](src/rom.py) grades,
[`src/calibrate.py`](src/calibrate.py) learns. Every threshold is in
[`config.py`](config.py).

## Counting reps

- **Signal:** the torso (mean of both shoulders and hips) measured against the
  hands on the bar, so a handheld camera doesn't register as movement. Smoothed,
  with the dead hang as zero. Frames with the hands off the bar are dropped.
- **Count:** armed at the bottom, confirmed once the body rises past a share of
  the typical rep, closed on the way back down. Two thresholds, so noise near
  one can't count twice.
- **Every pull counts**, short ones included. Grading is a separate step.
- **Pull time** (`--metric`): `moving` is onset to top (default), `ascent` is
  bottom to top, `total` is the whole cycle.

## Grading full or partial

Each rep is read at its own lowest and highest point, never a fixed height,
because both ends drift as a set goes on.

- **Bottom, a dead hang:** the elbow angle (shoulder–elbow–wrist) has to reach
  your straight arm, less `--hang-tolerance` (12.5°).
- **Top, clearing the bar:** the head has to rise above the hands by a share of
  what your good reps reached, in torso lengths, so camera distance doesn't
  matter.
- Both pass and it's full; either short and it's partial. An end the model
  couldn't see counts as short.
- **The head** is the nose, eyes and ears averaged, weighted by the model's
  confidence. From behind it leans on the ears, from the front on the face.

## Why calibrate

A straight arm doesn't read 180°: ViTPose places the shoulder on the deltoid,
so a dead hang reads about 167°, and that varies by person and camera. So
`calibrate.py` measures it from a set of full reps:

| learned | on the sample clip | cutoff |
|---|---|---|
| straight arm | 167.5° | full hang at ≥ 155° |
| head clearance | 0.42 torso lengths | clears the bar at ≥ 0.17 (40%) |
| elbow sweep | 132° | counts as a rep at ≥ 46° of flexion (35%) |

The sweep also catches half reps. Without a profile, a pull that stops halfway
can fall under the height gate and never be counted. With one, the gate drops
and a rep must also bend the elbow by a share of a full sweep.

Without a profile, each clip is its own reference: its deepest hang sets the
cutoff, with 150° as a floor. That works, but a clip whose best hang is unusually
deep holds every other rep to a higher bar.

## The panel

- **Header:** reps, average pull (full reps only), full-ROM tally, live elbow
  angle.
- **Graph:** the head above the hands over time. The "clears bar" line is the
  top cutoff itself, so each peak lands on the same side as its grade. Dots mark
  each rep's bottom and top, green or red.
- **Table:** grade, bottom and top elbow angle, pull time. A red number is the
  end that fell short; a partial rep has no pull time.

## Limits

- One athlete; a second person in frame is ignored.
- A profile belongs to one athlete and camera position; re-run `calibrate.py`
  for either.
- Range of motion only: no kipping check, nothing about the legs.
