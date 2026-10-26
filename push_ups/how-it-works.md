# How it works

Code: [`src/reps.py`](src/reps.py) counts and times,
[`src/form.py`](src/form.py) measures and grades. Every threshold is in
[`config.py`](config.py).

## One side of the body

Filmed side-on, the far arm and leg hide behind the body and ViTPose guesses
them. So every measurement uses the near side, picked as whichever side's
shoulder, elbow, wrist, hip, knee and ankle score higher over the clip
(`--side` forces one). On the sample clip that is the right side, 0.95 against
0.84.

## Counting reps

- **Signal:** the shoulder's height above the wrist, in arm lengths. The hands
  stay planted, so a handheld camera doesn't read as movement, and dividing by
  the arm (upper arm plus forearm, median over the clip) cancels the camera's
  distance. About 1.1 locked out, about 0.3 with the chest on the floor.
- **Depth** is how far that has come down from the plank at the top (a high
  percentile of the clip, not its maximum).
- **Count:** armed at the top, confirmed once the shoulder comes down past half
  the typical depth, closed when it comes back near the top. Two thresholds, so
  noise near one can't count twice. Every rep counts, shallow ones included;
  grading is a separate step.

## Timing the concentric

Each rep is split into its phases off the shoulder's vertical speed: from the
fastest point of the push, walk back until the speed falls under 10% of it
(leaving the bottom), and forward until it does again (lockout). The descent is
found the same way.

- `moving` (default): leaving the bottom to lockout. A pause with the chest on
  the floor isn't timed.
- `ascent`: the deepest point to lockout, pause included.

The eccentric and the pause at the bottom are in `reps.json` and `summary.txt`
too.

## Down to the floor, back up

Both ends are read off the rep signal itself, at each rep's own bottom and
lockout. The wrist is on the floor, so the shoulder's height above it is how
close the chest got.

- **Down:** the shoulder has to come within 0.40 arm lengths of the wrist
  (`--down`). The shoulder joint can't reach the floor, since the chest is under
  it: chest-down reps on the sample clip bottom out at 0.22–0.34. For scale, a
  90° elbow reads about 0.5, so this is stricter than the usual elbow rule.
- **Up:** the shoulder has to get back to 90% of the plank height at the top of
  the clip (`--up`).
- Both pass and the rep is full range; either short and it's partial. An end
  the model couldn't see counts as short. Each reading is the extreme of a
  5-frame median, so one bad pose can't decide a rep.

## A straight back

The **body line** is the hip's angle off the straight line from shoulder to
ankle: 0° is a rigid plank, negative is the hips dropped too low (toward the
floor), positive is the hips raised too high. A rep keeps a straight back when
every frame from its descent to its lockout stays within ±15°
(`--back-tolerance`).

The keypoints don't sit exactly in the joints, so a visually straight plank
reads a few degrees hips-low. On the sample clip early reps peak around −4° to
−9°, and the last two reach −16° and −19° as the hips drop with fatigue. That
is the same point where the concentric slows from about 0.8 s to 1.1 s.

## On the video

Only what is measured, on the side facing the camera: the arm, and the body
line from shoulder through hip to ankle, green while straight and red once it
leaves the band, over a dashed line from shoulder to ankle that shows where a
straight body would be. The far side isn't drawn: it's hidden behind
the body and nothing reads it. `OVERLAY = "full"` in `config.py` draws the whole
skeleton instead.

## The panel

- **Header:** reps, average push-up time, full-range tally, straight-back tally.
- **Vertical displacement:** the shoulder's height over time, with the "up" and
  "floor" cutoffs. Two dots per rep mark where its concentric is timed from
  (leaving the bottom) and to (lockout), green, or red if that end missed its
  cutoff.
- **Body line vs. straight line:** the hip's angle off straight over time, with
  the ±15° band; the line turns red wherever it leaves it.
- **Push-up time per rep:** the concentric time, one dot per rep as it locks out,
  joined in order, red if that rep's back left the band, ringed if it fell short
  of full range. The dashed line is the running average.

## Limits

- One athlete, filmed side-on. Filmed from the front, the shoulder height and
  body line foreshorten and stop meaning anything.
- The floor isn't detected; "down" is judged by the shoulder's height above the
  wrist, not by contact.
- Range of motion and the body line only: nothing about hand placement or
  elbow flare.
