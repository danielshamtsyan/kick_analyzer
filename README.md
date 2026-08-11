# Roundhouse Kick Analyzer — Validation Prototype

This is a rough, working prototype to test two things before you build anything real:

1. **Track 1 — Is the feedback accurate?** Does the coaching cue it generates match
   what a training partner or coach would actually say?
2. **Track 2 — Is phone footage reliable enough?** Or do you need to require slo-mo
   capture, which adds friction to onboarding?

It analyzes one kicking side, one moment (the peak of the kick), and four metrics:
hip rotation, standing-leg bend, torso lean, and guard-hand position. That's it —
deliberately narrow, per the "one technique done well" plan.

## Setup

```bash
pip install mediapipe==0.10.13 opencv-python
```

**Important version note:** the newest MediaPipe (0.10.30+) dropped the legacy
`solutions` API in favor of a Tasks API that requires downloading a model file
from Google's servers at runtime. If you're ever on a machine/network without
access to `storage.googleapis.com`, pin to `0.10.13` as above — it ships the
pose model bundled inside the package, so it works fully offline. This bit me
while building this, so saving you the debugging time.

## Usage

```bash
python analyze_kick.py your_clip.mp4 --side right --out ./output
```

- `--side`: which leg is kicking (`right` or `left`)
- `--out`: where to write the annotated video + confidence CSV

## What you get

**Console output** — plain-language feedback like:
```
[OK] hip rotation: Your hips are rotating well through the kick.
[!] guard position: Your guard hand is dropping during the kick — keep it near your chin.
```
Plus a Track 2 summary telling you what % of frames near the kick had reliable tracking.

**`output/annotated.mp4`** — your original video with the skeleton overlay drawn on,
and the frame it picked as "peak kick" marked in red text. Watch this first — if
it picked the wrong frame as the peak, the feedback downstream will be wrong too,
and that's useful to know before you trust the coaching output.

**`output/confidence_report.csv`** — per-frame visibility scores for the kicking
ankle/knee/hip/wrist. Open this in a spreadsheet and look for dips near the kick
frame — that's motion blur showing up as low confidence.

## How to actually run the two tracks

**Track 1 (feedback accuracy):**
1. Film 10-15 roundhouse kicks — mix clean reps with deliberately flawed ones
   (short hip rotation, dropped guard, locked standing leg).
2. Run each through the script.
3. Show the annotated video + printed feedback to training partners *without*
   telling them what you expect it to say. Ask: does this match what you'd
   actually tell someone?

**Track 2 (frame-rate reliability):**
1. Film the *same* kick three ways: normal phone video (usually 30fps), phone
   slo-mo (120fps+), and a deliberately shaky/close-framed clip.
2. Run all three through the script.
3. Compare the "% reliable" number the console prints for each. This tells you
   directly whether normal-speed footage is usable or whether you'll need to
   force users into slo-mo mode (which is real onboarding friction worth knowing
   about now, not after you've built a UI around it).

## Known limitations (by design — this is a validation prototype, not a product)

- **Reference ranges are rough starting points**, not coach-validated. The angle
  ranges in `pose_extract.py` (`REFERENCE_RANGES`) are educated guesses based on
  general coaching cues. Part of what Track 1 tells you is whether these need
  adjusting — if training partners consistently disagree with the feedback,
  the ranges (not the pipeline) are probably the problem.
- **Single-frame analysis.** It only looks at the peak-kick frame, not the whole
  kinetic chain (wind-up through follow-through). Real form errors often show up
  in the transition, not just the peak — this is a known simplification to keep
  the prototype fast to build and test.
- **Side-on camera angle assumed.** The angle math assumes a roughly side-on or
  3/4 view of the kicker. A head-on angle will give unreliable hip-rotation numbers.
- **One kicking side per run.** Pass `--side left` or `--side right` explicitly.

## Web app (upload instead of command line)

If you'd rather upload a clip through a browser than run the CLI by hand:

```bash
cd webapp
pip install flask
python app.py
```

Then open `http://localhost:5000`. Pick a video, choose kicking side, hit
Analyze. You'll get the same feedback, confidence score, and annotated video
as the CLI — just as a page instead of console output.

This is a local single-process Flask app: no login, no database, no cloud
upload. That's deliberate — the point right now is testing whether the
*pipeline* is accurate, not building deployment infrastructure around a
pipeline that hasn't been validated yet. Each upload gets a unique session
folder under `webapp/results/`, so you can run it back-to-back with a few
training partners without their clips overwriting each other.

`webapp/app.py` imports and calls `run_analysis()` from `analyze_kick.py`
directly — it's the exact same pipeline, not a separate implementation, so
whatever you validate through the CLI holds for the web version too.

## Files

- `pose_extract.py` — pose extraction + angle math + reference ranges + feedback logic
- `analyze_kick.py` — CLI entry point + `run_analysis()`, the shared pipeline function used by both the CLI and the web app
- `webapp/app.py` — Flask upload app
- `webapp/templates/index.html` — upload form
- `webapp/templates/results.html` — results page (feedback, confidence, annotated video)
