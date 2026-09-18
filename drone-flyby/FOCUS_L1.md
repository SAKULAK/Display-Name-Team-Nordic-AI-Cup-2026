# Selective L1 confirmation with individual bbox prediction

`hold_full` remains unchanged and remains the default. `focus_l1` still requires
three successful L0 observations before an eligible L1 request and returns to L0
after that one visit. No L2, exploration, API changes, warmup, detector/checkpoint
changes, training or evaluator changes. Only lightweight unit/static checks were run.

## Candidate filtering

Only these classes may consume a confirmation frame: `small_launcher`, `ta-ta`,
`mine_roller`, `hangar`, `medium_launcher`, `medium_plane`.

A candidate must also have confidence <= `FOCUS_MAX_CONF` (default **0.45**), pass
existing cooldown/confirmation gates and reach `FOCUS_MIN_SCORE` (default **1.2**).
Thus the reported ta-ta 0.164 and small_launcher 0.351 / 0.104 cases are eligible,
while spacecraft 0.141 and ta-ta 0.677 are not. Filtering focus candidates never
removes objects from the L0 response or saved full-frame snapshot.

The request DTO contains no reliable remaining-frame/end-of-sequence field. There
is no special treatment of frame 24, frame 25, or any assumed sequence length.
A last-frame focus request can still occur because the policy cannot know it is last.

## Snapshot bridge: preserved with per-object prediction

Before issuing L1, the policy deep-copies all current L0 annotations, the source
frame, target index and an aligned optional four-edge velocity per annotation.
Snapshot output remains independent of tracker identity, TTL and confidence decay.

For each current L0 object, matching to the previous L0 observation requires the
same class, IoU and center distance. Reliability is conservative: compare against
the globally shifted previous box, require IoU >= 0.2, center distance within the
existing frame-gap gate, width/height ratios within [0.5, 2], and reciprocal best
matches separated from runners-up by 0.1 cost. Ambiguous matches use fallback.

For a reliable match, each edge velocity is computed from the ORIGINAL previous
box: `(current_edge - previous_edge) / (current_frame - previous_frame)`.
For the immediate next source frame, each saved edge receives exactly its own
velocity once. Width and height can therefore change, and one object's motion can
differ from the scene translation. Nonpositive predicted sizes reject the velocity.

Unmatched/ambiguous objects use the existing median global per-source-frame dx/dy.
Individual velocity is never extrapolated over multiple source frames. If the next
received frame skips source frames, the prior global translation fallback is used
with the actual gap. The frame gap and counts are logged. All outputs are clipped
and snapshot confidence remains unchanged.

## Conservative L1 merge

The existing snapshot bridge is retained; only its association criteria change:

- The focused target's matching live box always wins, including a class correction.
  Its geometric match uses the individually predicted target box when available.
- An incidental object replaces a saved object only with the same class, IoU >= 0.65,
  center distance <= 0.25 of the smaller box diagonal, and unambiguous reciprocal
  matching. The matching margin is 0.1 in `1-IoU` cost.
- A weak/ambiguous incidental live box near a saved object is not added as a duplicate;
  the propagated saved box remains. Novelty checks are class-agnostic and conservative
  (IoU >= 0.1 or close centers means possibly the same object).
- A live detection may be added as new only if it is not a possible duplicate of a
  saved object or an already accepted live object.
- All unmatched saved objects, including objects inside the view, remain. Outside-view
  saved objects are not replaced. A live replacement can claim at most one saved box.
- There is still no NMS among saved objects. Only clipping to an empty/degenerate box
  or the 500-output protocol cap can remove an unmatched saved object; both are logged.

A repeated identical request returns its cached response. The snapshot is consumed
once, discarded on the next fresh L0, and never contaminates that authoritative L0
output. Unexpected extra L1 frames log a consumed/missing snapshot and request L0.

## Configuration

Settings are read once on first focus use; restart the API after changing them.

| Variable | Default | Purpose |
|---|---:|---|
| `FOCUS_MIN_L0_FRAMES` | 3 | Successful L0 frames before zoom |
| `FOCUS_MAX_CONF` | 0.45 | Maximum candidate confidence (inclusive) |
| `FOCUS_MIN_SCORE` | 1.2 | Minimum candidate score |
| `FOCUS_CONFIRMED_FRAMES` | 12 | Source-frame confirmation lifetime |
| `FOCUS_CONFIRM_CONF` | 0.70 | Same-class L1 confidence needed to confirm |
| `FOCUS_CONFIRMED_PENALTY` | 10 | Score penalty near a confirmed object |

The class/position confirmation record is independent of track ID. L0 matching
updates its predicted position without refreshing its L1 confirmation date.
Scoring weights and geometric match thresholds are in `FocusConfig`. Legacy tracker
TTL/decay settings still never apply to snapshot confidence or preservation.

## Diagnostics

`FOCUS_BRIDGE` retains count accounting:

```
snapshot_count = snapshot_kept + snapshot_replaced + snapshot_clipped_out + snapshot_overflow
final_count = live_count + snapshot_kept
```

`live_count` now counts ACCEPTED live boxes. `live_candidates` counts live boxes
after live-only deduplication; `live_rejected` counts rejected ambiguous/duplicate
incidental candidates. `live_duplicates` counts live-only deduplication losses.

Each L1 frame additionally logs compact JSON class counts:

```
FOCUS_BRIDGE_CLASSES frame=... propagated={...} live_replaced={...} live_added={...}
 individual_velocity=... global_fallback=...
```

The three class maps describe FINAL output sources. Replacement classes refer to
the emitted live label (including a target class correction). Their counts sum to
`final_count`. The velocity counters describe all valid predicted snapshot boxes
BEFORE replacement or the output cap, so they sum to `snapshot_count - snapshot_clipped_out`.

`FOCUS_RESULT` remains. L0 logs class/confidence filter counts and skip reasons.
No raw bbox arrays or tensors are logged.

## Evidence and limitations

User-supplied previous offline results: hold_full **0.693**, snapshot focus_l1
**0.669**. Detection counts were preserved, but jet_plane, large_tower and small_plane
lost AP. The new selective/individual-motion version is **unmeasured**. Confidence
gains and passing unit tests do not establish an mAP improvement.

Conservative matching can miss new objects close to existing ones; ambiguous motion
falls back to global translation. This remains an existing-detection confirmation
experiment, not exploration or long-term tracking.

## Next offline comparison: USER executes

Use two PowerShell terminals in `drone-flyby`, with a fresh API for each policy.
Do not use the old benchmark runner, which performs warmup.

In server terminal A:

```powershell
conda activate drone-yolo
New-Item -ItemType Directory -Force benchmarks/logs/selective_focus | Out-Null
$env:YOLO_WEIGHTS = (Resolve-Path 'runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt').Path
$env:YOLO_CONF = '0.05'
$env:YOLO_IOU = '0.5'
$env:YOLO_AGNOSTIC_NMS = 'true'
$env:YOLO_MAX_DET = '500'
$env:YOLO_DEVICE = '0'
$env:YOLO_LOG_EVERY = '0'
$env:FOCUS_MIN_L0_FRAMES = '3'
$env:FOCUS_MIN_SCORE = '1.2'
$env:FOCUS_MAX_CONF = '0.45'
$env:FOCUS_CONFIRMED_FRAMES = '12'
$env:FOCUS_CONFIRM_CONF = '0.70'
$env:FOCUS_CONFIRMED_PENALTY = '10'
Get-FileHash $env:YOLO_WEIGHTS -Algorithm SHA256
$env:CAMERA_POLICY = 'hold_full'
python api.py 2>&1 | Tee-Object benchmarks/logs/selective_focus/hold_server.log
```

Wait for startup, then run in evaluator terminal B:

```powershell
conda activate drone-yolo
python local_evaluator.py 2>&1 | Tee-Object benchmarks/logs/selective_focus/hold_eval.log
```

Stop the API in terminal A with Ctrl+C and restart it there:

```powershell
$env:CAMERA_POLICY = 'focus_l1'
python api.py 2>&1 | Tee-Object benchmarks/logs/selective_focus/focus_server.log
```

Then run in terminal B:

```powershell
python local_evaluator.py 2>&1 | Tee-Object benchmarks/logs/selective_focus/focus_eval.log
```

Return all four logs plus checkpoint SHA256. We need both overall/per-class AP
summaries, accepted/skipped frames, camera moves/refusals, every `FOCUS_BRIDGE`, `FOCUS_BRIDGE_CLASSES` and
`FOCUS_RESULT`, and focus counts. In particular compare jet_plane, large_tower and small_plane for recovery and
small_launcher/ta-ta for preserved gains. Include individual/global prediction counts
and the per-class propagated/replaced/added counts.
Do not infer improvement from confidence increases alone.

Lightweight checks only:

```powershell
python -m unittest test_focus_policy -v
python -m py_compile focus_policy.py example.py test_focus_policy.py
```
