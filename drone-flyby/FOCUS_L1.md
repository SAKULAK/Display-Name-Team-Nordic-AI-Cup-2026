# Strict target-only L1 confirmation

focus_l1 is target-only. L1 detections cannot add, replace, re-score, reclassify,
or remove non-target snapshot objects.

The most recent L0 snapshot is authoritative. L1 inspects only the selected weak
target, then requests L0 using the supplied camera constraints. There is no L2,
exploration, sweeping, sequence-length assumption, or special end-frame handling.
The default hold_full and baseline_sweep behavior remain unchanged.

## Integration and candidate rules

predict() calls the singleton's begin_request before detection in focus_l1 mode,
runs detect exactly once, then calls process with its annotations. A detector
exception is logged and passed to process as an empty list, so failed L1 still
bridges the saved snapshot and attempts to return to L0. Other policies never
invoke the focus policy; invalid focus environment settings cannot affect them.

By default, only small_launcher, ta-ta, mine_roller, hangar, medium_launcher, and medium_plane
can trigger inspection. Default gates are confidence <= 0.45, score >= 1.2, and
at least three L0 frames with detections since startup or the last L1. Empty L0
observations (including failed detector calls) replace authoritative L0 state but
do not advance that waiting counter. Source-frame gaps and retries do not add
extra observations. Existing cooldown and confirmation score penalties still apply.

`FOCUS_CLASSES` overrides the focus-class allowlist with comma-separated names;
whitespace is stripped and empty entries ignored. Unknown names simply do not
match detections. Unset preserves the six classes above; an explicitly empty
allowlist disables focus candidates. `FOCUS_MIN_CONF` defaults to **0.0**.
Confidence eligibility is inclusive: `FOCUS_MIN_CONF <= confidence <= FOCUS_MAX_CONF`.
Both bounds must be in [0, 1], with minimum <= maximum. Defaults preserve previous
behavior, including sub-0.15 persistence protection. L0 logs include both bounds
and the active allowlist. Restart the API after changing these settings.

For the ta-ta-only 0.35–0.45 experiment:

```powershell
$env:FOCUS_CLASSES = 'ta-ta'
$env:FOCUS_MIN_CONF = '0.35'
$env:FOCUS_MAX_CONF = '0.45'
```

Confidence < 0.15 additionally requires a reliable geometric same-class match in
the immediately previous authoritative L0 observation. It uses the existing
individual-velocity correspondence: globally shifted previous box, actual frame
gap, IoU >= 0.2, the existing center-distance gate, size ratios in [0.5, 2], and
reciprocal best matches with a 0.1 cost margin. Invalid next-frame size rejects the
correspondence. L1 cannot supply evidence. Confidence exactly 0.15 follows normal
candidate rules. An empty intervening L0 prevents using an older observation.

## Snapshot and geometry

When requesting L1, deep-copy the full L0 annotation list (up to the protocol cap
of 500), source frame, target index, and optional per-object four-edge velocities.
Reliable edge velocities use (current edge - previous edge) / source-frame gap.

For the immediately next source frame, use each reliable individual velocity,
including size changes. Otherwise use the existing median global motion fallback.
If source frames are skipped, use global dx/dy times the actual gap for every
object; never extrapolate individual velocities across that gap. Clip boxes to
the frame and omit degenerate/out-of-frame projections. Preserve saved confidence.

The projected snapshot supplies every L1 output box. Tracker TTL, confidence
decay, live duplicate suppression, and incidental detections cannot remove it.

## Target-only outcome

Associate L1 detections with the predicted target box using the existing
deterministic IoU/distance ranking. Both boxes must intersect the received crop.
Reject a live candidate that matches another projected snapshot object's location
at least as well in overlap or center distance.

- Same-class match: use L1 confidence, always retaining the projected L0 bbox.
- Different-class match: change class and confidence only at confidence >=
  FOCUS_CONFIRM_CONF (default 0.70) and a strong spatial match. Reuse the existing
  strong thresholds: IoU >= 0.65 and center distance <= 0.25 of the smaller
  diagonal. Always retain the projected L0 bbox.
- A weak class correction retains the original projected target unchanged.
- No spatial match: remove only the target if its saved confidence <= 0.15;
  otherwise retain it unchanged. This includes a failed L1 detection call.

Updates occupy the target's original position in the output; unrelated objects
retain their order, class, confidence, and predicted geometry. New L1 objects are
never emitted, even when the snapshot is already at the 500-object cap.

Same-class accepted confidence >= 0.70 creates the existing confirmation record
at the emitted projected geometry. A correction creates no old-class confirmation.
Confirmation lifetime, trajectory following, and four-L0-refresh cooldown remain.
Configuration is loaded lazily on first focus use; restart after changing it.

| Variable | Default |
|---|---:|
| FOCUS_MIN_L0_FRAMES | 3 |
| FOCUS_CLASSES | The six classes listed above |
| FOCUS_MIN_CONF | 0.0 |
| FOCUS_MAX_CONF | 0.45 |
| FOCUS_MIN_SCORE | 1.2 |
| FOCUS_CONFIRMED_FRAMES | 12 |
| FOCUS_CONFIRM_CONF | 0.70 |
| FOCUS_CONFIRMED_PENALTY | 10 |

An identical retry returns the cached decision without advancing policy state.
A snapshot is consumed once and discarded on fresh L0. Extra L1 frames with a
consumed/missing snapshot return no annotations and attempt L0. Sequence changes
and non-retry source-frame rewinds reset state.

## Diagnostics

FOCUS_BRIDGE reports frame, snapshot_count, snapshot_kept, target_updated,
target_removed, final_count, snapshot_clipped_out, source_frame_gap,
snapshot_status, individual_velocity, and global_fallback. snapshot_kept counts
unchanged projected objects; target_updated counts accepted target matches.
Legacy snapshot_replaced equals target_updated; live_added and snapshot_overflow
are always zero. live_count is the number of input L1 detections.

```
snapshot_count = snapshot_kept + target_updated + target_removed + snapshot_clipped_out
final_count = snapshot_kept + target_updated
individual_velocity + global_fallback = snapshot_count - snapshot_clipped_out
```

FOCUS_RESULT reports L0 class/confidence/area, spatial matched status, L1
class/confidence, same_class, accepted class_changed/confidence_changed,
target_removed, and reclassification_accepted. A spatially matched but rejected
class correction reports matched=true with no accepted changes. No raw tensors
or large box arrays are logged.

## Evidence and limitations

Historical user-supplied offline results: hold_full baseline **0.693**; older
snapshot focus experiment **0.669**, which was worse. This strict target-only
version is **unmeasured** until the user runs the evaluator. Passing unit tests
does not imply mAP improvement.

Conservative correspondence can reject real weak targets or class corrections.
Global fallback can miss individual motion. A detector failure can remove an
unmatched <=0.15 target by design. These tradeoffs need the offline comparison.
No training, inference sweep, API benchmark, or evaluator was run for this change.

## Next offline comparison: USER executes

Use two PowerShell terminals in `drone-flyby`, with a fresh API for each policy.
Do not use the old benchmark runner, which performs warmup.

In server terminal A:

```powershell
conda activate drone-yolo
New-Item -ItemType Directory -Force benchmarks/logs/target_only_focus | Out-Null
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
python api.py 2>&1 | Tee-Object benchmarks/logs/target_only_focus/hold_server.log
```

Wait for startup, then run in evaluator terminal B:

```powershell
conda activate drone-yolo
python local_evaluator.py 2>&1 | Tee-Object benchmarks/logs/target_only_focus/hold_eval.log
```

Stop the API in terminal A with Ctrl+C and restart it there:

```powershell
$env:CAMERA_POLICY = 'focus_l1'
python api.py 2>&1 | Tee-Object benchmarks/logs/target_only_focus/focus_server.log
```

Then run in terminal B:

```powershell
python local_evaluator.py 2>&1 | Tee-Object benchmarks/logs/target_only_focus/focus_eval.log
```

Keep all four logs and the checkpoint SHA256. Compare overall/per-class AP,
accepted/skipped frames, camera moves/refusals, focus counts, and FOCUS_BRIDGE /
FOCUS_RESULT diagnostics. Confidence increases alone do not establish improvement.

Lightweight checks only:

```powershell
python -m unittest test_focus_policy -v
python -m py_compile focus_policy.py example.py test_focus_policy.py
```
