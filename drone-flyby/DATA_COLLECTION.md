# Validation data collection

This optional mode records received validation views. It is for validation data
collection, **not final evaluation**: capture-only intentionally returns no
detections. It never runs image decoding, YOLO, or detector initialization.
With capture disabled and capture-only disabled, normal competition prediction
behavior is unchanged. Recording can also accompany normal inference, including
focus_l1, without changing its camera decisions or returned annotations.

| Environment variable | Default | Meaning |
|---|---|---|
| CAPTURE_ENABLED | false | Save received PNG bytes and metadata |
| CAPTURE_ROOT | captures | Output directory (relative to process working directory) |
| CAPTURE_RUN_ID | run | Run directory; use a distinct ID per attempt/phase |
| CAPTURE_ONLY | false | Skip image decoding/detection and return empty annotations |
| CAMERA_POLICY | hold_full | Set to data_collect to follow collection routes |
| COLLECT_LEVEL | 0 | Desired collection resolution: 0, 1, or 2 |
| COLLECT_PHASE | 0 | Integer starting position, modulo 9 for L1 or 28 for L2 |

Boolean switches accept true/1/yes (case-insensitive); other values are disabled.
CAPTURE_ONLY does not implicitly enable CAPTURE_ENABLED. Use data_collect for
deliberate collection routes. Other camera policies still run; focus_l1 receives
empty detections in capture-only mode and cannot select new focus targets.
Its normal inference behavior is unchanged. Collection settings are
read only when data_collect runs; invalid level/phase settings raise a configuration
error. Set the environment before starting the server.

## Image geometry and files

All received images are 960x540 lossless PNGs. The represented source region is:

| Level | Source region size |
|---|---|
| L0 | 3840x2160 |
| L1 | 1920x1080 |
| L2 | 960x540 |

source_region_xyxy places each image in the original 3840x2160 source frame.
To map transmitted pixel (u,v) to source pixels, use
`x=x1+u*(x2-x1)/transmitted_width`, `y=y1+v*(y2-y1)/transmitted_height`.
Metadata retains the actual original and transmitted dimensions from the request.

Files are written under `<root>/<sanitized_run_id>/<sanitized_sequence_id>/`:

```text
frame_000022_L2_x1200_y700.png
frame_000022_L2_x1200_y700.json
```

The PNG is the exact Base64-decoded payload, never re-encoded. JSON includes
sequence_id, frame, frame_index, request_id, resolution_level, center_x/center_y,
source_region_xyxy, transmitted_width/height, original_width/height,
image_media_type, image_sha256, and camera_command_feedback when present.
Original IDs remain in metadata. Unsafe directory IDs are sanitized with a hash
suffix; Windows reserved device names are handled as well.

Existing PNGs or JSONs are never silently replaced: duplicates gain a request-ID
hash and numeric suffix, including identical retries. Exclusive reservations
prevent concurrent capture calls from claiming the same pair. Each file is written
via a temporary file and atomic replace. The pair is not a single transaction:
a process interruption or disk failure can leave a PNG without a sidecar or a
stale reservation. Subsequent captures use a new name. Capture failures are logged
and do not fail prediction. Disabled capture performs no filesystem operations.

## Collection routes

L0 holds full view. Returning from L2 goes through L1 before requesting L0.

L1 follows these nine centers cyclically:

```text
(960,540), (1920,540), (2880,540), (2880,1080), (2880,1620),
(1920,1620), (960,1620), (960,1080), (1920,1080)
```

The final-to-first move is approximately 1101.45 pixels, within the nominal
1102-pixel L1 limit. On entry from L0, phase selects the first center. On L1,
the nearest current route point determines the next point. Returning from L2
first clamps the current center to L1 bounds.

L2 uses x = 480, 960, 1440, 1920, 2400, 2880, 3360 and
y = 270, 810, 1350, 1890. Its exact cyclic grid-index route is:

```text
(0,0) (1,0) (2,0) (3,0) (4,0) (5,0) (6,0)
(6,1) (5,1) (4,1) (3,1) (2,1) (1,1)
(1,2) (2,2) (3,2) (4,2) (5,2) (6,2)
(6,3) (5,3) (4,3) (3,3) (2,3) (1,3) (0,3)
(0,2) (0,1)
```

All 28 centers occur once; every edge, including the wrap, is horizontal 480
or vertical 540 pixels, within the nominal 551-pixel L2 limit. L0 first requests
an L1 bridge by clamping the phase-selected L2 center into the supplied L1 bounds.
If that bridge arrives, the next command enters L2 at the phase-selected center.
Otherwise L1 enters the nearest L2 route point. L2 advances from the nearest
actual route point, with route order breaking ties deterministically.

Every command checks the supplied allowed levels, target bounds, and maximum
center delta (including the supplied L0 reset exemption). No direct L0/L2
transition is sent even if supplied levels include both. Illegal desired commands
hold and log the reason; restrictive constraints may prevent route progress.
COLLECT lines report current/target positions and phase. CAPTURE lines report
frame, level, region, bytes, and SHA256. Identical camera retries use the cached
command. Sequence changes, frame rewinds, and collection setting changes reset
the small initial-bridge state.

Changing phase across validation attempts samples different positions for the
same source frame once at the requested level. Startup bridge frames, skipped
requests, and refused commands can affect alignment; use the recorded metadata.

## Linux / Azure examples

From the drone-flyby directory in your existing Python environment, collect L0:

```bash
CAPTURE_ENABLED=true CAPTURE_ONLY=true \
CAPTURE_ROOT=/mnt/validation-captures CAPTURE_RUN_ID=l0_phase0 \
CAMERA_POLICY=data_collect COLLECT_LEVEL=0 COLLECT_PHASE=0 python api.py
```

Stop that server before starting another run. For an L1 attempt:

```bash
CAPTURE_ENABLED=true CAPTURE_ONLY=true \
CAPTURE_ROOT=/mnt/validation-captures CAPTURE_RUN_ID=l1_phase4 \
CAMERA_POLICY=data_collect COLLECT_LEVEL=1 COLLECT_PHASE=4 python api.py
```

For an L2 attempt:

```bash
CAPTURE_ENABLED=true CAPTURE_ONLY=true \
CAPTURE_ROOT=/mnt/validation-captures CAPTURE_RUN_ID=l2_phase7 \
CAMERA_POLICY=data_collect COLLECT_LEVEL=2 COLLECT_PHASE=7 python api.py
```

Use an existing writable persistent directory on Azure in place of
/mnt/validation-captures if needed. These commands start the existing API on
port 9053; they do not initiate a validation attempt. Choose phases 0..8 or
0..27 and distinct run IDs for later attempts. For final competition operation,
unset CAPTURE_ONLY and CAPTURE_ENABLED and restore your intended CAMERA_POLICY.
