# L1 reconstruction and label projection

Run from the drone-flyby directory in the existing Python environment (NumPy,
OpenCV, and the existing DTO module). No model is loaded by either utility.

```bash
python build_l1_reconstructions.py --capture-root ~/drone-captures
```

The default capture root is `~/drone-captures`. Read only `l1_phase_00` through
`l1_phase_08`, recursively matching PNG files to JSON sidecars. Each phase selects
the sequence with the most **distinct source frame numbers**, counting received
levels before filtering to L1. Retries cannot inflate that count. This ignores
short Verify sequences when the main sequence is present. Ties use lexical
sequence ID order. Missing phases are allowed. If a phase contains only a Verify
sequence, there is no metadata-based way to distinguish it from a short main run;
ensure the nine input directories contain the intended runs.

Frames across phases are joined by `frame`, not frame_index or sequence_id.
This assumes those validation runs show the same source video and source-frame
numbering. Selected sequences are recorded in each frame's metadata for review.

Each unique L1 region contributes one original 960x540 PNG at `(source_x/2,
source_y/2)` on a 1920x1080 canvas. There is no resizing. Identical regions are
deduplicated by keeping the first lexical capture path, even if their pixels
differ. Overlaps use integer averaging rounded half up. Unobserved pixels remain
black and coverage masks distinguish them from genuinely observed black pixels.
L0/L2 images never supply pixels. Selected PNGs must match their metadata SHA256,
be 960x540 8-bit RGB images, and represent a valid 1920x1080 source region within
3840x2160. Odd source edges fail explicitly because exact placement on the
half-resolution integer grid would otherwise be ambiguous.

Default output is `<capture-root>/reconstructed_l1`:

```text
images/frame_000100.png
masks/frame_000100.png
metadata/frame_000100.json
manifest.csv
```

Mask values are 255 for covered pixels and 0 otherwise. Metadata records source
frame, unique regions, coverage percentage, chosen and all available capture
paths, original/reconstruction sizes, selected sequences, and averaging/dedup
rules. Manifest paths are relative to its directory. Frames with no L1 captures
produce no output. `--output` selects another **new** directory. Existing output
directories and locations within source phase directories are rejected; source
captures are never modified. Invalid metadata/images cause an explicit error;
an interrupted or failed build may leave a partial output directory.

## Annotation input and projection

Annotate the reconstructed canvas in **pixel XYXY** coordinates, not normalized
YOLO coordinates. Use this JSON format:

```json
{
  "frames": [
    {"frame": 100, "annotations": [
      {"object_id": "ta-ta", "bbox": [500, 300, 600, 400]}
    ]},
    {"frame": 101, "annotations": []}
  ]
}
```

Boxes must have positive area and lie within the reconstructed 1920x1080 canvas.
Class names must be in the existing `dtos.OBJECT_CLASSES` list. Explicitly empty
annotations mean a reviewed negative frame. Frames absent from this JSON are
unlabeled and receive no label files. Review only observed areas using the masks;
missing regions and overlap averages can obscure objects.

```bash
python project_reconstruction_labels.py \
  --capture-root ~/drone-captures \
  --annotations reconstruction_annotations.json \
  --min-visible-fraction 0.7
```

Projection uses `source_bbox = reconstruction_bbox * 2`, subtracts the L1 source
region origin, then divides by two to obtain crop pixels. Clip to 0..960 and
0..540. Visible fraction is intersection area divided by the **full original
object box area**, before clipping; the configured threshold is inclusive.
Non-intersecting or zero-area results never emit labels, even at threshold zero.

Every selected-sequence L1 capture of each annotated source frame gets its own
YOLO detection TXT, including duplicate regions and retries. Default output is
`<capture-root>/projected_l1_labels`, preserving phase/sequence/image-stem paths.
Each line is `class_id center_x center_y width height`, normalized to 960x540,
using the unchanged OBJECT_CLASSES order. Images with no surviving annotations
get an empty TXT. No images are copied or modified. `--output` chooses a different
new directory. The input convention assumes complete labels for each included
frame; crops containing only filtered-out partial objects will have empty labels.

Lightweight checks:

```bash
python -m unittest test_l1_reconstructions -v
python -m py_compile build_l1_reconstructions.py project_reconstruction_labels.py test_l1_reconstructions.py
```

## CVAT COCO to a prepared fine-tuning dataset

The complete workflow is:

```text
CVAT COCO detection export
  -> cvat_coco_to_reconstruction.py
  -> audit_reconstruction_annotations.py
  -> project_reconstruction_labels.py
  -> build_validation_yolo_dataset.py
  -> prepare_combined_finetune_dataset.py
  -> USER manually launches fine-tuning
```

These preparation tools do not load YOLO, train, run inference, or call an
evaluator. They write new outputs only; choose a new output path if a previous
one already exists. Inputs and source captures are never modified. Validation is
performed before dataset directories are created; I/O interruption can still
leave a partial output. The tools use the existing Python environment, including
PyYAML, NumPy/OpenCV (imported by the existing capture/projector helpers), and DTOs.

### 1. Convert and audit the CVAT export

From `drone-flyby`, place the exported COCO JSON at
`~/drone-captures/cvat_instances_default.json` (or substitute its actual path):

```bash
python cvat_coco_to_reconstruction.py \
  --coco ~/drone-captures/cvat_instances_default.json \
  --output ~/drone-captures/reconstruction_annotations.json \
  --summary ~/drone-captures/cvat_conversion_summary.json

python audit_reconstruction_annotations.py \
  --annotations ~/drone-captures/reconstruction_annotations.json
```

The converter reads source frame numbers from `frame_XXXXXX.png` basenames (six
or more digits; exported directory prefixes are accepted). It requires 1920x1080
images and resolves arbitrary COCO category IDs through category **names** in
`dtos.OBJECT_CLASSES`. XYWH boxes become floating-point pixel XYXY. Negative or
zero sizes, non-finite coordinates, unknown classes/references, duplicate image
IDs, duplicate category IDs, and duplicate source frames fail explicitly.
Canvas overflow of at most **1e-6 pixels** is clamped; larger overflow is rejected.
Every listed COCO image is retained, including empty reviewed frames. Frames sort
numerically; annotations sort by OBJECT_CLASSES index and then XYXY coordinates.

The console and optional summary JSON report image/annotated/empty counts, total
and per-class boxes, first/last frame, and absent classes. Both converter and audit
print this completeness warning for zero-count classes, without rejecting them:

> Classes with zero human annotations: ... Verify these objects were truly absent
> from reviewed frames; otherwise unlabeled visible objects will be treated as background.

The audit is read-only. It reports per-class box counts and distinct-frame counts,
median width/height/area, min/max area, boundary-touching boxes, exact duplicate
geometries (including conflicting classes), and frames with multiple same-class
instances. It flags width/height/area values outside 1.5 times the interquartile
range when a class has at least four boxes; these are review flags, not rejection
rules. It never removes boxes or creates pseudo-labels. Check these reports before
continuing: all included COCO images are assumed completely reviewed.

### 2. Project onto original L1 captures and package them

```bash
python project_reconstruction_labels.py \
  --capture-root ~/drone-captures \
  --annotations ~/drone-captures/reconstruction_annotations.json \
  --output ~/drone-captures/projected_l1_labels \
  --min-visible-fraction 0.7

python build_validation_yolo_dataset.py \
  --capture-root ~/drone-captures \
  --labels-root ~/drone-captures/projected_l1_labels \
  --output ~/drone-captures/validation_yolo
```

The builder reuses the reconstruction's main-sequence selection. Labels must have
the exact relative phase/sequence/PNG-stem path produced by the projector. Entirely
unreviewed source frames with no TXT files are skipped and reported, never assumed
negative. If a frame has labels elsewhere but a selected capture is missing its
TXT, the builder fails. Orphan labels also fail. Add `--require-all-labels` to fail
even for entirely unreviewed frames. An explicitly empty TXT is a valid negative.
All labels are validated as normalized five-column YOLO detection records with
class indices 0..15 and positive-area boxes within the crop.

The builder verifies original image SHA256 and 960x540 PNG header dimensions and
copies bytes without resizing or re-encoding. Output contains:

```text
validation_yolo/images/all/*.png
validation_yolo/labels/all/*.txt
validation_yolo/manifest.csv
validation_yolo/data.yaml
```

Names include phase, source frame, center, and a deterministic unique input
ordinal. Images deduplicate **only** when image SHA256 and exact label bytes both
match; whitespace-different label files conservatively remain separate. Manifest
rows retain **every source capture**, including deduplicated aliases pointing to
the same output filename. Columns include phase, sequence_id, source_frame,
center, region, source image/label paths, image SHA256, annotation_count, and a
deduplicated flag. Do not discard alias rows: they are needed for safe splitting.

`data.yaml` records `path`, `all: images/all`, and the unchanged 16-class names in
OBJECT_CLASSES order. It intentionally omits train/val: this intermediate dataset
is unsplit and must not be handed directly to training as a train/val dataset.

### 3. Prepare Helsinki plus validation with a contiguous holdout

```bash
python prepare_combined_finetune_dataset.py \
  --helsinki-root datasets/helsinki_multires \
  --validation-root ~/drone-captures/validation_yolo \
  --output ~/drone-captures/combined_finetune \
  --validation-holdout-start-frame 200 \
  --validation-holdout-end-frame 249
```

The existing checkpoint's saved training arguments point to
`datasets/helsinki_multires/data.yaml`. That is the assumed Helsinki directory in
this example; `datasets/helsinki` also has the supported layout. Pass the directory
containing the dataset you intend to retain. The script requires `data.yaml`,
`images/train`, `images/val`, and matching `labels/train` and `labels/val`, with the
same 16-class order. Nested image directories are supported; image-list TXT files,
remote datasets, and custom split directory layouts are not.

Existing Helsinki train/val assignment is preserved exactly. The explicitly
supplied `--helsinki-root` determines the dataset location; a stale `path:` in
its YAML is ignored, which permits copying the dataset between Windows and Linux.
Relative train/val entries must resolve to the conventional directories above;
absolute entries must already point there. Labels must exist, including explicit
empty labels for negative images.

All validation samples with source frames **200 through 249 inclusive** go to val;
all others go to train. Every phase/crop of a source frame stays in that split.
There is no random frame split or extra temporal buffer. The requested holdout
must contain at least one sample. A deduplicated group whose provenance spans
both sides of the holdout boundary causes an error rather than silently dropping
provenance or leaking data. Choose a holdout range keeping the group on one side.
Identical image bytes across train/val are also rejected, even if labels differ
or the two copies came from Helsinki and validation separately.

Output is `images/train`, `images/val`, `labels/train`, `labels/val`, a train-ready
`data.yaml`, `manifest.csv`, and `preparation.json`. Filenames use `helsinki_` or
`validation_` prefixes. The manifest retains source paths, splits, all contributing
validation frame numbers, hashes, counts, and validation provenance. The preparation
JSON records the inclusive range, counts, and Helsinki assignment policy. Review
the counts before training. Contiguous splitting controls source-frame placement;
it does not prove that adjacent frames at its edges are visually unrelated.

### 4. User-only manual fine-tuning, after reviewing the prepared data

No preparation step launches training. If and when you choose to train, this
command uses the existing local checkpoint and training script:

```bash
python train_yolo.py \
  --model runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt \
  --data ~/drone-captures/combined_finetune/data.yaml \
  --name helsinki_validation_finetune \
  --device 0
```

This manual command retains the training script's existing defaults (including
100 epochs); adjust them yourself as needed. It was **not executed** during
pipeline implementation. No mAP improvement is claimed.

Pipeline-only verification:

```bash
python -m unittest test_validation_dataset_pipeline test_l1_reconstructions -v
python -m py_compile cvat_coco_to_reconstruction.py audit_reconstruction_annotations.py build_validation_yolo_dataset.py prepare_combined_finetune_dataset.py validation_dataset_utils.py test_validation_dataset_pipeline.py
```

## Offline diagnostic AP on captured L0 images

`evaluate_validation_annotations.py` is a **user-run local inference utility**.
It does not train, contact the validation website, launch an API, or run
local_evaluator.py. It reuses the existing detector and mirrors the supplied
evaluator's COCO AP@0.50 scoring semantics using faster-coco-eval.

Run this on your RTX GPU PC, from `drone-flyby` in the existing drone-yolo
environment (PowerShell; replace the COCO path if needed):

```powershell
python evaluate_validation_annotations.py --coco "$HOME/drone-captures/cvat_instances_default.json" --capture-root "$HOME/drone-captures" --weights runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt --device 0 --conf 0.001 --iou 0.5 --prediction-conf-report 0.05 --save-predictions validation_predictions.json --save-errors validation_errors.csv
```

The COCO export determines the reviewed subset (currently expected to be 180
images, but no count is hardcoded). Categories resolve by name through the same
converter. Only `l0_repeat*` directories under the capture root are searched.
Within each run, the main sequence is selected by the greatest count of distinct
source frame numbers, with lexical ID tie-breaking, exactly as reconstruction.
Only reviewed frames at L0 are eligible. For repeated copies of one frame, the
first lexical run/path wins; selected image paths are included in the predictions
JSON. Short Verify sequences are ignored when a longer main run exists.

Reconstruction boxes become source pixels by multiplying by two, then normalize
by 3840x2160. Detector boxes already use that same full-source normalized XYXY
system: the exact captured 960x540 L0 image covers [0,0,3840,2160]. There is no
raw reconstruction-pixel/L0-pixel comparison, image-shape guessing, or preprocessing
resize of the capture. Reconstruction size, source size, L0 metadata dimensions,
full-frame region, decoded image dimensions/type, PNG signature, and SHA256 are
validated explicitly. YOLO's normal internal `imgsz=960` preprocessing remains.

Inference defaults: device 0, conf 0.001, NMS IoU 0.5, imgsz 960, agnostic_nms true,
max_det 500. The script uses explicit arguments, not YOLO_* environment overrides.
AP includes all low-confidence detections. `--prediction-conf-report` (default
0.05) separately counts detections above the endpoint threshold from that result;
it does not filter the AP inputs or cause a second inference pass.

The script reports reviewed/matched/missing frames, GT and prediction counts,
per-class AP, and the mean over **GT-present classes only**. Classes with no human
GT print N/A and are listed separately. Missing L0 frames stay in the reviewed
subset with empty predictions, costing recall; they are never silently dropped.
If all GT is empty, overall AP is N/A. Empty reviewed frames still count false
positives for GT-present classes. COCO's default 100 detections per image/class
cap, all-area precision slice, IoU 0.50, and 101 recall points match local_evaluator;
this scoring cap is distinct from the detector's 500-output limit.

Optional prediction JSON stores every reviewed source frame, selected capture
path, normalized class/bbox/confidence records, settings, missing frames, and AP.
The error CSV has one row per GT. `matched` uses confidence-ordered, class-aware,
one-to-one association at IoU >= 0.50 under the same per-class scoring cap.
Matched class/confidence/IoU are included. Separate `best_predicted_*` columns
report the highest-IoU overlapping prediction of **any class** from the full
low-threshold output, exposing potential misclassifications; ties prefer higher
confidence then input order. A missing capture is explicitly marked. These
diagnostics use the AP confidence threshold, not the endpoint reporting threshold.
Existing report files are not overwritten; use new report filenames for reruns.

This subset score is diagnostic, not an official leaderboard result: omitted
human labels and uncovered reconstruction regions can affect its interpretation.

Non-inference verification:

```bash
python -m unittest test_evaluate_validation_annotations -v
python -m py_compile evaluate_validation_annotations.py test_evaluate_validation_annotations.py
```
