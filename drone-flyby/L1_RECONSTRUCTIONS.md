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
