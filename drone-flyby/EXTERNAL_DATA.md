# Download data, prepare the dataset, then train

```text
Google Drive download: <GOOGLE_DRIVE_LINK>
    ↓
C:\Users\<USERNAME>\nordicai\drone-captures\
    ↓
l1_phase_00 ... l1_phase_08 (PNGs AND matching JSON sidecars)
    ↓
CVAT conversion → audit → L1 label projection → validation YOLO → combined dataset
    ↓
User launches fresh fine-tune or resumes last.pt
```

No Google Drive connector or download is needed to read the repo. The owner uploads
the large data separately and supplies the link. Preserve directory names and JSON
sidecars when extracting; do not put another wrapper folder between the capture
root and the phase folders.

```text
C:\Users\<USERNAME>\nordicai\drone-captures\
  l1_phase_00\<sequence>\*.png + *.json
  l1_phase_01\<sequence>\*.png + *.json
  l1_phase_02\<sequence>\*.png + *.json
  l1_phase_03\<sequence>\*.png + *.json
  l1_phase_04\<sequence>\*.png + *.json
  l1_phase_05\<sequence>\*.png + *.json
  l1_phase_06\<sequence>\*.png + *.json
  l1_phase_07\<sequence>\*.png + *.json
  l1_phase_08\<sequence>\*.png + *.json
  l0_repeat_01\...                     # optional diagnostic-only
  l0_repeat_02\...                     # optional diagnostic-only
  projected_l1_labels\...              # generated locally
  validation_yolo\...                  # generated locally
  combined_finetune\...                # generated locally
```

**L1 is required for projected-label training. L0 is only required for reproducing
the old validation diagnostic.** Reconstructed images are optional for inspection
or re-annotation; the provided CVAT JSON is enough for label projection.

Also obtain these separately, relative to the cloned repo's `drone-flyby` directory:

```text
datasets/helsinki_multires/data.yaml
datasets/helsinki_multires/images/train/ and images/val/
datasets/helsinki_multires/labels/train/ and labels/val/
runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt
runs/detect/validation_finetune_v1/     # only if sharing an existing run for resume
```

The best baseline checksum is in [TEAM_HANDOFF.md](TEAM_HANDOFF.md). No Drone Flyby
checkpoint is currently tracked in Git. Keep checkpoint versions/hash information
with the Drive download. An existing fine-tune folder should include weights,
args.yaml and results.csv, not just a best.pt intended for inference.

## Reproducible preparation commands (PowerShell)

Replace `<REPO_DIRECTORY>` with the cloned repository directory. The CVAT export is
the small `validation_cvat.json` in Git. Every build step refuses existing outputs;
on reruns reuse verified unchanged outputs or choose a new set of output paths.
No destructive cleanup is necessary.

```powershell
Set-Location '<REPO_DIRECTORY>\drone-flyby'
conda activate drone-yolo
$CaptureRoot = Join-Path $HOME 'nordicai\drone-captures'
python teammate_preflight.py --capture-root $CaptureRoot
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }

python cvat_coco_to_reconstruction.py --coco validation_cvat.json --output reconstruction_annotations.json
if ($LASTEXITCODE -ne 0) { throw 'Conversion failed; if already generated, verify before reusing' }
python audit_reconstruction_annotations.py --annotations reconstruction_annotations.json
if ($LASTEXITCODE -ne 0) { throw 'Annotation audit failed' }
# Read the audit, especially classes with no human labels, before proceeding.

python project_reconstruction_labels.py --capture-root $CaptureRoot --annotations reconstruction_annotations.json --output "$CaptureRoot\projected_l1_labels" --min-visible-fraction 0.7
if ($LASTEXITCODE -ne 0) { throw 'Projection failed' }
python build_validation_yolo_dataset.py --capture-root $CaptureRoot --labels-root "$CaptureRoot\projected_l1_labels" --output "$CaptureRoot\validation_yolo"
if ($LASTEXITCODE -ne 0) { throw 'Validation dataset build failed' }
python prepare_combined_finetune_dataset.py --helsinki-root datasets/helsinki_multires --validation-root "$CaptureRoot\validation_yolo" --output "$CaptureRoot\combined_finetune" --validation-holdout-start-frame 200 --validation-holdout-end-frame 249
if ($LASTEXITCODE -ne 0) { throw 'Combined dataset build failed' }
python teammate_preflight.py --capture-root $CaptureRoot --require-dataset
```

This places the small converted `reconstruction_annotations.json` next to the
scripts so it can be committed. All original and generated image data remains
outside normal Git. Generated combined YAML points to your absolute local dataset
path. Versioned templates/config choices live in `configs/`; do not copy another
machine's absolute data paths without checking them.

Fresh training, explicitly launched by the teammate after reviewing the data:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_validation_finetune.ps1 -CaptureRoot $CaptureRoot -RunName validation_finetune_v1 -Epochs 100 -Batch 4 -Device 0
```

Resume, only after downloading a resumable run and preparing the same dataset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_validation_finetune.ps1 -CaptureRoot $CaptureRoot -RunName validation_finetune_v1 -Resume -Batch 4 -Device 0
```

Optional old L0 diagnostic, **user-run only**, unrelated to training data generation:

```powershell
python evaluate_validation_annotations.py --coco validation_cvat.json --capture-root $CaptureRoot --weights runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt --device 0 --conf 0.001 --iou 0.5 --prediction-conf-report 0.05 --save-predictions validation_predictions.json --save-errors validation_errors.csv
```

The older standalone utilities default to `~/drone-captures`. These commands
explicitly override that with the agreed `$HOME\nordicai\drone-captures` location.
The new handoff preflight and launcher default directly to the agreed location.
