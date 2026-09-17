# Detector baseline, 2026-09-17

Checkpoint: `runs/detect/helsinki-2/weights/best.pt` (YOLOv8n).
SHA256: `e6446d2f061959438165ae2d0c52e2930ceb903d4652067b21400406b1dd6992`.
Hardware: NVIDIA RTX 4070 Ti; Python 3.12, torch 2.11.0+cu128, Ultralytics 8.4.154.
All experiments use 960px inference, IoU 0.5 and max_det 500. Scores are official
evaluator output rounded to three decimals. Full rows are in `detector_sweep.csv`.

| Camera | Confidence | Agnostic NMS | Realtime | mAP50 | Mean inference ms |
|---|---:|---|---|---:|---:|
| hold_full | 0.05 | true | false | 0.388 | 18.22 |
| hold_full | 0.10 | true | false | 0.356 | 15.15 |
| hold_full | 0.15 | true | false | 0.351 | 16.62 |
| hold_full | 0.25 | true | false | 0.344 | 16.69 |
| hold_full | 0.05 | false | false | 0.381 | 16.54 |
| baseline_sweep | 0.05 | true | false | 0.022 | 16.95 |
| hold_full | 0.05 | true | true | 0.388 | 15.68 |

Best tested confidence: **0.05**. Agnostic NMS improved mAP50 by **0.007** over
class-aware NMS. This is one short local sequence, not enough to infer universal
superiority. The realtime best configuration accepted all 25 frames: zero skipped,
unanswered, invalid, or timed-out frames. API round-trip mean/median/max was
46/47/109 ms. Mean inference was 15.68 ms (~63.8 inference FPS), comfortably under
333 ms on this GPU. Timing excludes three warmup calls and model load, includes
preprocessing/NMS/CPU result transfer, and is not a statistical latency distribution.

All 25 frames were scored, including training frames. These are diagnostic baseline
scores, not independent held-out accuracy. NMS and confidence sweeps were sequential;
no training competed for GPU resources. No long training runs were launched.
YOLOv8s score and latency therefore remain unmeasured.

## Data generated and validated

`datasets/helsinki_multires`: **834 images**, 684 train / 150 validation.

- L0: 25.
- L1: 255.
- L2: 554.
- Verified background crops: 58 (included in L1/L2 totals).
- Requested negative crops not found within the search budget: 42.

All images are 960x540. All labels were checked for finite, valid normalized boxes
and class IDs. Source-frame sets do not overlap between splits. Every source class
has at least one annotation. Persistent physical objects still cross the temporal
split. `manifest.json` records every region and split for reproducibility.

`datasets/helsinki_scout`: 25 L0 images, 20 train / 5 val, one `target` class.
Scout generation is available; scout inference is intentionally not integrated.

## Implementation and checks

Created: `benchmark_detector.py`, `benchmark_models.py`, `benchmark_server.py`,
`prepare_multires_yolo.py`, this report, CSV results and generated datasets.

Modified: `example.py`, `detector.py`, `train_yolo.py`, `test_yolo.py`, `YOLO.md`,
`.gitignore`. The pre-existing requirements change was preserved.

Unchanged: `dtos.py`, `utils.py`, `local_evaluator.py`, `api.py`, `prepare_yolo.py`
and the evaluation protocol. No tracking, persistent object memory or active zoom.

Nine unit tests passed using `python -m unittest test_yolo -v`: label/class order,
all three view geometries, malformed detection rejection, policy selection,
official camera legality, boolean parsing, detector configuration/class guard,
crop dimensions/clipping/negative labels, split determinism and score extraction.
Additionally validated every generated image and label, ran seven successful
official evaluator experiments, and checked `git diff --check`.

Constraint discrepancy: L2 → L0 is forbidden by the official allowed-level table,
despite the evaluator's reset comment. `hold_full` returns via L1, then L0. The
default checkpoint path refers to `helsinki`, but the trained weights are in
`helsinki-2`; explicitly set `YOLO_WEIGHTS` or pass `--models`.

## Next commands

From `drone-flyby`, use the existing environment:

```powershell
conda activate drone-yolo
python train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 960 --device 0 --batch 4 --degrees 180 --flipud 0.5 --fliplr 0.5 --name helsinki_yolov8s_rot
```

To isolate architecture from the augmentation change, train a matched nano control
before interpreting small-versus-nano differences:

```powershell
python train_yolo.py --model yolov8n.pt --epochs 100 --imgsz 960 --device 0 --batch 4 --degrees 180 --flipud 0.5 --fliplr 0.5 --name helsinki_yolov8n_rot
python benchmark_models.py --models runs/detect/helsinki_yolov8n_rot/weights/best.pt runs/detect/helsinki_yolov8s_rot/weights/best.pt --confidence 0.05 --agnostic-nms true --output benchmarks/model_comparison.csv
```

For the next dataset-only comparison, retain the matched nano settings:

```powershell
python train_yolo.py --model yolov8n.pt --data datasets/helsinki_multires/data.yaml --epochs 100 --imgsz 960 --device 0 --batch 4 --degrees 180 --flipud 0.5 --fliplr 0.5 --name helsinki_yolov8n_multires_rot
```

Use actual printed checkpoint paths if Ultralytics suffixes repeated run names.
See `../YOLO.md` for serving, all environment variables and benchmark commands.
