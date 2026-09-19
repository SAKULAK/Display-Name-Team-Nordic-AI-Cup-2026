# Detector experiments

Use the dedicated environment, not Anaconda base:

```powershell
conda activate drone-yolo
python -m pip install -r requirements.txt
python -m unittest test_yolo -v
```

`dtos.py`, `utils.py`, `local_evaluator.py`, and the response schema are unchanged.

## Serving

```powershell
$env:YOLO_WEIGHTS = (Resolve-Path 'runs/detect/helsinki-2/weights/best.pt').Path
$env:CAMERA_POLICY = 'hold_full'
$env:YOLO_CONF = '0.05'
$env:YOLO_IOU = '0.5'
$env:YOLO_AGNOSTIC_NMS = 'true'
$env:YOLO_MAX_DET = '500'
$env:YOLO_DEVICE = '0'
$env:YOLO_LOG_EVERY = '100'
python api.py
```

Defaults: `CAMERA_POLICY=hold_full`, `YOLO_CONF=0.25`, `YOLO_IOU=0.5`,
`YOLO_AGNOSTIC_NMS=true`, `YOLO_MAX_DET=500`, `YOLO_LOG_EVERY=0` (disabled).
`YOLO_WEIGHTS` defaults to `runs/detect/helsinki/weights/best.pt`; explicitly select
another run when that file does not exist. `YOLO_DEVICE` defaults to auto.
NMS accepts true/false, 1/0, yes/no (case-insensitive); invalid strings fail.
Maximum detections must be 1?500. Restart the API after changing detector settings.
Missing or incompatible weights retain the existing logged empty-response fallback.

`hold_full` stays at L0 and resets from L1 to (1920,1080). The official evaluator
forbids L2 ? L0, even though its comment suggests a one-move reset: we honor its
actual constraints and return via L1. `baseline_sweep` retains the original policy.

## Repeatable official scoring

No existing API is required. Each experiment starts its own warmed API on an
available localhost port, then shuts it down; existing servers are not touched.
The runner uses the current Python interpreter and a fresh detector cache for
every combination. Never benchmark latency while training another model.

```powershell
python benchmark_detector.py --models runs/detect/helsinki-2/weights/best.pt --confidence 0.05 0.10 0.15 0.25 --iou 0.5 --camera-policies hold_full --agnostic-nms true
python benchmark_detector.py --models runs/detect/helsinki-2/weights/best.pt --confidence 0.05 --agnostic-nms false
```

Results append to `benchmarks/detector_sweep.csv`. Logs and per-run environment
metadata live under `benchmarks/logs/`. Rows include weight SHA256, official mAP50,
wall time, configuration, synchronized mean inference milliseconds and approximate
inference FPS. Failed experiments get explicit status and a nonzero script exit.
The official score is parsed at its printed precision (three decimals).

Timing covers YOLO predict (preprocessing, forward pass, NMS and CPU result transfer),
excluding model load, three warmup calls, HTTP transport, image decoding and global
coordinate conversion. `elapsed_seconds` includes server lifecycle and evaluation.
Use `--realtime` to measure actual frame-clock losses; inference FPS alone does not
prove endpoint throughput. API round-trip statistics are retained in evaluator logs.

The official Helsinki replay scores all 25 frames, including the 20 training
frames. Treat these as local diagnostic scores, not held-out generalization.
Confidence tuning on this scene also makes it unsuitable as an independent test.

## Training, changing one variable at a time

`prepare_yolo.py` is unchanged. It produces a temporal 20/5 split, source-normalized
YOLO labels and `datasets/helsinki/data.yaml`. Conversion refuses existing output;
use a new `--output` when regenerating. Dataset YAML paths are absolute: update or
regenerate them after moving the dataset.

Training supports nano/small/medium pretrained checkpoints through `--model`.
`--name` and `--run-name` are aliases. All runs default to 100 epochs, 960px, batch 4,
seed 42, workers 0. Augmentation defaults are degrees=180, flipud=0.5, fliplr=0.5,
scale=0.5, translate=0.1, mosaic=1.0. Every one is configurable.

Requested small-model training (not automatically launched):

```powershell
python train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 960 --device 0 --batch 4 --degrees 180 --flipud 0.5 --fliplr 0.5 --name helsinki_yolov8s_rot
```

This changes both model size and augmentations relative to the existing nano.
For a controlled model-size comparison, first train a matching nano control:

```powershell
python train_yolo.py --model yolov8n.pt --epochs 100 --imgsz 960 --device 0 --batch 4 --degrees 180 --flipud 0.5 --fliplr 0.5 --name helsinki_yolov8n_rot
python benchmark_models.py --models runs/detect/helsinki_yolov8n_rot/weights/best.pt runs/detect/helsinki_yolov8s_rot/weights/best.pt --confidence 0.05 --agnostic-nms true --output benchmarks/model_comparison.csv
```

To reproduce the previous augmentation defaults instead, use `--degrees 0 --flipud 0`.
Ultralytics may suffix repeated run names; use the checkpoint path printed by training.
Official [augmentation reference](https://docs.ultralytics.com/guides/yolo-data-augmentation/).

## Multi-resolution and scout data

```powershell
python prepare_multires_yolo.py --l1-crops-per-object 1 --l2-crops-per-object 2 --background-crops 2 --seed 42
python prepare_multires_yolo.py --scout
```

Outputs default to `datasets/helsinki_multires` and `datasets/helsinki_scout`.
Use a new `--output` if already generated. L0 downsamples the full frame; L1 crops
1920x1080; L2 crops 960x540. Every saved image is 960x540. Crops include all intersecting
objects, clipping at the crop boundary. Object centers receive up to 30% crop-size
jitter. Background requests are per frame per level, use bounded rejection sampling,
and only create empty labels after verifying no annotation intersects the crop.
Any background shortfall is reported rather than assigning false negative labels.

`manifest.json` records source frame, split, crop region, level, seed, options and
sample counts. The final 20% of source frames are validation; all derivatives of one
frame stay together. Persistent physical objects still overlap across time, so this
is not a split by object identity. Scout mode emits only L0 and a single `target`
class; it is intentionally not accepted by the 16-class inference pipeline.

To isolate the dataset change, repeat an existing model/augmentation configuration
with only `--data datasets/helsinki_multires/data.yaml` and `--name` changed.
