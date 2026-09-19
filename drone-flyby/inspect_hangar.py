import json
from pathlib import Path

import cv2
from ultralytics import YOLO


WEIGHTS = Path("runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt")
FRAMES = [22, 23, 24]

# Deliberately much lower than the submission threshold.
# We want to see whether YOLO has *any* belief about the hangar.
CONF = 0.001
IOU = 0.5
IMGSZ = 960


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_file(filename, wanted_type):
    candidates = list(Path(".").rglob(filename))

    # Avoid generated/visualization outputs where possible.
    filtered = [
        p for p in candidates
        if not any(part.lower() in {"annotated", "runs"} for part in p.parts)
    ]

    if filtered:
        candidates = filtered

    if wanted_type == "image":
        candidates = [
            p for p in candidates
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

    if not candidates:
        return None

    if len(candidates) > 1:
        print(f"\nMultiple candidates for {filename}:")
        for p in candidates:
            print("   ", p)
        print(f"Using: {candidates[0]}")

    return candidates[0]


def load_hangar_box(annotation_path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    objects = data.get("annotations", data.get("objects", []))

    for obj in objects:
        if obj.get("object_id") == "hangar":
            return list(map(float, obj["bbox"]))

    return None


model = YOLO(str(WEIGHTS))

print("weights:", WEIGHTS)
print("classes:", model.names)
print()

for frame in FRAMES:
    stem = f"frame_{frame:06d}"

    image_path = find_file(stem + ".png", "image")
    if image_path is None:
        image_path = find_file(stem + ".jpg", "image")

    annotation_path = find_file(stem + ".json", "annotation")

    print("=" * 80)
    print(f"FRAME {frame}")

    if image_path is None:
        print("IMAGE NOT FOUND")
        continue

    if annotation_path is None:
        print("ANNOTATION NOT FOUND")
        continue

    print("image:     ", image_path)
    print("annotation:", annotation_path)

    gt = load_hangar_box(annotation_path)

    if gt is None:
        print("No hangar ground truth in this frame.")
        continue

    print("hangar GT:", [round(v, 1) for v in gt])

    image = cv2.imread(str(image_path))
    if image is None:
        print("Could not read image.")
        continue

    results = model.predict(
        image,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        agnostic_nms=True,
        max_det=500,
        device=0,
        verbose=False,
    )

    predictions = []

    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]

        overlap = iou(xyxy, gt)

        gx = (gt[0] + gt[2]) / 2
        gy = (gt[1] + gt[3]) / 2
        px = (xyxy[0] + xyxy[2]) / 2
        py = (xyxy[1] + xyxy[3]) / 2

        gt_w = gt[2] - gt[0]
        gt_h = gt[3] - gt[1]

        # Include anything overlapping the GT or reasonably near its center.
        near = (
            abs(px - gx) <= max(gt_w, 50) * 1.5
            and abs(py - gy) <= max(gt_h, 50) * 1.5
        )

        if overlap > 0 or near:
            predictions.append(
                (overlap, conf, name, xyxy)
            )

    predictions.sort(key=lambda x: (-x[0], -x[1]))

    if not predictions:
        print("\nNO YOLO PREDICTIONS NEAR THE HANGAR, even at conf=0.001")
        continue

    print("\nPredictions near hangar:")
    for overlap, conf, name, xyxy in predictions:
        print(
            f"  {name:20s}"
            f" conf={conf:.4f}"
            f" IoU={overlap:.3f}"
            f" bbox={[round(v, 1) for v in xyxy]}"
        )