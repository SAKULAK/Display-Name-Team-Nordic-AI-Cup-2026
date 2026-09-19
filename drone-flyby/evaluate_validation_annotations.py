"""User-run offline L0 inference against CVAT reconstruction annotations. Never trains."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from cvat_coco_to_reconstruction import convert_coco
from dtos import OBJECT_CLASSES
from project_reconstruction_labels import reconstruction_to_source


def normalize_reconstruction(box):
    source = reconstruction_to_source(box)
    return [value/scale for value, scale in zip(source, (3840, 2160, 3840, 2160))]


def load_ground_truth(path):
    converted, _ = convert_coco(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    return {item['frame']: [dict(object_id=a['object_id'], bbox=normalize_reconstruction(a['bbox']))
                           for a in item['annotations']] for item in converted['frames']}


def select_l0_captures(root, frames):
    """Same main-sequence rule as reconstruction: most distinct frames, lexical tie.

    Read l0_repeat* runs only; choose first lexical path when runs repeat a frame.
    Main-sequence selection uses all paired received levels before filtering to L0.
    """
    selected, sequences = {}, {}
    for run in sorted(Path(root).glob('l0_repeat*')):
        if not run.is_dir():
            continue
        groups = defaultdict(list)
        for sidecar in sorted(run.rglob('*.json')):
            image = sidecar.with_suffix('.png')
            if not image.is_file():
                continue
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            if type(metadata['frame']) is not int or metadata['frame'] < 0:
                raise ValueError(f'Invalid source frame: {sidecar}')
            groups[str(metadata['sequence_id'])].append((image, metadata))
        if not groups:
            continue
        main = min(groups, key=lambda key: (-len({m['frame'] for _, m in groups[key]}), key))
        sequences[run.name] = main
        for image, metadata in groups[main]:
            frame = metadata['frame']
            if metadata['resolution_level'] != 0 or frame not in frames:
                continue
            if (metadata['original_width'], metadata['original_height']) != (3840, 2160):
                raise ValueError(f'Expected 3840x2160 source dimensions: {image}')
            if metadata['source_region_xyxy'] != [0, 0, 3840, 2160]:
                raise ValueError(f'L0 must represent the complete source frame: {image}')
            if (metadata['transmitted_width'], metadata['transmitted_height']) != (960, 540):
                raise ValueError(f'Expected transmitted dimensions 960x540: {image}')
            selected.setdefault(frame, (image, metadata))
    return selected, sequences


def read_l0(image, metadata):
    raw = image.read_bytes()
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or hashlib.sha256(raw).hexdigest() != metadata['image_sha256']:
        raise ValueError(f'PNG signature or SHA256 mismatch: {image}')
    pixels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.shape != (540, 960, 3) or pixels.dtype != np.uint8:
        raise ValueError(f'Expected exact 960x540 uint8 three-channel L0 image: {image}')
    return pixels


def xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2-x1, y2-y1]


def score_ap(ground_truth, predictions):
    """Mirror local_evaluator.score, without importing/running its replay machinery.

    Both inputs are full-source normalized XYXY. As in local_evaluator, keep
    COCO's default maxDets, use only IoU .50, area=all, and GT-present classes.
    """
    from faster_coco_eval import COCO, COCOeval_faster
    present = {a['object_id'] for annotations in ground_truth.values() for a in annotations}
    classes = [name for name in OBJECT_CLASSES if name in present]
    ids = {name: index for index, name in enumerate(OBJECT_CLASSES, 1)}
    frames = sorted(ground_truth)
    image_ids = {frame: index for index, frame in enumerate(frames, 1)}
    annotations, detections = [], []
    for frame in frames:
        for gt in ground_truth[frame]:
            box = xywh(gt['bbox'])
            annotations.append(dict(id=len(annotations)+1, image_id=image_ids[frame],
                category_id=ids[gt['object_id']], bbox=box, area=box[2]*box[3], iscrowd=0))
        for prediction in predictions.get(frame, []):
            detections.append(dict(image_id=image_ids[frame], category_id=ids[prediction['object_id']],
                                   bbox=xywh(prediction['bbox']), score=prediction['confidence']))
    dataset = dict(info={'description': 'Human-reviewed Nordic validation subset'}, licenses=[],
        images=[dict(id=image_ids[f], file_name=f'frame_{f:06d}.png', width=3840, height=2160) for f in frames],
        categories=[dict(id=ids[name], name=name, supercategory='object') for name in OBJECT_CLASSES],
        annotations=annotations)
    gt_api = COCO(dataset)
    # Instantiate even for empty predictions to report the same backend default cap.
    evaluator = COCOeval_faster(gt_api, iouType='bbox')
    cap = evaluator.params.maxDets[-1]
    result = {name: None for name in OBJECT_CLASSES}
    if not classes:
        return None, result, cap
    if not detections:
        result.update({name: 0.0 for name in classes})
        return 0.0, result, cap
    evaluator.cocoDt = gt_api.loadRes(detections)
    evaluator.params.imgIds = list(image_ids.values())
    evaluator.params.catIds = [ids[name] for name in classes]
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()
    for index, name in enumerate(classes):
        precision = evaluator.eval['precision'][0, :, index, 0, -1]
        valid = precision[precision > -1]
        result[name] = max(0.0, min(1.0, float(valid.mean()) if valid.size else 0.0))
    return sum(result[name] for name in classes)/len(classes), result, cap


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def error_rows(ground_truth, predictions, matched_frames, max_dets):
    """Class-aware one-to-one matches; separately report best spatial prediction of any class."""
    rows = []
    for frame, annotations in sorted(ground_truth.items()):
        live, assignments = predictions.get(frame, []), {}
        for name in OBJECT_CLASSES:
            candidates = [i for i, p in enumerate(live) if p['object_id'] == name]
            candidates.sort(key=lambda i: (-live[i]['confidence'], i))
            for i in candidates[:max_dets]:
                # COCO prefers the later GT at exactly tied overlaps (no crowd/ignore GT here).
                possible = [(box_iou(gt['bbox'], live[i]['bbox']), j) for j, gt in enumerate(annotations)
                            if gt['object_id'] == name and j not in assignments]
                overlap, j = max(possible, default=(0.0, -1))
                if overlap >= 0.5:
                    assignments[j] = i
        for index, gt in enumerate(annotations):
            best = max(range(len(live)), key=lambda i: (box_iou(gt['bbox'], live[i]['bbox']),
                       live[i]['confidence'], -i), default=None)
            best_iou = box_iou(gt['bbox'], live[best]['bbox']) if best is not None else 0.0
            best_prediction = live[best] if best_iou > 0 else None
            matched = live[assignments[index]] if index in assignments else None
            rows.append(dict(frame=frame, gt_index=index, gt_class=gt['object_id'],
                gt_bbox=json.dumps(gt['bbox']), capture_available=frame in matched_frames,
                matched=matched is not None, matched_class=matched['object_id'] if matched else '',
                matched_confidence=matched['confidence'] if matched else '',
                matched_iou=box_iou(gt['bbox'], matched['bbox']) if matched else '',
                best_predicted_class=best_prediction['object_id'] if best_prediction else '',
                best_predicted_confidence=best_prediction['confidence'] if best_prediction else '',
                best_predicted_iou=best_iou))
    return rows


def evaluate(args):
    if not (0 <= args.conf <= args.prediction_conf_report <= 1 and 0 <= args.iou <= 1):
        raise ValueError('Require 0 <= conf <= prediction-conf-report <= 1 and 0 <= iou <= 1')
    coco, root, weights = (Path(p).expanduser().resolve() for p in (args.coco, args.capture_root, args.weights))
    outputs = [Path(p).expanduser().resolve() for p in (args.save_predictions, args.save_errors) if p]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Prediction and error output paths must differ')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Choose a new report path: {path}')
    gt = load_ground_truth(coco)
    if not gt:
        raise ValueError('COCO export has no reviewed images')
    captures, sequences = select_l0_captures(root, gt)
    missing = sorted(set(gt)-set(captures))
    print(f'Reviewed frames: {len(gt)}\nMatched L0 frames: {len(captures)}\nMissing L0 frames: {len(missing)}')
    print(f'Selected main sequences: {json.dumps(sequences, sort_keys=True)}')
    if missing:
        print(f'Missing source frames: {missing}; scored with empty predictions, not excluded.')
    # Validate every selected image before loading the checkpoint; never guess sizes or resize.
    for image, metadata in captures.values():
        read_l0(image, metadata)
    predictions = {frame: [] for frame in gt}
    if captures:
        from detector import YoloDetector
        detector = YoloDetector(weights, confidence=args.conf, iou=args.iou,
                                device=args.device, agnostic_nms=True, max_det=500)
        for number, (frame, (image, metadata)) in enumerate(sorted(captures.items()), 1):
            request = SimpleNamespace(original_width=3840, original_height=2160,
                                     view=SimpleNamespace(source_region_xyxy=[0, 0, 3840, 2160]))
            predictions[frame] = [d.model_dump() for d in detector.detect(read_l0(image, metadata), request)]
            if number % 20 == 0 or number == len(captures):
                print(f'Inference: {number}/{len(captures)} captured L0 frames')
    mean, per_class, cap = score_ap(gt, predictions)
    counts = Counter(a['object_id'] for annotations in gt.values() for a in annotations)
    total_predictions = sum(len(items) for items in predictions.values())
    report_count = sum(p['confidence'] >= args.prediction_conf_report for items in predictions.values() for p in items)
    print(f'Total GT boxes: {sum(counts.values())}\nTotal predictions (conf >= {args.conf:g}): {total_predictions}')
    print(f'Predictions at endpoint reporting threshold {args.prediction_conf_report:g}: {report_count}')
    print(f'AP uses all low-threshold predictions; COCO maxDets={cap} per image/class, as in local_evaluator.')
    print('AP@0.50 by class')
    for name in OBJECT_CLASSES:
        value = per_class[name]
        print(f'  {name}: {"N/A" if value is None else f"{value:.6f}"} (GT={counts[name]})')
    print('Zero GT classes (excluded from subset mean): '+', '.join(name for name in OBJECT_CLASSES if not counts[name]))
    print('COCO mAP@0.50: '+('N/A' if mean is None else f'{mean:.6f}'))
    if args.save_predictions:
        path = Path(args.save_predictions).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(coordinate_system='full_source_normalized_xyxy', source_size=[3840, 2160],
            reconstruction_size=[1920, 1080], l0_size=[960, 540], weights=str(weights),
            settings=dict(conf=args.conf, iou=args.iou, imgsz=960, agnostic_nms=True, max_det=500),
            prediction_conf_report=args.prediction_conf_report, reporting_threshold_count=report_count,
            missing_l0_frames=missing, selected_sequences=sequences, coco_max_dets=cap,
            map50=mean, ap50=per_class,
            frames=[dict(frame=frame, capture_path=str(captures[frame][0]) if frame in captures else None,
                         predictions=predictions[frame]) for frame in sorted(gt)])
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    if args.save_errors:
        path = Path(args.save_errors).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ('frame', 'gt_index', 'gt_class', 'gt_bbox', 'capture_available', 'matched',
                  'matched_class', 'matched_confidence', 'matched_iou', 'best_predicted_class',
                  'best_predicted_confidence', 'best_predicted_iou')
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(error_rows(gt, predictions, captures, cap))
    return mean, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco', required=True)
    parser.add_argument('--capture-root', default='~/drone-captures')
    parser.add_argument('--weights', default='runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt')
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--prediction-conf-report', type=float, default=0.05)
    parser.add_argument('--save-predictions')
    parser.add_argument('--save-errors')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
