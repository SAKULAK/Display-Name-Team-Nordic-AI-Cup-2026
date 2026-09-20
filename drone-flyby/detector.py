"""Lazy YOLO inference for decoded BGR views and Nordic frame-global DTOs."""

import os
import logging
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto
from utils import clip_bbox_to_frame, view_bbox_to_global

ROOT = Path(__file__).resolve().parent


logger = logging.getLogger(__name__)


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ('true', '1', 'yes'):
        return True
    if normalized in ('false', '0', 'no'):
        return False
    raise ValueError(f'Expected true/false, 1/0 or yes/no, got {value!r}')


class YoloDetector:
    def __init__(self, weights, confidence=0.25, iou=0.5, device=None, agnostic_nms=True, max_det=500, log_every=0):
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError('Confidence and IoU thresholds must be in [0, 1]')
        if type(max_det) is not int or not 1 <= max_det <= 500:
            raise ValueError('max_det must be an integer in [1, 500]')
        if type(log_every) is not int or log_every < 0:
            raise ValueError('log_every must be a nonnegative integer')
        self.agnostic_nms = parse_bool(agnostic_nms)
        self.max_det, self.log_every = max_det, log_every
        self.number_of_calls = 0
        self.total_inference_seconds = 0.0
        if not Path(weights).is_file():
            raise FileNotFoundError(f'Train Helsinki first or set YOLO_WEIGHTS: {weights}')
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        names = self.model.names
        ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
        if tuple(ordered) != OBJECT_CLASSES:
            raise ValueError('Checkpoint classes must exactly match dtos.OBJECT_CLASSES')
        self.confidence, self.iou, self.device = confidence, iou, device
        self.lock = Lock()
        # First inference is much slower than the rest (CUDA context, kernel
        # autotuning, etc.) and there is no separate timing allowance for it
        # in an attempt. Pay that cost now, not on frame 0.
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=960, conf=self.confidence, iou=self.iou,
                            agnostic_nms=self.agnostic_nms, max_det=1,
                            device=self.device, verbose=False)

    def detect(self, image, request):
        if image.shape != (540, 960, 3) or image.dtype != np.uint8:
            raise ValueError('Expected a decoded 960x540 uint8 BGR image')
        # YOLOv8 applies NMS; class-agnostic suppression removes competing labels too.
        with self.lock:
            started = time.perf_counter()
            result = self.model.predict(image, imgsz=960, conf=self.confidence,
                                        iou=self.iou, agnostic_nms=self.agnostic_nms, max_det=self.max_det,
                                        device=self.device, verbose=False)[0]
            # Moving results to CPU synchronizes CUDA before stopping the timer.
            rows = [] if result.boxes is None else result.boxes.data.cpu().tolist()
            self.total_inference_seconds += time.perf_counter() - started
            self.number_of_calls += 1
            if self.log_every and self.number_of_calls % self.log_every == 0:
                logger.info('YOLO calls=%d average_ms=%.3f', self.number_of_calls,
                            1000 * self.total_inference_seconds / self.number_of_calls)
        predictions = []
        if result.boxes is None:
            return predictions
        # xyxy is already scaled back from letterboxing to the original view pixels.
        for row in rows:
            if len(row) != 6:
                continue
            x1, y1, x2, y2, confidence, class_id = row
            if not np.isfinite(row).all() or not self.confidence <= confidence <= 1:
                continue
            if not float(class_id).is_integer() or not 0 <= class_id < len(OBJECT_CLASSES):
                continue
            local = clip_bbox_to_frame((x1 / 960, y1 / 540, x2 / 960, y2 / 540))
            if local is None:
                continue
            bbox = clip_bbox_to_frame(view_bbox_to_global(
                local, request.view.source_region_xyxy,
                request.original_width, request.original_height))
            if bbox is not None:
                predictions.append(DroneFlybyPredictionDto(
                    object_id=OBJECT_CLASSES[int(class_id)], bbox=list(bbox),
                    confidence=float(confidence)))
        return predictions[:self.max_det]


@lru_cache(maxsize=1)
def get_detector():
    return YoloDetector(
        os.environ.get('YOLO_WEIGHTS', str(ROOT / 'runs/detect/helsinki/weights/best.pt')),
        confidence=float(os.environ.get('YOLO_CONF', '0.25')),
        iou=float(os.environ.get('YOLO_IOU', '0.5')),
        device=os.environ.get('YOLO_DEVICE') or None,
        agnostic_nms=parse_bool(os.environ.get('YOLO_AGNOSTIC_NMS', 'true')),
        max_det=int(os.environ.get('YOLO_MAX_DET', '500')),
        log_every=int(os.environ.get('YOLO_LOG_EVERY', '0')),
    )
