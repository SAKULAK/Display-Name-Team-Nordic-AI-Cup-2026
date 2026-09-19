"""Synthetic AP and geometry tests; never load a checkpoint or run inference."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from evaluate_validation_annotations import (box_iou, error_rows, load_ground_truth,
    normalize_reconstruction, read_l0, score_ap, select_l0_captures, evaluate)
from dtos import DroneFlybyPredictionDto


def annotation(name='ta-ta', box=(0.1, 0.1, 0.2, 0.2), confidence=None):
    result = dict(object_id=name, bbox=list(box))
    if confidence is not None:
        result['confidence'] = confidence
    return result


class EvaluationTests(unittest.TestCase):
    def score(self, gt, predictions):
        with redirect_stdout(io.StringIO()):
            return score_ap(gt, predictions)

    def test_reconstruction_and_l0_have_identical_normalized_geometry(self):
        reconstructed = normalize_reconstruction([192, 108, 384, 216])
        l0_pixels = [96, 54, 192, 108]
        l0_normalized = [v / scale for v, scale in zip(l0_pixels, (960, 540, 960, 540))]
        self.assertEqual(reconstructed, l0_normalized)
        self.assertEqual(reconstructed, [0.1, 0.1, 0.2, 0.2])
        self.assertAlmostEqual(box_iou(reconstructed, l0_normalized), 1)

    def test_perfect_ap_category_name_mapping_empty_reviewed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coco.json'
            path.write_text(json.dumps(dict(categories=[dict(id=90, name='ta-ta')],
                images=[dict(id=9, width=1920, height=1080, file_name='frame_000123.png'),
                        dict(id=8, width=1920, height=1080, file_name='frame_000124.png')],
                annotations=[dict(image_id=9, category_id=90, bbox=[192, 108, 192, 108])])))
            gt = load_ground_truth(path)
        self.assertEqual(gt, {123: [annotation()], 124: []})
        mean, classes, cap = self.score(gt, {123: [annotation(confidence=0.9)]})
        self.assertAlmostEqual(mean, 1)
        self.assertAlmostEqual(classes['ta-ta'], 1)
        self.assertIsNone(classes['hangar'])
        self.assertEqual(cap, 100)

    def test_wrong_class_and_no_predictions_score_zero(self):
        gt = {1: [annotation()]}
        for predictions in ({}, {1: [annotation('tank', confidence=0.99)]}):
            mean, classes, _ = self.score(gt, predictions)
            self.assertEqual(mean, 0)
            self.assertEqual(classes['ta-ta'], 0)
            self.assertIsNone(classes['tank'])

    def test_false_positive_on_reviewed_empty_frame_costs_precision(self):
        gt = {1: [annotation()], 2: []}
        mean, _, _ = self.score(gt, {1: [annotation(confidence=0.5)], 2: [annotation(confidence=0.9)]})
        self.assertAlmostEqual(mean, 0.5)

    def test_missing_frame_gt_costs_recall_and_absent_classes_not_in_mean(self):
        mean, classes, _ = self.score({1: [annotation()], 2: [annotation()]},
                                      {1: [annotation(confidence=0.8)]})
        self.assertAlmostEqual(mean, 51/101)  # COCO's 101-point interpolated recall grid.
        self.assertEqual(mean, classes['ta-ta'])
        self.assertTrue(all(value is None for name, value in classes.items() if name != 'ta-ta'))

    def test_all_empty_ground_truth_is_na(self):
        mean, classes, _ = self.score({1: []}, {1: [annotation(confidence=0.8)]})
        self.assertIsNone(mean)
        self.assertTrue(all(value is None for value in classes.values()))

    def test_errors_are_one_to_one_and_report_wrong_class_spatial_match(self):
        gt = {1: [annotation(), annotation()], 2: [annotation()], 3: [annotation()]}
        predictions = {1: [annotation(confidence=0.8)], 2: [annotation('tank', confidence=0.9)]}
        rows = error_rows(gt, predictions, {1, 2}, 100)
        self.assertEqual(sum(r['matched'] for r in rows), 1)
        self.assertFalse(rows[2]['matched'])
        self.assertEqual(rows[2]['best_predicted_class'], 'tank')
        self.assertEqual(rows[2]['best_predicted_iou'], 1)
        self.assertFalse(rows[3]['capture_available'])
        self.assertFalse(rows[3]['matched'])

    def test_iou_threshold_is_inclusive(self):
        gt = {1: [annotation(box=(0, 0, 0.5, 0.5))]}
        predictions = {1: [annotation(box=(0, 0, 0.25, 0.5), confidence=0.9)]}
        self.assertEqual(box_iou(gt[1][0]['bbox'], predictions[1][0]['bbox']), 0.5)
        self.assertTrue(error_rows(gt, predictions, {1}, 100)[0]['matched'])
        self.assertAlmostEqual(self.score(gt, predictions)[0], 1)

    def test_l0_selection_filters_verify_other_levels_and_unreviewed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def capture(run, sequence, frame, level=0):
                path = root/run/sequence/f'frame_{frame:06d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
                path.with_suffix('.json').write_text(json.dumps(dict(sequence_id=sequence, frame=frame,
                    resolution_level=level, original_width=3840, original_height=2160,
                    transmitted_width=960, transmitted_height=540, source_region_xyxy=[0, 0, 3840, 2160])))
                return path
            first = capture('l0_repeat_00', 'main', 123)
            capture('l0_repeat_00', 'main', 124, level=1)
            capture('l0_repeat_00', 'main', 125)
            capture('l0_repeat_00', 'Verify', 1)
            capture('l0_repeat_01', 'main', 123)
            capture('l1_phase_00', 'main', 124)
            selected, sequences = select_l0_captures(root, {123, 124})
            self.assertEqual(set(selected), {123})
            self.assertEqual(selected[123][0], first)
            self.assertEqual(sequences['l0_repeat_00'], 'main')

    def test_image_dimensions_and_hash_checked_without_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frame.png'
            cv2.imwrite(str(path), np.zeros((540, 960, 3), dtype=np.uint8))
            metadata = dict(image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(read_l0(path, metadata).shape, (540, 960, 3))
            cv2.imwrite(str(path), np.zeros((54, 96, 3), dtype=np.uint8))
            metadata['image_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, '960x540'):
                read_l0(path, metadata)
            metadata['image_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                read_l0(path, metadata)

    def test_mocked_runner_preserves_low_confidence_ap_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(coco=str(root/'coco.json'), capture_root=str(root),
                weights=str(root/'unused.pt'), device='0', conf=0.001, iou=0.5,
                prediction_conf_report=0.05, save_predictions=str(root/'predictions.json'),
                save_errors=str(root/'errors.csv'))
            gt = {123: [annotation()]}
            captures = {123: (root/'capture.png', {})}
            with patch('evaluate_validation_annotations.load_ground_truth', return_value=gt), \
                    patch('evaluate_validation_annotations.select_l0_captures', return_value=(captures, {})), \
                    patch('evaluate_validation_annotations.read_l0', return_value=np.zeros((540, 960, 3), dtype=np.uint8)), \
                    patch('detector.YoloDetector') as factory, redirect_stdout(io.StringIO()):
                factory.return_value.detect.return_value = [DroneFlybyPredictionDto(**annotation(confidence=0.01))]
                mean, _ = evaluate(args)
            self.assertAlmostEqual(mean, 1)
            factory.assert_called_once_with((root/'unused.pt').resolve(), confidence=0.001, iou=0.5,
                                            device='0', agnostic_nms=True, max_det=500)
            factory.return_value.detect.assert_called_once()
            payload = json.loads((root/'predictions.json').read_text())
            self.assertEqual(payload['reporting_threshold_count'], 0)
            self.assertEqual(payload['frames'][0]['predictions'][0]['confidence'], 0.01)
            self.assertIn('True', (root/'errors.csv').read_text())


if __name__ == '__main__':
    unittest.main()
