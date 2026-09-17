"""Pipeline checks without downloading weights or requiring a GPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml

import example
from detector import YoloDetector
from dtos import OBJECT_CLASSES
from prepare_yolo import convert


class PipelineTests(unittest.TestCase):
    def test_conversion_and_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/images').mkdir(parents=True)
            (root / 'source/annotations').mkdir()
            for i in range(2):
                (root / f'source/images/frame_{i}.png').write_bytes(b'fixture')
                objects = [{'object_id': 'jammer', 'bbox': [960, 540, 1920, 1080]}] if i == 0 else []
                (root / f'source/annotations/frame_{i}.json').write_text(
                    json.dumps({'annotations': objects}))
            config = yaml.safe_load(convert(root / 'source', root / 'output').read_text())
            self.assertEqual(tuple(config['names'].values()), OBJECT_CLASSES)
            fields = (root / 'output/labels/train/frame_0.txt').read_text().split()
            self.assertEqual(int(fields[0]), OBJECT_CLASSES.index('jammer'))
            self.assertEqual(list(map(float, fields[1:])), [0.375, 0.375, 0.25, 0.25])
            self.assertEqual((root / 'output/labels/val/frame_1.txt').read_text(), '')
            with self.assertRaises(FileExistsError):
                convert(root / 'source', root / 'output')

    def test_view_geometry_and_nms_configuration(self):
        from threading import Lock
        detector = YoloDetector.__new__(YoloDetector)
        detector.confidence, detector.iou, detector.device = 0.25, 0.5, None
        detector.agnostic_nms, detector.max_det, detector.log_every = True, 500, 0
        detector.number_of_calls, detector.total_inference_seconds = 0, 0.0
        detector.lock = Lock()
        detector.model = Mock()
        data = Mock()
        data.cpu.return_value.tolist.return_value = [[96., 54., 480., 270., 0.9, 14.], [1, 1, 0, 0, .9, 14], [0, 0, 10, 10, 1.1, 14], [0, 0, 10, 10, .9, 99], [float('nan'), 0, 10, 10, .9, 14]]
        detector.model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=data))]
        for region in ([0, 0, 3840, 2160], [960, 540, 2880, 1620], [1440, 810, 2400, 1350]):
            request = SimpleNamespace(view=SimpleNamespace(source_region_xyxy=region),
                                      original_width=3840, original_height=2160)
            predictions = detector.detect(np.zeros((540, 960, 3), dtype=np.uint8), request)
            self.assertEqual(len(predictions), 1)
            prediction = predictions[0]
            x, y, right, bottom = region
            expected = [(x + 0.1 * (right - x)) / 3840, (y + 0.1 * (bottom - y)) / 2160,
                        (x + 0.5 * (right - x)) / 3840, (y + 0.5 * (bottom - y)) / 2160]
            np.testing.assert_allclose(prediction.bbox, expected)
            self.assertEqual(prediction.object_id, 'jammer')
        kwargs = detector.model.predict.call_args.kwargs
        self.assertTrue(kwargs['agnostic_nms'])
        self.assertEqual(kwargs['max_det'], 500)
        self.assertEqual(kwargs['imgsz'], 960)

    def test_protocol_on_detector_failure(self):
        request = SimpleNamespace(camera_command_feedback=None, view=None,
                                  request_id='same-id', frame=7)
        with patch.object(example, 'decode_view', return_value=np.zeros((540, 960, 3))), \
                patch.object(example, 'detect', side_effect=RuntimeError('test')), \
                patch.object(example, 'choose_next_view', return_value=None), \
                self.assertLogs(example.logger, level='ERROR'):
            response = example.predict(request)
        self.assertEqual(response.request_id, 'same-id')
        self.assertEqual(response.frame, 7)
        self.assertEqual(response.annotations, [])

    def test_boolean_parser(self):
        from detector import parse_bool
        for value in ('true', '1', 'yes', ' TRUE '):
            self.assertTrue(parse_bool(value))
        for value in ('false', '0', 'no'):
            self.assertFalse(parse_bool(value))
        for value in ('maybe', '', '2'):
            with self.assertRaises(ValueError):
                parse_bool(value)

    def test_camera_policies(self):
        from dtos import CameraConstraintsDto
        from local_evaluator import Camera
        import os
        for level, x, y in [(0, 1920, 1080), (1, 960, 540), (2, 480, 270)]:
            camera = Camera(level, x, y)
            request = SimpleNamespace(view=SimpleNamespace(resolution_level=level, center_x=x, center_y=y),
                camera_constraints=CameraConstraintsDto(**camera.constraints()), sequence_id='test')
            with patch.dict(os.environ, {'CAMERA_POLICY': 'hold_full'}):
                command = example.choose_next_view(request)
            if level == 0:
                self.assertIsNone(command)
            else:
                self.assertEqual(command.resolution_level, 0 if level == 1 else 1)
                camera.apply(command.resolution_level, command.center_x, command.center_y)
                if level == 1:
                    self.assertEqual((command.center_x, command.center_y), (1920, 1080))
            with patch.dict(os.environ, {'CAMERA_POLICY': 'baseline_sweep'}):
                self.assertIsNotNone(example.choose_next_view(request))
            with patch.dict(os.environ, {'CAMERA_POLICY': 'nonsense'}):
                with self.assertRaises(ValueError):
                    example.choose_next_view(request)

    def test_crop_geometry_and_negative_label(self):
        from prepare_multires_yolo import crop_labels, make_region, write_sample
        import cv2
        objects = [{'object_id': 'jammer', 'bbox': [900, 500, 1100, 700]}]
        labels = crop_labels(objects, (1000, 600, 1960, 1140))
        np.testing.assert_allclose(labels[0], [14, 50/960, 50/540, 100/960, 100/540])
        self.assertEqual(crop_labels(objects, (2000, 1000, 2960, 1540)), [])
        for level, size in [(1, (1920, 1080)), (2, (960, 540))]:
            region = make_region(level, 0, 0)
            self.assertEqual((region[2]-region[0], region[3]-region[1]), size)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'images/train').mkdir(parents=True)
            (root/'labels/train').mkdir(parents=True)
            write_sample(np.zeros((2160, 3840, 3), dtype=np.uint8), objects,
                         (2000, 1000, 2960, 1540), root, 'train', 'negative')
            self.assertEqual((root/'labels/train/negative.txt').read_text(), '')
            self.assertEqual(cv2.imread(str(root/'images/train/negative.png')).shape, (540, 960, 3))
        self.assertEqual(crop_labels(objects, (0, 0, 3840, 2160), scout=True)[0][0], 0)

    def test_multires_split_determinism(self):
        from prepare_multires_yolo import generate
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source/images').mkdir(parents=True)
            (root/'source/annotations').mkdir()
            for i in range(3):
                cv2.imwrite(str(root/f'source/images/frame_{i:06d}.png'), np.zeros((2160, 3840, 3), dtype=np.uint8))
                (root/f'source/annotations/frame_{i:06d}.json').write_text(json.dumps({'annotations':
                    [{'object_id': 'jammer', 'bbox': [1000, 500, 1050, 550]}]}))
            a = generate(root/'source', root/'a', background_crops=1)
            b = generate(root/'source', root/'b', background_crops=1)
            self.assertEqual(a, b)
            training = {sample['frame'] for sample in a['samples'] if sample['split'] == 'train'}
            validation = {sample['frame'] for sample in a['samples'] if sample['split'] == 'val'}
            self.assertFalse(training & validation)
            self.assertEqual(validation, {'frame_000002'})
            config = yaml.safe_load((root/'a/data.yaml').read_text())
            self.assertEqual(tuple(config['names'].values()), OBJECT_CLASSES)
            self.assertGreater(a['counts']['background'], 0)

    def test_detector_configuration_and_class_guard(self):
        import os
        from detector import get_detector
        with patch.dict(os.environ, {'YOLO_AGNOSTIC_NMS': 'no', 'YOLO_MAX_DET': '17',
                'YOLO_LOG_EVERY': '10', 'YOLO_CONF': '0.1', 'YOLO_IOU': '0.4'}), \
                patch('detector.YoloDetector') as factory:
            get_detector.cache_clear()
            get_detector()
            self.assertFalse(factory.call_args.kwargs['agnostic_nms'])
            self.assertEqual(factory.call_args.kwargs['max_det'], 17)
            get_detector.cache_clear()
        for limit in (0, 501, -1):
            with self.assertRaises(ValueError):
                YoloDetector('unused.pt', max_det=limit)
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)/'best.pt'
            weights.touch()
            fake = SimpleNamespace(YOLO=Mock(return_value=SimpleNamespace(names={0: 'person'})))
            with patch.dict('sys.modules', {'ultralytics': fake}):
                with self.assertRaisesRegex(ValueError, 'classes must exactly match'):
                    YoloDetector(weights)

    def test_score_parser(self):
        from benchmark_detector import parse_score
        self.assertEqual(parse_score('AP@0.50 by class\nCOCO mAP@0.50: 0.123\n'), 0.123)
        with self.assertRaises(ValueError):
            parse_score('error')


if __name__ == '__main__':
    unittest.main()
