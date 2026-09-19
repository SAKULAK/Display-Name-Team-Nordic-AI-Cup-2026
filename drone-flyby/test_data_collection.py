"""Synthetic collection/capture checks; no network or model inference."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import capture_data
from collection_policy import CollectionPolicy, L1_PATH, L2_PATH
import example
from dtos import CameraCommandFeedbackDto, RequestedViewDto
from test_focus_policy import request, detection


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'captures'
        env = patch.dict(os.environ, {'CAPTURE_ROOT': str(self.root)}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.r = request(22, 2, cx=1200, cy=700)
        # The capture layer does not decode pixels or alter even arbitrary PNG bytes.
        self.raw = b'\x89PNG\r\n\x1a\nexact original payload\x00\xff'
        self.r.view.image = base64.b64encode(self.raw).decode()
        self.r.view.width, self.r.view.height = 960, 540
        self.r.view.image_media_type = 'image/png'

    def test_disabled_creates_nothing_and_does_not_inspect_request(self):
        capture_data.capture_request(object())
        self.assertFalse(self.root.exists())

    def test_exact_bytes_and_metadata(self):
        os.environ['CAPTURE_ENABLED'] = 'true'
        self.r.camera_command_feedback = CameraCommandFeedbackDto(frame=21,
            requested_view=RequestedViewDto(resolution_level=2, center_x=1200, center_y=700), reason='fixture')
        capture_data.capture_request(self.r)
        image = self.root / 'run' / 'one' / 'frame_000022_L2_x1200_y700.png'
        self.assertEqual(image.read_bytes(), self.raw)
        metadata = json.loads(image.with_suffix('.json').read_text())
        for field in ('sequence_id', 'frame', 'frame_index', 'request_id', 'original_width', 'original_height'):
            self.assertEqual(metadata[field], getattr(self.r, field))
        for field in ('resolution_level', 'center_x', 'center_y', 'source_region_xyxy', 'image_media_type'):
            self.assertEqual(metadata[field], getattr(self.r.view, field))
        self.assertEqual((metadata['transmitted_width'], metadata['transmitted_height']), (960, 540))
        self.assertEqual(metadata['image_sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(metadata['camera_command_feedback'], self.r.camera_command_feedback.model_dump(mode='json'))
        self.assertFalse(list(self.root.rglob('*.tmp')))
        self.assertFalse(list(self.root.rglob('*.reserve')))

    def test_duplicates_and_concurrent_requests_do_not_overwrite(self):
        os.environ['CAPTURE_ENABLED'] = 'true'
        capture_data.capture_request(self.r)
        self.r.view.image = base64.b64encode(b'new bytes').decode()
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(capture_data.capture_request, [self.r]*3))
        images = list(self.root.rglob('*.png'))
        self.assertEqual(len(images), 4)
        self.assertEqual(len(list(self.root.rglob('*.json'))), 4)
        self.assertEqual(sum(p.read_bytes() == self.raw for p in images), 1)

    def test_unsafe_ids_stay_under_capture_root(self):
        os.environ.update(CAPTURE_ENABLED='true', CAPTURE_RUN_ID='../../CON:bad')
        self.r.sequence_id = '../..\\outside/evil'
        capture_data.capture_request(self.r)
        image, = self.root.rglob('*.png')
        self.assertTrue(image.resolve().is_relative_to(self.root.resolve()))
        self.assertEqual(len(image.relative_to(self.root).parts), 3)
        for value in ('CON', 'NUL', 'COM1', '..', '/', 'a:b', 'a?b'):
            safe = capture_data.safe_component(value)
            self.assertRegex(safe, r'^[A-Za-z0-9_-]+$')
            self.assertNotIn(safe.upper(), ('CON', 'NUL', 'COM1'))
        self.assertNotEqual(capture_data.safe_component('a:b'), capture_data.safe_component('a?b'))

    def test_capture_failure_does_not_break_normal_prediction(self):
        os.environ['CAPTURE_ENABLED'] = 'true'
        with patch('capture_data._atomic_write', side_effect=OSError('disk full')), \
                patch.object(example, 'decode_view', return_value=None) as decode, \
                patch.object(example, 'detect', return_value=[detection()]) as detect, \
                self.assertLogs('capture_data', level='ERROR'):
            response = example.predict(self.r)
        self.assertEqual(response.annotations, [detection()])
        decode.assert_called_once()
        detect.assert_called_once()

    def test_capture_only_skips_decode_detector_and_runs_camera(self):
        os.environ.update(CAPTURE_ENABLED='true', CAPTURE_ONLY='true', CAMERA_POLICY='data_collect', COLLECT_LEVEL='2')
        with patch.object(example, 'decode_view', side_effect=AssertionError('decode')), \
                patch.object(example, 'detect', side_effect=AssertionError('detect')), \
                patch.object(example, 'collection_policy', CollectionPolicy()), \
                patch.object(example, 'choose_next_view', wraps=example.choose_next_view) as choose:
            response = example.predict(self.r)
        choose.assert_called_once_with(self.r)
        self.assertEqual(response.annotations, [])
        self.assertEqual((response.request_id, response.frame), (self.r.request_id, self.r.frame))
        self.assertIsNotNone(response.requested_view)
        self.assertEqual(len(list(self.root.rglob('*.png'))), 1)

    def test_capture_only_does_not_implicitly_enable_capture(self):
        os.environ['CAPTURE_ONLY'] = 'true'
        with patch.object(example, 'decode_view', side_effect=AssertionError), \
                patch.object(example, 'detect', side_effect=AssertionError):
            self.assertEqual(example.predict(self.r).annotations, [])
        self.assertFalse(self.root.exists())

    def test_capture_only_focus_returns_full_without_inference(self):
        from focus_policy import FocusPolicy, FocusConfig
        os.environ.update(CAPTURE_ONLY='true', CAMERA_POLICY='focus_l1')
        self.r = request(22, 1, cx=1200, cy=700)
        with patch.object(example, 'focus_policy', FocusPolicy(FocusConfig())), \
                patch.object(example, 'decode_view', side_effect=AssertionError), \
                patch.object(example, 'detect', side_effect=AssertionError):
            response = example.predict(self.r)
        self.assertEqual(response.annotations, [])
        self.assertEqual(response.requested_view.resolution_level, 0)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def assert_legal(self, r, command):
        self.assertIsNotNone(command)
        self.assertIn(command.resolution_level, r.camera_constraints.allowed_resolution_levels)
        self.assertLessEqual(abs(command.resolution_level-r.view.resolution_level), 1)
        self.assertIs(type(command.center_x), int)
        self.assertIs(type(command.center_y), int)
        bounds = r.camera_constraints.bounds_for_level(command.resolution_level)
        self.assertTrue(bounds.minimum_center_x <= command.center_x <= bounds.maximum_center_x)
        self.assertTrue(bounds.minimum_center_y <= command.center_y <= bounds.maximum_center_y)
        if not (command.resolution_level == 0 and r.camera_constraints.full_view_reset_exempt_from_delta):
            self.assertLessEqual(math.dist((r.view.center_x, r.view.center_y),
                                          (command.center_x, command.center_y)), r.camera_constraints.maximum_center_delta)

    def test_level_zero_holds_and_returns_via_l1(self):
        policy = CollectionPolicy()
        self.assertIsNone(policy.choose(request()))
        for frame, level, x, y, expected in [(1, 2, 480, 270, 1), (2, 1, 960, 540, 0)]:
            r = request(frame, level, cx=x, cy=y)
            command = policy.choose(r)
            self.assert_legal(r, command)
            self.assertEqual(command.resolution_level, expected)

    def test_l1_exact_cycle_and_wrap(self):
        os.environ['COLLECT_LEVEL'] = '1'
        self.assertEqual(len(set(L1_PATH)), 9)
        self.assertEqual(L1_PATH, ((960, 540), (1920, 540), (2880, 540), (2880, 1080),
                         (2880, 1620), (1920, 1620), (960, 1620), (960, 1080), (1920, 1080)))
        policy = CollectionPolicy()
        for i, (x, y) in enumerate(L1_PATH):
            r = request(i, 1, cx=x, cy=y)
            command = policy.choose(r)
            self.assert_legal(r, command)
            self.assertEqual((command.center_x, command.center_y), L1_PATH[(i+1) % 9])

    def test_l2_exact_cycle_and_wrap(self):
        os.environ['COLLECT_LEVEL'] = '2'
        self.assertEqual(len(L2_PATH), 28)
        self.assertEqual(set(L2_PATH), {(x, y) for x in range(480, 3361, 480) for y in (270, 810, 1350, 1890)})
        policy = CollectionPolicy()
        for i, (x, y) in enumerate(L2_PATH):
            expected = L2_PATH[(i+1) % 28]
            self.assertLessEqual(math.dist((x, y), expected), 551)
            r = request(i, 2, cx=x, cy=y)
            command = policy.choose(r)
            self.assert_legal(r, command)
            self.assertEqual((command.center_x, command.center_y), expected)

    def test_all_phases_and_modulo_enter_correct_centers(self):
        for level, path in ((1, L1_PATH), (2, L2_PATH)):
            os.environ['COLLECT_LEVEL'] = str(level)
            for phase in range(-1, 2*len(path)):
                with self.subTest(level=level, phase=phase):
                    os.environ['COLLECT_PHASE'] = str(phase)
                    policy = CollectionPolicy()
                    r = request()
                    command = policy.choose(r)
                    self.assert_legal(r, command)
                    self.assertEqual(command.resolution_level, 1)
                    if level == 2:
                        r = request(1, 1, cx=command.center_x, cy=command.center_y)
                        command = policy.choose(r)
                        self.assert_legal(r, command)
                        self.assertEqual(command.resolution_level, 2)
                    self.assertEqual((command.center_x, command.center_y), path[phase % len(path)])

    def test_actual_constraints_can_force_hold(self):
        os.environ['COLLECT_LEVEL'] = '2'
        for constraint in ('level', 'delta', 'bounds', 'missing'):
            with self.subTest(constraint=constraint):
                r = request(1, 1)
                if constraint == 'level':
                    r.camera_constraints.allowed_resolution_levels = [0, 1]
                elif constraint == 'delta':
                    r.camera_constraints.maximum_center_delta = 0
                elif constraint == 'bounds':
                    r.camera_constraints.bounds_for_level(2).minimum_center_x = 2000
                else:
                    r.camera_constraints.center_bounds = []
                with self.assertLogs('collection_policy', level='INFO') as logs:
                    self.assertIsNone(CollectionPolicy().choose(r))
                self.assertIn('hold reason=', '\n'.join(logs.output))

    def test_noninitial_l1_uses_nearest_l2_and_l2_returns_to_l1(self):
        os.environ.update(COLLECT_LEVEL='2', COLLECT_PHASE='0')
        command = CollectionPolicy().choose(request(7, 1, cx=2880, cy=1620))
        self.assertEqual((command.center_x, command.center_y), (2880, 1350))
        os.environ['COLLECT_LEVEL'] = '1'
        r = request(8, 2, cx=3360, cy=1890)
        self.assert_legal(r, CollectionPolicy().choose(r))

    def test_retry_and_sequence_reset_preserve_initial_bridge(self):
        os.environ.update(COLLECT_LEVEL='2', COLLECT_PHASE='0')
        policy = CollectionPolicy()
        r = request()
        first = policy.choose(r)
        self.assertEqual(first, policy.choose(r))
        zoom = request(1, 1, cx=first.center_x, cy=first.center_y)
        self.assertEqual(policy.choose(zoom).center_x, 480)
        zoom.sequence_id = 'new'
        self.assertEqual(policy.choose(zoom).center_x, 960)

    def test_no_direct_l0_l2_even_if_supplied_levels_allow_it(self):
        for desired, current in ((2, 0), (0, 2)):
            os.environ['COLLECT_LEVEL'] = str(desired)
            r = request(level=current)
            r.camera_constraints.allowed_resolution_levels = [0, 1, 2]
            command = CollectionPolicy().choose(r)
            self.assertEqual(command.resolution_level, 1)


if __name__ == '__main__':
    unittest.main()
