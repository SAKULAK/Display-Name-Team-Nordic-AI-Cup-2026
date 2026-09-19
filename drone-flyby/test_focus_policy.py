"""Synthetic camera/memory unit tests: no model, server, evaluator, or replay."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import example
from dtos import (ALLOWED_RESOLUTION_LEVELS, CameraConstraintsDto,
                  DroneFlybyPredictionDto, MAXIMUM_CENTER_DELTA_PIXELS,
                  SOURCE_REGION_SIZES)
from focus_policy import (FOCUS_CLASSES, FocusConfig, FocusPolicy, SequenceState,
                          Track, legal_command, match_detections)


def detection(name='mine_roller', box=(0.1, 0.1, 0.12, 0.12), confidence=0.8):
    return DroneFlybyPredictionDto(object_id=name, bbox=list(box), confidence=confidence)


def request(frame=0, level=0, sequence='one', cx=None, cy=None):
    cx, cy = (1920 if cx is None else cx), (1080 if cy is None else cy)
    width, height = SOURCE_REGION_SIZES[level]
    bounds = []
    for allowed in ALLOWED_RESOLUTION_LEVELS[level]:
        w, h = SOURCE_REGION_SIZES[allowed]
        bounds.append(dict(resolution_level=allowed, width=960, height=540,
            minimum_center_x=w//2, maximum_center_x=3840-w//2,
            minimum_center_y=h//2, maximum_center_y=2160-h//2))
    constraints = CameraConstraintsDto(maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[level],
        allowed_resolution_levels=list(ALLOWED_RESOLUTION_LEVELS[level]),
        center_bounds=bounds, full_view_reset_exempt_from_delta=True)
    return SimpleNamespace(sequence_id=sequence, frame=frame, frame_index=frame,
        request_id=f'{sequence}:{frame}:{level}', original_width=3840, original_height=2160,
        camera_constraints=constraints, camera_command_feedback=None,
        view=SimpleNamespace(resolution_level=level, center_x=cx, center_y=cy,
            source_region_xyxy=[cx-width//2, cy-height//2, cx+width//2, cy+height//2]))


class FocusTests(unittest.TestCase):
    def setUp(self):
        self.policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0))

    def test_default_spacing_at_start_and_after_zoom(self):
        policy = FocusPolicy(FocusConfig(max_focus_conf=1.0))
        live = [detection(), detection('small_launcher', (0.8, 0.8, 0.82, 0.82))]
        self.assertIsNone(policy.process(request(0), live).requested_view)
        self.assertIsNone(policy.process(request(1), live).requested_view)
        result = policy.process(request(2), live)
        self.assertEqual(result.requested_view.resolution_level, 1)
        self.assertEqual(policy.process(request(3, 1), []).requested_view.resolution_level, 0)
        self.assertIsNone(policy.process(request(4), live).requested_view)
        self.assertIsNone(policy.process(request(5), live).requested_view)
        self.assertEqual(policy.process(request(6), live).requested_view.resolution_level, 1)

    def test_gaps_retries_and_failed_l0_do_not_count_as_successes(self):
        policy = FocusPolicy(FocusConfig(max_focus_conf=1.0))
        with patch.dict(os.environ, {'CAMERA_POLICY': 'focus_l1'}), \
                patch.object(example, 'focus_policy', policy), \
                patch.object(example, 'decode_view', return_value=None), \
                patch.object(example, 'detect', return_value=[detection()]):
            self.assertIsNone(example.predict(request(0)).requested_view)
            self.assertIsNone(example.predict(request(0)).requested_view)
            with patch.object(example, 'detect', side_effect=RuntimeError('fixture')), \
                    self.assertLogs(example.logger, level='ERROR'):
                example.predict(request(1))
            self.assertIsNone(example.predict(request(10)).requested_view)
            self.assertEqual(example.predict(request(11)).requested_view.resolution_level, 1)
            with patch.object(example, 'detect', side_effect=RuntimeError('fixture')), \
                    self.assertLogs(example.logger, level='ERROR'):
                example.predict(request(12, 1))
            self.assertEqual(policy.states['one'].l0_frames_since_focus, 0)
            self.assertIsNone(example.predict(request(13)).requested_view)

    def test_easy_targets_do_not_trigger_focus_and_skip_is_logged(self):
        policy = FocusPolicy(FocusConfig(max_focus_conf=1.0))
        for frame in (0, 1):
            policy.process(request(frame), [detection('tank', confidence=0.99)])
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = policy.process(request(2), [detection('tank', confidence=0.99)])
        self.assertIsNone(result.requested_view)
        self.assertIn('skip=class_filter', '\n'.join(logs.output))

    def test_exact_source_frame_velocity_2_3_4(self):
        policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0))
        old = [detection(name, (x, 0.8, x+0.02, 0.82)) for name, x in
               [('mine_roller', 0.6), ('small_launcher', 0.7), ('small_plane', 0.8)]]
        policy.process(request(2), old)
        policy.process(request(3, 1, cx=960, cy=540), [])
        new = [detection(d.object_id, (d.bbox[0]+20/3840, d.bbox[1]-20/2160,
                                     d.bbox[2]+20/3840, d.bbox[3]-20/2160)) for d in old]
        policy.process(request(4), new)
        state = policy.states['one']
        self.assertEqual(state.motion_frame_gap, 2)
        self.assertAlmostEqual(state.dx*3840, 10.0)
        self.assertAlmostEqual(state.dy*2160, -10.0)
        for frame in (5,):
            result = policy.process(request(frame, 1, cx=960, cy=540), [])
            self.assertEqual(len(result.annotations), 3)
            for d in result.annotations:
                direct = next(item for item in new if item.object_id == d.object_id)
                self.assertAlmostEqual((d.bbox[0]-direct.bbox[0])*3840, 10*(frame-4))
                self.assertAlmostEqual((d.bbox[1]-direct.bbox[1])*2160, -10*(frame-4))

    def test_l0_replaces_boxes_and_confidence_without_blending(self):
        self.policy.process(request(0), [detection(confidence=0.99)])
        self.policy.process(request(1, 1, cx=960, cy=540), [detection(confidence=0.98)])
        fresh = detection(box=(0.101, 0.101, 0.122, 0.122), confidence=0.61)
        result = self.policy.process(request(2), [fresh])
        self.assertEqual(result.annotations, [fresh])
        self.assertIs(self.policy.states['one'].tracks[0].detection, fresh)
        self.assertEqual(self.policy.states['one'].tracks[0].observed_frame, 2)

    def test_focus_outcome_class_change_and_memory_diagnostics(self):
        self.policy.process(request(0), [detection(confidence=0.41)])
        with self.assertLogs('focus_policy', level='INFO') as logs:
            self.policy.process(request(1, 1, cx=960, cy=540), [detection('tank', confidence=0.78)])
        text = '\n'.join(logs.output)
        for field in ('FOCUS_RESULT frame=1', 'target_track=0', 'L0_class=mine_roller',
                      'L0_conf=0.4100', 'L0_area_px2=', 'L1_class=tank', 'L1_conf=0.7800',
                      'matched=true', 'class_changed=true', 'confidence_increased=true',
                      'memory_count=0', 'memory_expired=', 'memory_replaced=',
                      'dx_per_frame=', 'dy_per_frame=', 'frame_gap=0'):
            self.assertIn(field, text)

    def test_unmatched_outcome_does_not_associate_distant_box(self):
        self.policy.process(request(0), [detection()])
        with self.assertLogs('focus_policy', level='INFO') as logs:
            self.policy.process(request(1, 1), [detection(box=(0.7, 0.7, 0.72, 0.72))])
        text = '\n'.join(logs.output)
        self.assertIn('matched=false', text)
        self.assertIn('L1_class=none', text)
        self.assertIn('confidence_increased=false', text)

    def test_environment_configuration_validation_and_baseline_isolation(self):
        with patch.dict(os.environ, {'FOCUS_MIN_L0_FRAMES': '5', 'FOCUS_MIN_SCORE': '2.5',
                'FOCUS_MEMORY_TTL': '4', 'FOCUS_CONFIDENCE_DECAY': '0.9',
                'FOCUS_CONFIDENCE_GRACE_FRAMES': '2'}):
            config = FocusPolicy().config
            self.assertEqual((config.min_l0_frames, config.minimum_focus_score), (5, 2.5))
            self.assertEqual((config.memory_ttl, config.confidence_grace_frames), (4, 2))
            self.assertEqual(config.confidence_decay, 0.9)
        for name, value in [('FOCUS_MIN_L0_FRAMES', '0'), ('FOCUS_MIN_L0_FRAMES', 'bad'),
                            ('FOCUS_MIN_SCORE', 'nan'), ('FOCUS_CONFIDENCE_DECAY', '1.1')]:
            with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                FocusPolicy().config
        with patch.dict(os.environ, {'CAMERA_POLICY': 'hold_full', 'FOCUS_MIN_L0_FRAMES': 'bad'}), \
                patch.object(example, 'focus_policy', FocusPolicy()), \
                patch.object(example, 'decode_view', return_value=None), \
                patch.object(example, 'detect', return_value=[detection()]):
            self.assertIsNone(example.predict(request()).requested_view)

    def test_focus_and_legal_center_clamping(self):
        for box, expected in [((0.0, 0.0, 0.01, 0.01), (960, 540)),
                              ((0.98, 0.98, 1.0, 1.0), (2880, 1620))]:
            policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0))
            result = policy.process(request(), [detection(box=box)])
            command = result.requested_view
            self.assertEqual(command.resolution_level, 1)
            self.assertEqual((command.center_x, command.center_y), expected)
        limited = request()
        limited.camera_constraints.maximum_center_delta = 1.0
        self.assertIsNone(FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0)).process(limited, [detection()]).requested_view)

    def test_one_focus_frame_returns_full(self):
        first = self.policy.process(request(), [detection()])
        command = first.requested_view
        zoom = request(1, 1, cx=command.center_x, cy=command.center_y)
        result = self.policy.process(zoom, [detection(confidence=0.9)])
        self.assertEqual(result.requested_view.model_dump(),
                         dict(resolution_level=0, center_x=1920, center_y=1080))
        self.assertEqual(self.policy.states['one'].focus_operations, 1)
        # An identical retried request must not advance memory/cooldown/counters.
        self.assertIs(self.policy.process(zoom, []), result)
        self.assertEqual(self.policy.states['one'].focus_operations, 1)

    def test_no_level_two_commands(self):
        self.assertIsNone(legal_command(request(1, 1), 2, 1920, 1080))
        for level in (0, 1, 2):
            result = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0)).process(request(level=level), [detection()])
            self.assertTrue(result.requested_view is None or result.requested_view.resolution_level in (0, 1))

    def test_sequence_reset_discards_all_state(self):
        self.policy.process(request(), [detection()])
        self.policy.process(request(1, 1), [detection()])
        state = self.policy.states['one']
        state.dx, state.dy = 0.1, -0.1
        result = self.policy.process(request(2, 1, sequence='two'), [])
        self.assertEqual(set(self.policy.states), {'two'})
        self.assertEqual(result.annotations, [])
        fresh = self.policy.states['two']
        self.assertEqual((fresh.dx, fresh.dy), (0.0, 0.0))
        self.assertEqual(fresh.cooldowns, [])
        self.assertEqual(fresh.last_full_detections, [])
        self.assertIsNone(fresh.last_focused_class)

    def test_robust_motion_and_gap_normalization(self):
        old = [detection(name, (x, 0.3, x+0.015, 0.32)) for name, x in
               [('tank', 0.2), ('helicopter', 0.4), ('jammer', 0.6), ('condor', 0.8)]]
        self.policy.process(request(), old)
        new = []
        for index, d in enumerate(old):
            dx, dy = (-0.02, 0.03) if index < 3 else (0.03, -0.01)
            new.append(detection(d.object_id, (d.bbox[0]+dx, d.bbox[1]+dy,
                                              d.bbox[2]+dx, d.bbox[3]+dy)))
        self.policy.process(request(2), new)
        state = self.policy.states['one']
        self.assertAlmostEqual(state.dx, -0.01)
        self.assertAlmostEqual(state.dy, 0.015)
        self.assertEqual(state.previous_full_detections, old)
        self.assertEqual(state.last_full_detections, new)
        self.policy.process(request(4), new[:1])
        self.assertEqual((state.dx, state.dy), (0.0, 0.0))

    def test_match_is_same_class_one_to_one_and_deterministic(self):
        old = [detection(box=(0.1, 0.1, 0.2, 0.2)), detection(box=(0.3, 0.1, 0.4, 0.2))]
        new = [old[1], old[0], detection('tank', old[0].bbox)]
        self.assertEqual(match_detections(old, new, 0.08), [(0, 1), (1, 0)])

    def test_tracker_projection_ttl_is_identity_only(self):
        self.policy.process(request(), [detection(box=(0.8, 0.8, 0.85, 0.85))])
        state = self.policy.states['one']
        track = state.tracks[0]
        state.dx, state.dy = -0.01, 0.005
        for frame in (0, 1, 2):
            projected = self.policy._project(track, state, frame)
            self.assertAlmostEqual(projected.confidence, 0.8*0.95**max(0, frame-1))
            self.assertAlmostEqual(projected.bbox[0], 0.8-0.01*frame)
        self.assertIsNone(self.policy._project(track, state, 3))

    def test_propagation_clips_and_drops_outside_frame(self):
        self.policy.process(request(), [detection(box=(0.96, 0.8, 0.99, 0.85))])
        self.policy.states['one'].dx = 0.02
        result = self.policy.process(request(1, 1, cx=960, cy=540), [])
        self.assertEqual(result.annotations[0].bbox[2], 1.0)
        result = self.policy.process(request(3, 1, cx=960, cy=540), [])
        self.assertEqual(result.annotations, [])

    def test_live_replaces_memory_and_suppresses_duplicates(self):
        self.policy.process(request(), [detection(confidence=0.99)])
        live = detection(confidence=0.7)
        duplicate = detection(box=(0.101, 0.101, 0.121, 0.121), confidence=0.6)
        result = self.policy.process(request(1, 1, cx=960, cy=540), [live, duplicate])
        self.assertEqual(result.annotations, [live])
        self.assertEqual(result.memory_emitted, 0)
        self.assertEqual(result.replaced, 1)

    def test_inside_snapshot_kept_when_unmatched_and_target_can_change_class(self):
        original = detection()
        for live in ([], [detection('tank')]):
            policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0))
            policy.process(request(), [original])
            result = policy.process(request(1, 1, cx=960, cy=540), live)
            self.assertEqual(result.annotations, live if live else [original])

    def test_l0_refresh_is_authoritative(self):
        self.policy.process(request(), [detection()])
        result = self.policy.process(request(1), [])
        self.assertEqual(result.annotations, [])
        self.assertIsNone(result.requested_view)
        self.assertEqual(self.policy.states['one'].tracks, [])

    def test_cooldown_selects_next_target(self):
        weak = detection()
        secondary = detection('ta-ta', (0.8, 0.8, 0.82, 0.82))
        self.policy.process(request(), [weak, secondary])
        self.assertEqual(self.policy.states['one'].focus_target.detection.object_id, 'mine_roller')
        self.policy.process(request(1, 1, cx=960, cy=540), [weak])
        self.policy.process(request(2), [weak, secondary])
        self.assertEqual(self.policy.states['one'].focus_target.detection.object_id, 'ta-ta')

    def test_cooldown_lasts_four_full_refreshes(self):
        self.policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0, confirmed_frames=1))
        self.policy.process(request(), [detection()])
        self.policy.process(request(1, 1, cx=960, cy=540), [detection()])
        for frame in (2, 3, 4, 5):
            result = self.policy.process(request(frame), [detection()])
            self.assertIsNone(result.requested_view)
        self.assertIsNotNone(self.policy.process(request(6), [detection()]).requested_view)

    def test_no_focus_request_means_no_snapshot_or_tracker_fallback(self):
        policy = FocusPolicy(FocusConfig(confidence_decay=0.9, minimum_focus_score=100))
        result = policy.process(request(), [detection(box=(0.8, 0.8, 0.85, 0.85))])
        self.assertIsNone(result.requested_view)
        self.assertIsNone(policy.states['one'].focus_snapshot)
        result = policy.process(request(1, 1, cx=960, cy=540), [])
        self.assertEqual(result.annotations, [])

    def test_focus_score_favors_uncertainty_smallness_and_staleness(self):
        uncertain = detection(confidence=0.2)
        certain = detection(confidence=0.9)
        large = detection(box=(0.1, 0.1, 0.4, 0.4), confidence=0.2)
        self.assertGreater(self.policy.focus_score(uncertain, None), self.policy.focus_score(certain, None))
        self.assertGreater(self.policy.focus_score(uncertain, None), self.policy.focus_score(large, None))
        self.assertGreater(self.policy.focus_score(uncertain, 12), self.policy.focus_score(uncertain, 5))

    def test_cooldown_region_survives_temporary_missing_track(self):
        self.policy.process(request(), [detection()])
        self.policy.process(request(1, 1, cx=960, cy=540), [detection()])
        self.policy.process(request(2), [])
        result = self.policy.process(request(3), [detection()])
        self.assertIsNone(result.requested_view)

    def test_snapshot_survives_source_gap_without_ttl(self):
        live = [detection(box=(i/1000, 0.1, i/1000+0.0005, 0.101)) for i in range(501)]
        result = self.policy.process(request(), live)
        self.assertEqual(len(result.annotations), 500)
        result = self.policy.process(request(10, 1, cx=960, cy=540), [])
        self.assertEqual(len(result.annotations), 500)
        self.assertEqual(result.expired, 0)

    def test_outside_snapshot_not_suppressed_by_nearby_live_box(self):
        remembered = detection(box=(0.505, 0.1, 0.525, 0.12), confidence=0.99)
        live = detection(box=(0.49, 0.1, 0.5, 0.12), confidence=0.7)
        self.policy.process(request(), [remembered])
        result = self.policy.process(request(1, 1, cx=960, cy=540), [live])
        self.assertEqual(result.annotations, [live, remembered])
        self.assertEqual(result.replaced, 0)

    def test_return_obeys_supplied_constraints(self):
        r = request(1, 1, cx=960, cy=540)
        r.camera_constraints.full_view_reset_exempt_from_delta = False
        r.camera_constraints.maximum_center_delta = 1.0
        self.assertIsNone(self.policy.process(r, []).requested_view)

    def test_hold_full_default_and_annotations_unchanged(self):
        live = [detection()]
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(example, 'decode_view', return_value=None), \
                patch.object(example, 'detect', return_value=live), \
                patch.object(example.focus_policy, 'process', side_effect=AssertionError('must not run')):
            for level in (0, 1):
                result = example.predict(request(level=level))
                self.assertEqual(result.annotations, live)
                if level == 0:
                    self.assertIsNone(result.requested_view)
                else:
                    self.assertEqual(result.requested_view.resolution_level, 0)

    def test_predict_integration_and_failure_return(self):
        with patch.dict(os.environ, {'CAMERA_POLICY': 'focus_l1'}), \
                patch.object(example, 'focus_policy', self.policy), \
                patch.object(example, 'decode_view', return_value=None), \
                patch.object(example, 'detect', return_value=[detection()]):
            result = example.predict(request())
            self.assertEqual(result.requested_view.resolution_level, 1)
            self.assertEqual(result.request_id, 'one:0:0')
            self.assertEqual(result.frame, 0)
            with patch.object(example, 'detect', side_effect=RuntimeError('fixture')), \
                    self.assertLogs(example.logger, level='ERROR'):
                result = example.predict(request(1, 1))
            self.assertEqual(result.annotations, [])
            self.assertEqual(result.requested_view.resolution_level, 0)

    def test_new_sequence_resets_even_if_detection_fails(self):
        self.policy.process(request(), [detection()])
        with patch.dict(os.environ, {'CAMERA_POLICY': 'focus_l1'}), \
                patch.object(example, 'focus_policy', self.policy), \
                patch.object(example, 'decode_view', return_value=None), \
                patch.object(example, 'detect', side_effect=RuntimeError('fixture')), \
                self.assertLogs(example.logger, level='ERROR'):
            example.predict(request(sequence='two'))
        self.assertEqual(set(self.policy.states), {'two'})
        self.assertEqual(self.policy.states['two'].tracks, [])

    def test_snapshot_bridge_preserves_all_outside_objects_without_tracker(self):
        policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0, memory_ttl=1,
                                        confidence_decay=0.01, confidence_grace_frames=0))
        original = [detection(), detection('jammer', (0.7, 0.7, 0.72, 0.72), 0.91),
                    detection('small_plane', (0.8, 0.8, 0.83, 0.82), 0.88),
                    detection('tank', (0.9, 0.7, 0.93, 0.73), 0.96)]
        policy.process(request(), original)
        state = policy.states['one']
        state.tracks.clear()  # No track identities or tracker data remain.
        state.dx, state.dy = 10/3840, -5/2160
        target = detection(box=(0.1+10/3840, 0.1-5/2160, 0.12+10/3840, 0.12-5/2160), confidence=0.95)
        result = policy.process(request(1, 1, cx=960, cy=540), [target])
        self.assertEqual(len(result.annotations), 4)
        self.assertEqual(result.memory_emitted, 3)
        for direct in original[1:]:
            projected = next(d for d in result.annotations if d.object_id == direct.object_id)
            self.assertEqual(projected.confidence, direct.confidence)
            np.testing.assert_allclose(projected.bbox, [direct.bbox[0]+10/3840, direct.bbox[1]-5/2160,
                                                        direct.bbox[2]+10/3840, direct.bbox[3]-5/2160])

    def test_snapshot_is_deep_copy_and_translation_is_applied_once(self):
        original = detection(box=(0.8, 0.8, 0.82, 0.82))
        self.policy.process(request(), [original])
        state = self.policy.states['one']
        saved = state.focus_snapshot
        original.bbox[0] = 0.79
        original.confidence = 0.2
        state.dx = 0.01
        result = self.policy.process(request(1, 1, cx=960, cy=540), [])
        np.testing.assert_allclose(result.annotations[0].bbox, [0.81, 0.8, 0.83, 0.82])
        self.assertEqual(result.annotations[0].confidence, 0.8)
        self.assertEqual(saved.annotations[0].bbox, [0.8, 0.8, 0.82, 0.82])
        self.assertIs(self.policy.process(request(1, 1, cx=960, cy=540), []), result)
        with self.assertLogs('focus_policy', level='INFO') as logs:
            repeated_l1 = self.policy.process(request(2, 1, cx=960, cy=540), [])
        self.assertEqual(repeated_l1.annotations, [])
        self.assertIn('snapshot_status=consumed', '\n'.join(logs.output))

    def test_snapshot_discarded_on_next_full_view(self):
        self.policy.process(request(), [detection()])
        self.policy.process(request(1, 1, cx=960, cy=540), [detection(confidence=0.9)])
        self.assertIsNotNone(self.policy.states['one'].focus_snapshot)
        result = self.policy.process(request(2), [])
        self.assertEqual(result.annotations, [])
        self.assertIsNone(self.policy.states['one'].focus_snapshot)

    def test_unmatched_non_target_inside_crop_survives(self):
        old = [detection(), detection('jammer', (0.3, 0.3, 0.32, 0.32))]
        self.policy.process(request(), old)
        fresh_target = detection(confidence=0.95)
        unrelated = detection('helicopter', (0.3, 0.3, 0.32, 0.32), 0.9)
        result = self.policy.process(request(1, 1, cx=960, cy=540), [fresh_target, unrelated])
        self.assertEqual(result.annotations, [fresh_target, old[1]])
        self.assertEqual(result.replaced, 1)

    def test_one_live_box_cannot_delete_two_snapshot_objects(self):
        old = [detection(), detection('tank', (0.3, 0.3, 0.34, 0.34)),
               detection('tank', (0.31, 0.3, 0.35, 0.34))]
        self.policy.process(request(), old)
        fresh = detection('tank', (0.305, 0.3, 0.345, 0.34), 0.99)
        result = self.policy.process(request(1, 1, cx=960, cy=540), [fresh])
        self.assertEqual(result.replaced, 0)
        self.assertEqual(result.memory_emitted, 3)
        self.assertEqual(len(result.annotations), 3)
        self.assertEqual(result.annotations, old)

    def test_bridge_logs_explain_clipping_and_counts(self):
        old = [detection(), detection('jammer', (0.995, 0.7, 1.0, 0.72)),
               detection('tank', (0.8, 0.8, 0.82, 0.82))]
        self.policy.process(request(), old)
        self.policy.states['one'].dx = 0.01
        fresh = detection(box=(0.11, 0.1, 0.13, 0.12), confidence=0.95)
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = self.policy.process(request(1, 1, cx=960, cy=540), [fresh])
        text = '\n'.join(logs.output)
        for field in ('FOCUS_BRIDGE frame=1', 'snapshot_count=3', 'live_count=1', 'snapshot_kept=1',
                      'snapshot_replaced=1', 'snapshot_clipped_out=1', 'snapshot_overflow=0',
                      'final_count=2', 'FOCUS_RESULT'):
            self.assertIn(field, text)
        self.assertEqual(len(result.annotations), 2)

    def test_bridge_logs_protocol_cap_loss(self):
        old = [detection(box=(i/1000, 0.1, i/1000+0.0005, 0.101)) for i in range(500)]
        self.policy.process(request(), old)
        fresh = detection('tank', (0.3, 0.3, 0.32, 0.32), 0.99)
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = self.policy.process(request(1, 1, cx=960, cy=540), [fresh])
        self.assertEqual(len(result.annotations), 500)
        self.assertIs(result.annotations[0], fresh)
        self.assertIn('snapshot_overflow=1', '\n'.join(logs.output))

    def test_confirmed_trajectory_survives_track_id_change_and_expires(self):
        old = [detection(), detection('tank', (0.3, 0.3, 0.32, 0.32), 0.99),
               detection('jammer', (0.6, 0.3, 0.62, 0.32), 0.99)]
        self.policy.process(request(), old)
        old_id = self.policy.states['one'].focus_target.track_id
        fresh = detection(box=(0.1+10/3840, 0.1, 0.12+10/3840, 0.12), confidence=0.9)
        self.policy.process(request(1, 1, cx=960, cy=540), [fresh])
        state = self.policy.states['one']
        self.assertEqual(len(state.confirmations), 1)
        state.tracks.clear()
        state.cooldowns.clear()  # Prove confirmation does not rely on either ID/cooldown.
        def translated(frame):
            return [detection(d.object_id, (d.bbox[0]+10*frame/3840, d.bbox[1],
                                           d.bbox[2]+10*frame/3840, d.bbox[3]), d.confidence) for d in old]
        result = self.policy.process(request(4), translated(4))
        self.assertNotEqual(state.tracks[0].track_id, old_id)
        self.assertIsNone(result.requested_view)
        self.assertEqual(state.confirmations[0].confirmed_frame, 1)
        self.assertAlmostEqual(state.confirmations[0].bbox[0], 0.1+40/3840)
        self.assertIsNone(self.policy.process(request(13), translated(13)).requested_view)
        self.assertIsNotNone(self.policy.process(request(14), translated(14)).requested_view)

    def test_confirmation_requires_same_class_and_threshold(self):
        for fresh, expected in [(detection(confidence=0.70), 1), (detection(confidence=0.69), 0),
                                (detection('tank', confidence=0.99), 0)]:
            policy = FocusPolicy(FocusConfig(min_l0_frames=1, max_focus_conf=1.0))
            policy.process(request(), [detection()])
            policy.process(request(1, 1, cx=960, cy=540), [fresh])
            self.assertEqual(len(policy.states['one'].confirmations), expected)

    def test_only_six_difficult_classes_can_be_focus_targets(self):
        from dtos import OBJECT_CLASSES
        self.assertEqual(FOCUS_CLASSES, {'small_launcher', 'ta-ta', 'mine_roller',
                                        'hangar', 'medium_launcher', 'medium_plane'})
        for name in OBJECT_CLASSES:
            with self.subTest(name=name):
                policy = FocusPolicy(FocusConfig(min_l0_frames=1))
                result = policy.process(request(), [detection(name, confidence=0.15)])
                self.assertEqual(result.requested_view is not None, name in FOCUS_CLASSES)

    def test_max_focus_confidence_boundary_and_latest_examples(self):
        for name, confidence, eligible in [('ta-ta', 0.164, True), ('small_launcher', 0.351, True),
                ('spacecraft', 0.141, False), ('ta-ta', 0.677, False), ('small_launcher', 0.104, True),
                ('mine_roller', 0.45, True), ('mine_roller', 0.45001, False)]:
            with self.subTest(name=name, confidence=confidence):
                result = FocusPolicy(FocusConfig(min_l0_frames=1)).process(
                    request(), [detection(name, confidence=confidence)])
                self.assertEqual(result.requested_view is not None, eligible)
        with patch.dict(os.environ, {'FOCUS_MAX_CONF': '0.25'}):
            self.assertEqual(FocusPolicy().config.max_focus_conf, 0.25)
        for value in ('bad', 'nan', '-0.1', '1.1'):
            with patch.dict(os.environ, {'FOCUS_MAX_CONF': value}), self.assertRaises(ValueError):
                FocusPolicy().config

    def test_threshold_still_applies_after_class_and_confidence_filters(self):
        policy = FocusPolicy(FocusConfig(min_l0_frames=1, minimum_focus_score=100))
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = policy.process(request(), [detection('ta-ta', confidence=0.2)])
        self.assertIsNone(result.requested_view)
        self.assertIn('skip=no_target_above_threshold', '\n'.join(logs.output))

    def test_no_hardcoded_end_of_sequence(self):
        for frame in (24, 25, 1000):
            policy = FocusPolicy(FocusConfig(min_l0_frames=1))
            self.assertIsNotNone(policy.process(request(frame), [detection(confidence=0.2)]).requested_view)

    def velocity_fixture(self):
        policy = FocusPolicy(FocusConfig(min_l0_frames=2))
        old = [detection(confidence=0.3),
               detection('tank', (0.3, 0.7, 0.34, 0.74)),
               detection('jammer', (0.45, 0.7, 0.49, 0.74)),
               detection('jet_plane', (0.6, 0.7, 0.64, 0.74)),
               detection('small_plane', (0.8, 0.8, 0.84, 0.84))]
        current = [detection(d.object_id, (d.bbox[0]+0.002, d.bbox[1]+0.001,
                                           d.bbox[2]+0.002, d.bbox[3]+0.001), d.confidence) for d in old[:-1]]
        current += [detection('small_plane', (0.808, 0.794, 0.85, 0.838)),
                    detection('condor', (0.9, 0.85, 0.93, 0.88))]
        policy.process(request(2), old)
        result = policy.process(request(4), current)
        self.assertIsNotNone(result.requested_view)
        return policy, current

    def test_individual_edge_velocity_changes_size_and_differs_from_global(self):
        policy, current = self.velocity_fixture()
        state = policy.states['one']
        self.assertAlmostEqual(state.dx, 0.001)
        self.assertAlmostEqual(state.dy, 0.0005)
        np.testing.assert_allclose(state.focus_snapshot.edge_velocities[4], (0.004, -0.003, 0.005, -0.001))
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = policy.process(request(5, 1, cx=960, cy=540), [])
        predicted = next(d for d in result.annotations if d.object_id == 'small_plane')
        np.testing.assert_allclose(predicted.bbox, [0.812, 0.791, 0.855, 0.837])
        self.assertAlmostEqual(predicted.bbox[2]-predicted.bbox[0], 0.043)
        self.assertAlmostEqual(predicted.bbox[3]-predicted.bbox[1], 0.046)
        self.assertEqual(predicted.confidence, current[4].confidence)
        self.assertIn('individual_velocity=5 global_fallback=1', '\n'.join(logs.output))

    def test_unmatched_object_uses_global_fallback(self):
        policy, _ = self.velocity_fixture()
        self.assertIsNone(policy.states['one'].focus_snapshot.edge_velocities[5])
        result = policy.process(request(5, 1, cx=960, cy=540), [])
        box = next(d.bbox for d in result.annotations if d.object_id == 'condor')
        np.testing.assert_allclose(box, [0.901, 0.8505, 0.931, 0.8805])

    def test_individual_velocity_not_used_for_longer_source_gap(self):
        policy, current = self.velocity_fixture()
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = policy.process(request(6, 1, cx=960, cy=540), [])
        box = next(d.bbox for d in result.annotations if d.object_id == 'small_plane')
        np.testing.assert_allclose(box, [0.810, 0.795, 0.852, 0.839])
        self.assertIn('individual_velocity=0 global_fallback=6', '\n'.join(logs.output))

    def test_ambiguous_object_velocity_falls_back(self):
        state = SequenceState('one', last_full_frame=2,
            last_full_detections=[detection('tank', (0.3, 0.3, 0.34, 0.34)),
                                  detection('tank', (0.31, 0.3, 0.35, 0.34))])
        result = self.policy._individual_velocities(state,
            [detection('tank', (0.305, 0.3, 0.345, 0.34))], 4)
        self.assertEqual(result, (None,))

    def test_weak_incidental_match_keeps_saved_box_but_target_live_wins(self):
        policy = FocusPolicy(FocusConfig(min_l0_frames=1))
        saved = detection('jet_plane', (0.3, 0.3, 0.32, 0.32), 0.99)
        policy.process(request(), [detection(confidence=0.2), saved])
        target_live = detection(box=(0.108, 0.1, 0.128, 0.12), confidence=0.9)
        weak_incidental = detection('jet_plane', (0.308, 0.3, 0.328, 0.32), 0.95)
        new_object = detection('helicopter', (0.4, 0.4, 0.42, 0.42), 0.8)
        with self.assertLogs('focus_policy', level='INFO') as logs:
            result = policy.process(request(1, 1, cx=960, cy=540), [target_live, weak_incidental, new_object])
        self.assertEqual(result.annotations, [target_live, new_object, saved])
        text = '\n'.join(logs.output)
        for field in ('FOCUS_BRIDGE_CLASSES frame=1', 'propagated={"jet_plane":1}',
                      'live_replaced={"mine_roller":1}', 'live_added={"helicopter":1}', 'live_rejected=1'):
            self.assertIn(field, text)

    def test_strong_incidental_same_class_can_replace(self):
        saved = detection('large_tower', (0.3, 0.3, 0.34, 0.34), 0.9)
        self.policy.process(request(), [detection(), saved])
        fresh = detection('large_tower', (0.301, 0.3, 0.341, 0.34), 0.95)
        result = self.policy.process(request(1, 1, cx=960, cy=540), [fresh])
        self.assertEqual(result.replaced, 1)
        self.assertIs(result.annotations[0], fresh)


if __name__ == '__main__':
    unittest.main()
