"""Geometry and synthetic-file tests only; no real captures or inference."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from build_l1_reconstructions import Capture, build, load_captures, reconstruct, unique_regions, validate_region
from project_reconstruction_labels import project_box, project_labels, reconstruction_to_source, yolo_line
from dtos import OBJECT_CLASSES


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def capture(self, phase=0, sequence='main', frame=1, region=(0, 0, 1920, 1080), value=50,
                suffix='', level=1):
        directory = self.root / f'l1_phase_{phase:02d}' / sequence
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'frame_{frame:06d}{suffix}.png'
        image = np.full((540, 960, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = dict(sequence_id=sequence, frame=frame, frame_index=frame, resolution_level=level,
                        source_region_xyxy=list(region), center_x=(region[0]+region[2])//2,
                        center_y=(region[1]+region[3])//2, image_sha256=sha,
                        original_width=3840, original_height=2160)
        path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        return Capture(path, frame, sequence, region, sha)

    def test_placement_dimensions_black_uncovered_and_mask(self):
        capture = self.capture(region=(1920, 1080, 3840, 2160), value=81)
        image, mask, _ = reconstruct([capture])
        self.assertEqual(image.shape, (1080, 1920, 3))
        self.assertEqual(mask.shape, (1080, 1920))
        self.assertTrue(np.all(image[540:, 960:] == 81))
        self.assertTrue(np.all(mask[540:, 960:] == 255))
        self.assertTrue(np.all(image[:540] == 0))
        self.assertTrue(np.all(image[:, :960] == 0))
        self.assertEqual(np.count_nonzero(mask)/mask.size, 0.25)

    def test_identical_region_dedup_and_deterministic_overlap_mean(self):
        first = self.capture(value=20)
        duplicate = self.capture(phase=1, value=240)
        overlapping = self.capture(phase=2, region=(960, 0, 2880, 1080), value=80)
        self.assertEqual(unique_regions([duplicate, first]), [first])
        image, mask, chosen = reconstruct([overlapping, duplicate, first])
        self.assertEqual(len(chosen), 2)
        self.assertTrue(np.all(image[:540, :480] == 20))
        self.assertTrue(np.all(image[:540, 480:960] == 50))
        self.assertTrue(np.all(image[:540, 960:1440] == 80))
        self.assertEqual(np.count_nonzero(mask)/mask.size, 0.375)
        reverse, _, _ = reconstruct([first, duplicate, overlapping])
        np.testing.assert_array_equal(image, reverse)

    def test_largest_sequence_uses_distinct_frames_and_ignores_l0(self):
        self.capture(frame=1)
        self.capture(frame=2)
        for index in range(4):
            self.capture(sequence='Verify', frame=0, suffix=f'_{index}')
        self.capture(frame=3, level=0)
        captures, selections = load_captures(self.root)
        self.assertEqual(selections, {'l1_phase_00': 'main'})
        self.assertEqual([c.frame for c in captures], [1, 2])

    def test_manifest_metadata_and_source_files_preserved(self):
        first = self.capture(frame=100)
        second = self.capture(phase=1, frame=100, region=(1920, 1080, 3840, 2160))
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(build(self.root), 1)
        output = self.root / 'reconstructed_l1'
        metadata = json.loads((output / 'metadata/frame_000100.json').read_text())
        self.assertEqual(metadata['unique_regions'], 2)
        self.assertEqual(metadata['coverage_percent'], 50.0)
        self.assertEqual(metadata['original_size'], [3840, 2160])
        self.assertEqual(metadata['reconstruction_size'], [1920, 1080])
        self.assertEqual(metadata['contributing_capture_paths'], [str(first.path), str(second.path)])
        with (output / 'manifest.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['frame'], '100')
        self.assertTrue((output / rows[0]['image_path']).is_file())
        self.assertTrue((output / rows[0]['mask_path']).is_file())
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            build(self.root)
        with self.assertRaises(ValueError):
            build(self.root, self.root / 'l1_phase_00' / 'unsafe_output')

    def test_integrity_and_invalid_geometry_fail_explicitly(self):
        capture = self.capture()
        capture.path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            reconstruct([capture])
        for region in ((1, 0, 1921, 1080), (0, 0, 960, 540), (-2, 0, 1918, 1080)):
            with self.assertRaises(ValueError):
                validate_region(region)

    def test_scaling_and_crop_projection(self):
        box = (500, 300, 600, 400)
        self.assertEqual(reconstruction_to_source(box), (1000, 600, 1200, 800))
        self.assertEqual(project_box(box, (960, 540, 2880, 1620)), (20, 30, 120, 130))
        values = yolo_line('ta-ta', (20, 30, 120, 130)).split()
        self.assertEqual(int(values[0]), OBJECT_CLASSES.index('ta-ta'))
        np.testing.assert_allclose([float(v) for v in values[1:]],
                                   [70/960, 80/540, 100/960, 100/540], atol=1e-8)

    def test_clipping_and_visible_fraction_inclusive_boundary(self):
        region = (960, 540, 2880, 1620)
        box = (450, 300, 550, 400)  # 70% remains to the right of crop's x=480 edge.
        self.assertEqual(project_box(box, region), (0, 30, 70, 130))
        self.assertIsNone(project_box(box, region, 0.70001))
        self.assertIsNone(project_box((449, 300, 549, 400), region))
        self.assertEqual(project_box((1400, 780, 1500, 880), region, 0.1), (920, 510, 960, 540))
        self.assertIsNone(project_box((0, 0, 100, 100), region, 0))
        self.assertIsNone(project_box((380, 300, 480, 400), region, 0))
        for threshold in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                project_box(box, region, threshold)

    def test_projection_emits_every_capture_including_duplicates(self):
        first = self.capture()
        duplicate = self.capture(phase=1)
        empty = self.capture(phase=2, region=(1920, 1080, 3840, 2160))
        annotations = self.root / 'annotations.json'
        annotations.write_text(json.dumps({'frames': [{'frame': 1, 'annotations': [
            {'object_id': 'ta-ta', 'bbox': [10, 20, 110, 120]}]}]}))
        self.assertEqual(project_labels(self.root, annotations), 3)
        output = self.root / 'projected_l1_labels'
        first_text = (output / first.path.relative_to(self.root).with_suffix('.txt')).read_text()
        self.assertEqual(first_text.strip(), yolo_line('ta-ta', (10, 20, 110, 120)))
        self.assertEqual((output / duplicate.path.relative_to(self.root).with_suffix('.txt')).read_text(), first_text)
        self.assertEqual((output / empty.path.relative_to(self.root).with_suffix('.txt')).read_text(), '')


if __name__ == '__main__':
    unittest.main()
