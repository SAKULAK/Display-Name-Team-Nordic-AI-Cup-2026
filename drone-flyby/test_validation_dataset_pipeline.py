"""Offline synthetic tests for CVAT conversion, auditing, packaging, and splitting."""

from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import yaml

from audit_reconstruction_annotations import audit
from build_validation_yolo_dataset import build_dataset
from cvat_coco_to_reconstruction import convert_coco, convert_file, source_frame, xywh_to_xyxy
from dtos import OBJECT_CLASSES
from prepare_combined_finetune_dataset import prepare
from project_reconstruction_labels import load_annotations, project_labels
from validation_dataset_utils import validate_yolo, write_yaml


def png_bytes(value):
    """Small compressed, genuine 960x540 grayscale PNG, without an imaging library."""
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 960, 540, 8, 0, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress((b'\x00'+bytes([value])*960)*540)) + chunk(b'IEND', b''))


def coco_fixture():
    return dict(categories=[{'id': 99, 'name': 'ta-ta'}, {'id': 7, 'name': 'hangar'}],
                images=[dict(id=8, width=1920, height=1080, file_name='images/frame_000123.png'),
                        dict(id=9, width=1920, height=1080, file_name='frame_000002.png')],
                annotations=[dict(image_id=8, category_id=99, bbox=[1.25, 2.5, 30.75, 40.125]),
                             dict(image_id=8, category_id=7, bbox=[100, 100, 40, 50])])


class ConversionTests(unittest.TestCase):
    def test_xywh_floats_category_names_empty_frames_and_sort(self):
        result, summary = convert_coco(coco_fixture())
        self.assertEqual([f['frame'] for f in result['frames']], [2, 123])
        self.assertEqual(result['frames'][0]['annotations'], [])
        self.assertEqual(result['frames'][1]['annotations'], [
            dict(object_id='hangar', bbox=[100.0, 100.0, 140.0, 150.0]),
            dict(object_id='ta-ta', bbox=[1.25, 2.5, 32.0, 42.625])])
        self.assertEqual((summary['images'], summary['annotated_images'], summary['empty_images'], summary['boxes']),
                         (2, 1, 1, 2))
        self.assertEqual((summary['first_frame'], summary['last_frame']), (2, 123))
        self.assertEqual(list(summary['per_class_counts']), list(OBJECT_CLASSES))
        coco = coco_fixture()
        coco['annotations'].reverse()
        self.assertEqual(convert_coco(coco)[0], result)

    def test_source_frame_extraction_and_duplicate_detection(self):
        self.assertEqual(source_frame('prefix\\frame_000123.png'), 123)
        for name in ('frame_123.jpg', 'image.png', 'frame_-00001.png'):
            with self.assertRaises(ValueError):
                source_frame(name)
        coco = coco_fixture()
        coco['images'][1]['file_name'] = 'other/frame_000123.png'
        with self.assertRaisesRegex(ValueError, 'Duplicate source frame'):
            convert_coco(coco)

    def test_unknown_categories_and_bad_references(self):
        for mutate in (lambda c: c['categories'][0].update(name='unknown'),
                       lambda c: c['annotations'][0].update(category_id=100),
                       lambda c: c['annotations'][0].update(image_id=100),
                       lambda c: c['images'][0].update(width=960),
                       lambda c: c['images'][1].update(id=8),
                       lambda c: c['categories'][1].update(id=99)):
            coco = coco_fixture()
            mutate(coco)
            with self.assertRaises(ValueError):
                convert_coco(coco)

    def test_invalid_boxes_and_epsilon_only_clamping(self):
        for box in ([1, 2, 0, 5], [1, 2, -1, 5], [-0.01, 2, 5, 5], [1919, 0, 2, 10],
                    [1, 2, float('nan'), 4], [True, 2, 3, 4], [0, 0, 1, float('inf')]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                xywh_to_xyxy(box)
        self.assertEqual(xywh_to_xyxy([-1e-7, 0, 1920.0000002, 1080.0000001]),
                         [0.0, 0.0, 1920.0, 1080.0])

    def test_file_conversion_summary_completeness_warning_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, summary = root/'coco.json', root/'annotations.json', root/'summary.json'
            source.write_text(json.dumps(coco_fixture()))
            before = source.read_bytes()
            log = io.StringIO()
            with redirect_stdout(log):
                convert_file(source, output, summary)
            self.assertIn('Classes with zero human annotations:', log.getvalue())
            self.assertIn('unlabeled visible objects will be treated as background', log.getvalue())
            self.assertEqual(len(load_annotations(output)), 2)
            self.assertEqual(json.loads(summary.read_text())['boxes'], 2)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                convert_file(source, output)

    def test_audit_flags_without_mutation(self):
        frames = {1: [dict(object_id='hangar', bbox=[0, 0, 10, 10])]*2, 2: [],
                  3: [dict(object_id='hangar', bbox=[20, 20, 30, 30])]*3 +
                     [dict(object_id='hangar', bbox=[20, 20, 1020, 1020])]}
        before = copy.deepcopy(frames)
        report = audit(frames)
        self.assertEqual(report['total_reviewed_frames'], 3)
        self.assertEqual(report['total_boxes'], 6)
        self.assertEqual(report['empty_frames'], 1)
        self.assertEqual(report['per_class']['hangar']['unique_frames'], 2)
        self.assertEqual(report['per_class']['hangar']['median_width'], 10)
        self.assertEqual(report['per_class']['hangar']['max_area'], 1000000)
        self.assertEqual(report['boundary_box_count'], 2)
        self.assertEqual(len(report['suspicious_exact_duplicate_boxes']), 2)
        self.assertEqual(len(report['frames_with_multiple_instances']), 2)
        self.assertTrue(report['statistical_outliers'])
        self.assertEqual(frames, before)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.captures = self.root/'captures'
        self.labels = self.root/'projected_l1_labels'
        self.labels.mkdir()
        self.validation = self.root/'validation_yolo'
        self.helsinki = self.root/'helsinki'

    def capture(self, frame=123, phase=0, value=50, label='11 0.5 0.5 0.2 0.2\n',
                sequence='main', suffix='', level=1):
        relative = Path(f'l1_phase_{phase:02d}') / sequence / f'frame_{frame:06d}{suffix}.png'
        image = self.captures/relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(png_bytes(value))
        image.with_suffix('.json').write_text(json.dumps(dict(sequence_id=sequence, frame=frame,
            resolution_level=level, source_region_xyxy=[0, 0, 1920, 1080],
            original_width=3840, original_height=2160, center_x=960, center_y=540,
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())))
        path = self.labels/relative.with_suffix('.txt')
        if label is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(label, encoding='utf-8')
        return image, path

    def build(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return build_dataset(self.captures, self.labels, self.validation, **kwargs)

    def make_helsinki(self):
        for split, value in (('train', 245), ('val', 246)):
            (self.helsinki/'images'/split).mkdir(parents=True)
            (self.helsinki/'labels'/split).mkdir(parents=True)
            (self.helsinki/'images'/split/'same_name.png').write_bytes(png_bytes(value))
            (self.helsinki/'labels'/split/'same_name.txt').write_text('0 0.5 0.5 0.1 0.1\n')
        write_yaml(self.helsinki, split=True)

    def test_matching_collision_safe_names_class_order_and_original_bytes(self):
        originals = [self.capture(phase=0, value=10), self.capture(phase=1, value=20),
                     self.capture(phase=0, value=30, suffix='_retry')]
        summary = self.build()
        self.assertEqual(summary['unique_samples'], 3)
        outputs = list((self.validation/'images/all').glob('*.png'))
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len({p.stem for p in outputs}), 3)
        self.assertEqual({p.read_bytes() for p in outputs}, {image.read_bytes() for image, _ in originals})
        config = yaml.safe_load((self.validation/'data.yaml').read_text())
        self.assertEqual(list(config['names'].values()), list(OBJECT_CLASSES))
        self.assertNotIn('train', config)
        self.assertNotIn('val', config)
        with (self.validation/'manifest.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['output_filename']
                self.assertTrue((self.validation/'labels/all'/Path(name).with_suffix('.txt')).is_file())
                self.assertEqual(row['annotation_count'], '1')
                self.assertEqual(row['source_frame'], '123')
                self.assertEqual(row['source_region_xyxy'], '[0, 0, 1920, 1080]')

    def test_dedup_requires_identical_image_and_label_bytes_and_keeps_provenance(self):
        self.capture(phase=0)
        self.capture(phase=1)
        self.capture(phase=2, label='0 0.5 0.5 0.2 0.2\n')
        self.capture(phase=3, value=51)
        result = self.build()
        self.assertEqual((result['source_captures'], result['unique_samples'], result['deduplicated']), (4, 3, 1))
        with (self.validation/'manifest.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['output_filename'], rows[1]['output_filename'])

    def test_unreviewed_skipped_but_partial_missing_labels_rejected(self):
        self.capture(frame=1)
        self.capture(frame=2, label=None)
        with self.assertRaisesRegex(ValueError, 'no projected TXT'):
            self.build(require_all_labels=True)
        self.capture(frame=1, phase=1, label=None)
        with self.assertRaisesRegex(ValueError, 'frame labeled elsewhere'):
            self.build()
        self.assertFalse(self.validation.exists())

    def test_main_sequence_only_and_unreviewed_count(self):
        self.capture(frame=1)
        self.capture(frame=2, label=None)
        self.capture(frame=0, sequence='Verify', label=None)
        self.capture(frame=3, level=0, label=None)
        result = self.build()
        self.assertEqual(result['source_captures'], 1)
        self.assertEqual(result['skipped_unreviewed_captures'], 1)

    def test_orphan_label_and_hash_mismatch_fail_before_writes(self):
        image, _ = self.capture()
        orphan = self.labels/'orphan.txt'
        orphan.write_text('')
        with self.assertRaisesRegex(ValueError, 'do not match'):
            self.build()
        orphan.unlink()
        image.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.build()
        self.assertFalse(self.validation.exists())

    def test_yolo_label_validation(self):
        label = self.root/'label.txt'
        for text in ('16 0.5 0.5 0.1 0.1', '0 0.5 0.5 0 0.1', '0 nan 0.5 0.1 0.1',
                     '0 0.99 0.5 0.2 0.2', '0 0.5 0.5 0.1 0.1 0.2'):
            label.write_text(text)
            with self.assertRaises(ValueError):
                validate_yolo(label)
        label.write_text('')
        self.assertEqual(validate_yolo(label), (b'', 0))

    def test_contiguous_holdout_no_source_frame_leak_and_helsinki_preserved(self):
        for index, frame in enumerate((199, 200, 249, 250)):
            self.capture(frame=frame, value=index+1)
            self.capture(frame=frame, phase=1, value=index+11)
        self.build()
        self.make_helsinki()
        output = self.root/'combined'
        with redirect_stdout(io.StringIO()):
            prepare(self.helsinki, self.validation, output, 200, 249)
        with (output/'manifest.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        frame_splits = {}
        for row in rows:
            self.assertTrue(row['output_filename'].startswith(row['dataset']+'_'))
            for frame in json.loads(row['source_frames']):
                frame_splits.setdefault(frame, set()).add(row['split'])
        self.assertEqual(frame_splits, {199: {'train'}, 200: {'val'}, 249: {'val'}, 250: {'train'}})
        self.assertEqual({r['split'] for r in rows if r['dataset'] == 'helsinki'}, {'train', 'val'})
        config = yaml.safe_load((output/'data.yaml').read_text())
        self.assertEqual(config['train'], 'images/train')
        self.assertEqual(config['val'], 'images/val')
        self.assertEqual(list(config['names'].values()), list(OBJECT_CLASSES))

    def test_deduplicated_aliases_crossing_holdout_rejected(self):
        self.capture(frame=199, value=1)
        self.capture(frame=200, value=1)
        self.build()
        self.make_helsinki()
        output = self.root/'combined'
        with self.assertRaisesRegex(ValueError, 'spans holdout boundary'):
            prepare(self.helsinki, self.validation, output, 200, 249)
        self.assertFalse(output.exists())

    def test_identical_images_with_different_labels_cannot_leak(self):
        self.capture(frame=199, value=1)
        self.capture(frame=200, value=1, label='0 0.5 0.5 0.2 0.2\n')
        self.build()
        self.make_helsinki()
        with self.assertRaisesRegex(ValueError, 'Identical image bytes'):
            prepare(self.helsinki, self.validation, self.root/'combined', 200, 249)

    def test_coco_to_projector_to_packager_end_to_end(self):
        for frame in (2, 123):
            self.capture(frame=frame, value=frame, label=None)
        source = self.root/'coco.json'
        annotations = self.root/'annotations.json'
        source.write_text(json.dumps(coco_fixture()))
        projected = self.root/'new_projected'
        with redirect_stdout(io.StringIO()):
            convert_file(source, annotations)
            project_labels(self.captures, annotations, projected)
            result = build_dataset(self.captures, projected, self.validation)
        self.assertEqual(result['unique_samples'], 2)
        labels = [p.read_text() for p in (self.validation/'labels/all').glob('*.txt')]
        self.assertIn('', labels)
        populated = next(text for text in labels if text)
        self.assertEqual([int(line.split()[0]) for line in populated.splitlines()],
                         [OBJECT_CLASSES.index('hangar'), OBJECT_CLASSES.index('ta-ta')])


if __name__ == '__main__':
    unittest.main()
