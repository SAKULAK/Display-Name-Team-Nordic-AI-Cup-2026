"""Synthetic handoff checks; no model imports, checkpoint loads or training."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from dtos import OBJECT_CLASSES
from teammate_preflight import ROOT, SCRIPTS, check, dataset_paths


class HandoffTests(unittest.TestCase):
    def test_template_preserves_class_order(self):
        config = yaml.safe_load((ROOT/'configs/combined_finetune.example.yaml').read_text())
        self.assertEqual(list(config['names'].values()), list(OBJECT_CLASSES))
        self.assertEqual(config['train'], 'images/train')
        record = json.loads((ROOT/'configs/validation_handoff.json').read_text())
        self.assertEqual((record['reviewed_frames'], record['human_boxes']), (180, 687))

    def test_portable_paths_and_preflight_without_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(9):
                (root/f'l1_phase_{index:02d}').mkdir()
            for script in SCRIPTS:
                (root/script).touch()
            coco, weights, dataset = root/'cvat.json', root/'best.pt', root/'data.yaml'
            coco.write_text('{"images": [], "annotations": []}')
            weights.write_bytes(b'never opened')
            dataset.write_text(yaml.safe_dump(dict(path='.', train='images/train', val='images/val',
                                                 names=list(OBJECT_CLASSES))))
            for path in dataset_paths(dataset):
                path.mkdir(parents=True)
            with patch('teammate_preflight.importlib.util.find_spec', return_value=object()), redirect_stdout(io.StringIO()):
                self.assertTrue(check(root, coco, weights, dataset, True, repo=root))
                weights.unlink()
                self.assertFalse(check(root, coco, weights, dataset, True, repo=root))

    def test_stale_dataset_paths_are_detectable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.yaml'
            path.write_text(yaml.safe_dump(dict(path='not_here', train='images/train', val='images/val',
                                                names=list(OBJECT_CLASSES))))
            self.assertTrue(all(not item.exists() for item in dataset_paths(path)))


if __name__ == '__main__':
    unittest.main()
