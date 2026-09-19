"""Read-only handoff checks. No checkpoint loading, inference, training, or network."""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ('cvat_coco_to_reconstruction.py', 'audit_reconstruction_annotations.py',
           'build_l1_reconstructions.py', 'project_reconstruction_labels.py',
           'build_validation_yolo_dataset.py', 'prepare_combined_finetune_dataset.py',
           'evaluate_validation_annotations.py', 'validation_dataset_utils.py',
           'train_yolo.py', 'resume_validation_finetune.py', 'run_validation_finetune.ps1')
IMPORTS = ('numpy', 'cv2', 'yaml', 'pydantic', 'torch', 'ultralytics', 'faster_coco_eval')


def dataset_paths(path):
    import yaml
    from dtos import OBJECT_CLASSES
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names.get(i) for i in range(len(OBJECT_CLASSES))] if len(names) == len(OBJECT_CLASSES) else None
    if names != list(OBJECT_CLASSES):
        raise ValueError('Dataset class names/order do not match OBJECT_CLASSES')
    root = Path(data.get('path', '.')).expanduser()
    root = root if root.is_absolute() else path.parent / root
    paths = []
    for split in ('train', 'val'):
        directory = Path(data[split])
        directory = directory if directory.is_absolute() else root / directory
        paths.append(directory.resolve())
        paths.append((root / 'labels' / split).resolve())
    return paths


def check(capture_root, coco, weights, dataset_yaml, require_dataset=False, repo=ROOT):
    errors = []
    def report(ok, text):
        print(('OK   ' if ok else 'FAIL ') + text)
        if not ok:
            errors.append(text)
    for phase in range(9):
        directory = capture_root / f'l1_phase_{phase:02d}'
        report(directory.is_dir(), str(directory))
    for script in SCRIPTS:
        report((repo / script).is_file(), script)
    for module in IMPORTS:
        report(importlib.util.find_spec(module) is not None, f'Python module discoverable: {module}')
    report(weights.is_file(), f'Checkpoint file exists (not loaded): {weights}')
    report(coco.is_file(), f'CVAT JSON: {coco}')
    if coco.is_file():
        try:
            data = json.loads(coco.read_text(encoding='utf-8-sig'))
            print(f'INFO CVAT images={len(data["images"])} boxes={len(data["annotations"])} (handoff: 180 / 687)')
        except (ValueError, KeyError, TypeError) as exc:
            report(False, f'Invalid CVAT JSON: {exc}')
    if dataset_yaml.is_file():
        try:
            for directory in dataset_paths(dataset_yaml):
                report(directory.is_dir(), f'Dataset directory: {directory}')
        except (ValueError, KeyError, TypeError, ImportError) as exc:
            report(False, f'Dataset YAML invalid: {exc}')
    elif require_dataset:
        report(False, f'Dataset YAML missing: {dataset_yaml}')
    else:
        print(f'INFO Dataset not built yet: {dataset_yaml}; follow EXTERNAL_DATA.md')
    print(f'Preflight: {len(errors)} failure(s). No inference or training performed.')
    return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=Path.home()/'nordicai/drone-captures')
    parser.add_argument('--coco', type=Path, default=ROOT/'validation_cvat.json')
    parser.add_argument('--weights', type=Path, default=ROOT/'runs/detect/helsinki_yolov8n_multires_rot/weights/best.pt')
    parser.add_argument('--dataset-yaml', type=Path)
    parser.add_argument('--require-dataset', action='store_true')
    args = parser.parse_args()
    capture_root = args.capture_root.expanduser().resolve()
    success = check(capture_root, args.coco.expanduser().resolve(), args.weights.expanduser().resolve(),
                    (args.dataset_yaml or capture_root/'combined_finetune/data.yaml').expanduser().resolve(),
                    args.require_dataset)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
