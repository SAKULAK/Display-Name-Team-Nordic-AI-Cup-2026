"""Convert Helsinki source-pixel annotations to a standalone YOLO dataset."""

import argparse
import json
import math
import shutil
from pathlib import Path

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'datasets' / 'helsinki'


def convert(source=ROOT / 'src' / 'helsinki', output=DEFAULT_OUTPUT, val_fraction=0.2):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new --output')
    images = sorted(p for p in (source / 'images').iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if len(images) < 2 or len({p.stem for p in images}) != len(images):
        raise ValueError('Need at least two images with unique stems')
    annotations = {p.stem: p for p in (source / 'annotations').glob('*.json')}
    if set(annotations) != {p.stem for p in images}:
        raise ValueError('Images and annotation JSON stems must match exactly')
    class_ids = {name: i for i, name in enumerate(OBJECT_CLASSES)}
    records = []
    # Validate everything before creating output. Missing labels are never negatives.
    for image in images:
        lines = []
        objects = json.loads(annotations[image.stem].read_text(encoding='utf-8'))['annotations']
        for obj in objects:
            class_id = class_ids[obj['object_id']]
            x1, y1, x2, y2 = map(float, obj['bbox'])
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise ValueError(f'Non-finite box in {image.stem}')
            if not (0 <= x1 < x2 <= IMAGE_WIDTH and 0 <= y1 < y2 <= IMAGE_HEIGHT):
                raise ValueError(f'Invalid source box in {image.stem}: {obj}')
            box = ((x1 + x2) / (2 * IMAGE_WIDTH), (y1 + y2) / (2 * IMAGE_HEIGHT),
                   (x2 - x1) / IMAGE_WIDTH, (y2 - y1) / IMAGE_HEIGHT)
            lines.append(f'{class_id} ' + ' '.join(f'{v:.10f}' for v in box))
        records.append((image, '\n'.join(lines) + ('\n' if lines else '')))
    val_count = max(1, min(len(images) - 1, round(len(images) * val_fraction)))
    split_at = len(images) - val_count
    for i, (image, labels) in enumerate(records):
        split = 'train' if i < split_at else 'val'
        image_dir, label_dir = output / 'images' / split, output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dir / image.name)
        (label_dir / f'{image.stem}.txt').write_text(labels, encoding='utf-8')
    # Absolute path avoids Ultralytics' configurable datasets_dir resolution.
    yaml = f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\nnames:\n'
    yaml += ''.join(f'  {i}: {json.dumps(name)}\n' for i, name in enumerate(OBJECT_CLASSES))
    (output / 'data.yaml').write_text(yaml, encoding='utf-8')
    print(f'Converted {split_at} train / {val_count} val images: {output / "data.yaml"}')
    return output / 'data.yaml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'src' / 'helsinki')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    convert(args.source, args.output, args.val_fraction)
