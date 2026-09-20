"""Deterministic camera-view crops; all derivatives of a frame share a split."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE
from prepare_yolo import ROOT


def crop_labels(objects, region, scout=False):
    left, top, right, bottom = region
    width, height = right-left, bottom-top
    labels = []
    for obj in objects:
        class_id = OBJECT_CLASSES.index(obj['object_id'])
        x1, y1, x2, y2 = obj['bbox']
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError(f'Invalid annotation: {obj}')
        x1, y1, x2, y2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append((0 if scout else class_id, (x1+x2-2*left)/(2*width),
                       (y1+y2-2*top)/(2*height), (x2-x1)/width, (y2-y1)/height))
    return labels


def make_region(level, cx, cy):
    width, height = SOURCE_REGION_SIZES[level]
    left = min(max(round(cx-width/2), 0), 3840-width)
    top = min(max(round(cy-height/2), 0), 2160-height)
    return left, top, left+width, top+height


def write_sample(image, objects, region, output, split, name, scout=False):
    left, top, right, bottom = region
    view = cv2.resize(image[top:bottom, left:right], TRANSMITTED_VIEW_SIZE,
                      interpolation=cv2.INTER_AREA)
    labels = crop_labels(objects, region, scout)
    if not cv2.imwrite(str(output / 'images' / split / f'{name}.png'), view):
        raise OSError(f'Could not write {name}')
    text = ''.join(f'{row[0]} ' + ' '.join(f'{v:.10f}' for v in row[1:]) + '\n' for row in labels)
    (output / 'labels' / split / f'{name}.txt').write_text(text, encoding='utf-8')


def generate(source, output, l1_crops_per_object=1, l2_crops_per_object=2,
             background_crops=2, seed=42, val_fraction=0.2, scout=False, train_frames=None):
    """``train_frames`` (frame numbers), when given, picks an explicit
    training set instead of a temporal prefix -- see prepare_yolo.convert's
    docstring for why a short prefix can miss classes entirely. All crops
    derived from one source frame still share that frame's split.
    """
    source, output = Path(source), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if not 0 < val_fraction < 1 or min(l1_crops_per_object, l2_crops_per_object, background_crops) < 0:
        raise ValueError('Invalid split fraction or crop count')
    frames = sorted((source / 'images').glob('*.png'))
    if len(frames) < 2:
        raise ValueError('Need at least two source frames')
    # Validate annotations before writing any derived samples.
    annotations = {}
    for frame in frames:
        objects = json.loads((source/'annotations'/f'{frame.stem}.json').read_text())['annotations']
        crop_labels(objects, (0, 0, 3840, 2160))
        annotations[frame.stem] = objects
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            (output/kind/split).mkdir(parents=True)
    if train_frames is not None:
        present = {int(frame.stem.split('_')[-1]) for frame in frames}
        unknown = set(train_frames) - present
        if unknown:
            raise ValueError(f'--train-frames names frames not present: {sorted(unknown)}')
        if len(train_frames) >= len(frames):
            raise ValueError('train_frames must leave at least one frame for val')
        train_frame_numbers = set(train_frames)
    else:
        split_at = len(frames)-max(1, min(len(frames)-1, round(len(frames)*val_fraction)))
        train_frame_numbers = {int(f.stem.split('_')[-1]) for f in frames[:split_at]}
    rng, counts, manifest = random.Random(seed), Counter(), []
    for index, frame in enumerate(frames):
        split = 'train' if int(frame.stem.split('_')[-1]) in train_frame_numbers else 'val'
        image = cv2.imread(str(frame))
        if image is None or image.shape[:2] != (2160, 3840):
            raise ValueError(f'Expected 3840x2160 source: {frame}')
        objects = annotations[frame.stem]
        regions = [(0, (0, 0, 3840, 2160), False)]
        if not scout:
            for level, repetitions in ((1, l1_crops_per_object), (2, l2_crops_per_object)):
                width, height = SOURCE_REGION_SIZES[level]
                seen = set()
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    for _ in range(repetitions):
                        region = make_region(level, (x1+x2)/2+rng.uniform(-0.3, 0.3)*width,
                                             (y1+y2)/2+rng.uniform(-0.3, 0.3)*height)
                        if region not in seen:
                            regions.append((level, region, False))
                            seen.add(region)
                found = 0
                for _ in range(max(100, background_crops*200)):
                    if found >= background_crops:
                        break
                    region = make_region(level, rng.uniform(width/2, 3840-width/2),
                                         rng.uniform(height/2, 2160-height/2))
                    if region not in seen and not crop_labels(objects, region):
                        regions.append((level, region, True))
                        seen.add(region)
                        found += 1
                counts['background_shortfall'] += background_crops-found
        for number, (level, region, background) in enumerate(regions):
            name = f'{frame.stem}_L{level}_{number:04d}'
            write_sample(image, objects, region, output, split, name, scout)
            counts[f'L{level}'] += 1
            counts[split] += 1
            counts['background'] += int(background)
            manifest.append(dict(sample=name, frame=frame.stem, split=split, level=level,
                                 region=region, background=background))
    names = ('target',) if scout else OBJECT_CLASSES
    (output/'data.yaml').write_text(
        f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\nnames:\n' +
        ''.join(f'  {i}: {json.dumps(name)}\n' for i, name in enumerate(names)), encoding='utf-8')
    report = dict(counts=dict(counts), seed=seed, val_fraction=val_fraction,
                  l1_crops_per_object=l1_crops_per_object, l2_crops_per_object=l2_crops_per_object,
                  background_crops=background_crops, scout=scout, samples=manifest)
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'src/helsinki')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--l1-crops-per-object', type=int, default=1)
    parser.add_argument('--l2-crops-per-object', type=int, default=2)
    parser.add_argument('--background-crops', type=int, default=2, help='Per frame per L1/L2; only verified empty crops')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--scout', action='store_true', help='L0 only, one target class; no inference integration')
    parser.add_argument(
        '--train-frames', type=int, nargs='+', default=None,
        help='Explicit frame numbers for training (rest become val); overrides --val-fraction',
    )
    args = vars(parser.parse_args())
    args['output'] = args['output'] or ROOT/'datasets'/('helsinki_scout' if args['scout'] else 'helsinki_multires')
    if args['train_frames'] is not None:
        args['train_frames'] = set(args['train_frames'])
    generate(**args)
