"""Shared file validation for offline dataset preparation; no model imports."""

import hashlib
import json
import math
from pathlib import Path

import yaml

from dtos import OBJECT_CLASSES


def resolved(path):
    return Path(path).expanduser().resolve()


def new_output(output, protected=()):
    output = resolved(output)
    for source in protected:
        source = resolved(source)
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f'Output overlaps input: {source}')
    output.mkdir(parents=True, exist_ok=False)
    return output


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_yolo(path):
    """Validate detection labels; missing labels are never negative samples."""
    raw = Path(path).read_bytes()
    count = 0
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5 or not fields[0].isdigit():
                raise ValueError('expected class_id cx cy w h')
            class_id = int(fields[0])
            cx, cy, width, height = map(float, fields[1:])
            if not (0 <= class_id < len(OBJECT_CLASSES) and
                    all(math.isfinite(v) for v in (cx, cy, width, height)) and
                    0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1 and
                    cx-width/2 >= -1e-7 and cy-height/2 >= -1e-7 and
                    cx+width/2 <= 1+1e-7 and cy+height/2 <= 1+1e-7):
                raise ValueError('invalid class or normalized box')
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        count += 1
    return raw, count


def write_yaml(output, split=False):
    data = {'path': Path(output).as_posix()}
    if split:
        data.update(train='images/train', val='images/val')
    else:
        # Deliberately not train-ready: the merge script supplies a source-frame split.
        data['all'] = 'images/all'
    data['names'] = dict(enumerate(OBJECT_CLASSES))
    (Path(output) / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def check_class_order(config):
    names = config.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(OBJECT_CLASSES))):
            raise ValueError('Dataset names must use class indices 0..15')
        names = [names[i] for i in range(len(OBJECT_CLASSES))]
    if names != list(OBJECT_CLASSES):
        raise ValueError('Dataset names must match dtos.OBJECT_CLASSES exactly')


def warn_absent_classes(counts):
    absent = [name for name in OBJECT_CLASSES if counts.get(name, 0) == 0]
    if absent:
        print('WARNING: Classes with zero human annotations: ' + ', '.join(absent) +
              '. Verify these objects were truly absent from reviewed frames; otherwise '
              'unlabeled visible objects will be treated as background.')
    return absent


def write_new_json(path, data):
    path = resolved(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
