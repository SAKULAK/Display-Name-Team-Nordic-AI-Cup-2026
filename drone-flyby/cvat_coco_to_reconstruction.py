"""Convert CVAT COCO detection JSON to reviewed reconstruction pixel-XYXY labels."""

import argparse
from collections import Counter
import json
import math
import re

from dtos import OBJECT_CLASSES
from validation_dataset_utils import resolved, warn_absent_classes, write_new_json

EPSILON = 1e-6  # Absolute reconstructed-image pixels, not a fraction of box size.


def source_frame(file_name):
    name = str(file_name).replace('\\', '/').rsplit('/', 1)[-1]
    match = re.fullmatch(r'frame_(\d{6,})\.png', name)
    if not match:
        raise ValueError(f'Expected frame_XXXXXX.png filename: {file_name!r}')
    return int(match.group(1))


def xywh_to_xyxy(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError(f'Expected four finite bbox coordinates: {box}')
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        raise ValueError(f'Nonpositive bbox size: {box}')
    x2, y2 = x+width, y+height
    if x < -EPSILON or y < -EPSILON or x2 > 1920+EPSILON or y2 > 1080+EPSILON:
        raise ValueError(f'BBox outside 1920x1080 canvas: {box}')
    result = [max(0.0, x), max(0.0, y), min(1920.0, x2), min(1080.0, y2)]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f'BBox has no area after epsilon clamping: {box}')
    return result


def convert_coco(coco):
    categories = {}
    for category in coco['categories']:
        key, name = category['id'], category['name']
        if type(key) is not int or key in categories:
            raise ValueError(f'Invalid/duplicate category ID: {key}')
        if name not in OBJECT_CLASSES:
            raise ValueError(f'Unknown category name: {name}')
        categories[key] = name
    images, frames = {}, {}
    for image in coco['images']:
        key = image['id']
        if type(key) is not int or key in images:
            raise ValueError(f'Invalid/duplicate image ID: {key}')
        if (image['width'], image['height']) != (1920, 1080):
            raise ValueError(f'Expected 1920x1080 image: {image["file_name"]}')
        frame = source_frame(image['file_name'])
        if frame in frames:
            raise ValueError(f'Duplicate source frame filename: {image["file_name"]}')
        images[key], frames[frame] = frame, []
    for annotation in coco['annotations']:
        image_id, category_id = annotation['image_id'], annotation['category_id']
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f'Annotation references unknown image ID: {image_id}')
        if type(category_id) is not int or category_id not in categories:
            raise ValueError(f'Annotation references unknown category ID: {category_id}')
        frames[images[image_id]].append({'object_id': categories[category_id],
                                       'bbox': xywh_to_xyxy(annotation['bbox'])})
    class_ids = {name: i for i, name in enumerate(OBJECT_CLASSES)}
    result = {'frames': [{'frame': frame, 'annotations': sorted(annotations,
              key=lambda a: (class_ids[a['object_id']], *a['bbox']))}
              for frame, annotations in sorted(frames.items())]}
    counts = Counter(a['object_id'] for annotations in frames.values() for a in annotations)
    annotated = sum(bool(annotations) for annotations in frames.values())
    summary = dict(images=len(frames), annotated_images=annotated, empty_images=len(frames)-annotated,
                   boxes=sum(counts.values()), per_class_counts={name: counts[name] for name in OBJECT_CLASSES},
                   first_frame=min(frames, default=None), last_frame=max(frames, default=None),
                   classes_with_zero_annotations=[name for name in OBJECT_CLASSES if not counts[name]])
    return result, summary


def convert_file(coco_path, output, summary_path=None):
    source, output = resolved(coco_path), resolved(output)
    destinations = [output] + ([resolved(summary_path)] if summary_path is not None else [])
    if len(set(destinations)) != len(destinations) or source in destinations:
        raise ValueError('Input, output, and summary must be distinct paths')
    for path in destinations:
        if path.exists():
            raise FileExistsError(f'Choose a new output path: {path}')
    result, summary = convert_coco(json.loads(source.read_text(encoding='utf-8-sig')))
    write_new_json(output, result)
    if summary_path is not None:
        write_new_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    warn_absent_classes(summary['per_class_counts'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary')
    args = parser.parse_args()
    convert_file(args.coco, args.output, args.summary)


if __name__ == '__main__':
    main()
