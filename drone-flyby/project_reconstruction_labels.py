"""Project reconstructed-frame pixel XYXY JSON labels into captured L1 YOLO labels."""

import argparse
import json
import math
from pathlib import Path

from dtos import OBJECT_CLASSES
from build_l1_reconstructions import load_captures, create_output, validate_region


def reconstruction_to_source(box):
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                            not math.isfinite(v) for v in box):
        raise ValueError('bbox must contain four finite pixel coordinates')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
        raise ValueError('bbox must be a positive-area XYXY box within 1920x1080')
    return tuple(float(v)*2 for v in box)


def project_box(box, region, min_visible_fraction=0.7):
    if not 0 <= min_visible_fraction <= 1:
        raise ValueError('Minimum visible fraction must be in [0, 1]')
    rx1, ry1, _, _ = validate_region(region)
    gx1, gy1, gx2, gy2 = reconstruction_to_source(box)
    local = ((gx1-rx1)/2, (gy1-ry1)/2, (gx2-rx1)/2, (gy2-ry1)/2)
    x1, y1, x2, y2 = local
    clipped = (max(0.0, x1), max(0.0, y1), min(960.0, x2), min(540.0, y2))
    a, b, c, d = clipped
    if c <= a or d <= b:
        return None
    fraction = (c-a)*(d-b)/((x2-x1)*(y2-y1))
    return clipped if fraction >= min_visible_fraction else None


def yolo_line(object_id, box):
    class_id = OBJECT_CLASSES.index(object_id)
    x1, y1, x2, y2 = box
    return f'{class_id} {(x1+x2)/1920:.8f} {(y1+y2)/1080:.8f} {(x2-x1)/960:.8f} {(y2-y1)/540:.8f}'


def load_annotations(path):
    """Read {"frames": [{"frame": N, "annotations": [{"object_id": ..., "bbox": [...]}]}]}."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    frames = {}
    for item in payload['frames']:
        frame = item['frame']
        if type(frame) is not int or frame < 0 or frame in frames:
            raise ValueError(f'Invalid or duplicate annotation frame: {frame}')
        annotations = item['annotations']
        for annotation in annotations:
            if annotation['object_id'] not in OBJECT_CLASSES:
                raise ValueError(f'Unknown object_id: {annotation["object_id"]}')
            reconstruction_to_source(annotation['bbox'])
        frames[frame] = annotations
    return frames


def project_labels(root, annotations_path, output=None, min_visible_fraction=0.7):
    if not 0 <= min_visible_fraction <= 1:
        raise ValueError('Minimum visible fraction must be in [0, 1]')
    root = Path(root).expanduser().resolve()
    frames = load_annotations(Path(annotations_path).expanduser())
    captures, _ = load_captures(root)
    selected = [capture for capture in captures if capture.frame in frames]
    if not selected:
        raise ValueError('No captured L1 images match the annotated frames')
    output = create_output(root, output if output is not None else root / 'projected_l1_labels')
    for capture in selected:  # Keep all captures, including duplicate source regions.
        lines = []
        for annotation in frames[capture.frame]:
            box = project_box(annotation['bbox'], capture.region, min_visible_fraction)
            if box is not None:
                lines.append(yolo_line(annotation['object_id'], box))
        path = output / capture.path.relative_to(root).with_suffix('.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(line+'\n' for line in lines), encoding='utf-8')
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=Path('~/drone-captures'))
    parser.add_argument('--annotations', required=True, type=Path, help='Reconstruction pixel-XYXY JSON')
    parser.add_argument('--output', type=Path, help='New directory; default: <capture-root>/projected_l1_labels')
    parser.add_argument('--min-visible-fraction', type=float, default=0.7)
    args = parser.parse_args()
    print(f'Wrote {project_labels(args.capture_root, args.annotations, args.output, args.min_visible_fraction)} label files')


if __name__ == '__main__':
    main()
