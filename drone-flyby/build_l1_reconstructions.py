"""Build half-resolution L1 mosaics; see L1_RECONSTRUCTIONS.md for usage."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SOURCE_SIZE = (3840, 2160)
RECONSTRUCTION_SIZE = (1920, 1080)
CROP_SIZE = (960, 540)
PHASES = tuple(f'l1_phase_{i:02d}' for i in range(9))


@dataclass(frozen=True)
class Capture:
    path: Path
    frame: int
    sequence_id: str
    region: tuple
    sha256: str


def validate_region(region):
    if (len(region) != 4 or any(type(v) is not int for v in region) or
            any(v % 2 for v in region)):
        raise ValueError(f'L1 region must have four even integer edges: {region}')
    x1, y1, x2, y2 = region
    if not (0 <= x1 < x2 <= 3840 and 0 <= y1 < y2 <= 2160 and
            x2-x1 == 1920 and y2-y1 == 1080):
        raise ValueError(f'Invalid L1 source region: {region}')
    return tuple(region)


def load_captures(root):
    """Select longest sequence per phase by distinct frames across received levels.

    Missing phase directories are allowed. Ties use sequence ID lexical order.
    Only matching PNG/JSON pairs participate; malformed metadata fails explicitly.
    """
    root = Path(root).expanduser().resolve()
    captures, selections = [], {}
    for phase in PHASES:
        sequences = defaultdict(list)
        for sidecar in sorted((root / phase).rglob('*.json')):
            png = sidecar.with_suffix('.png')
            if not png.is_file():
                continue
            data = json.loads(sidecar.read_text(encoding='utf-8'))
            if type(data['frame']) is not int or data['frame'] < 0:
                raise ValueError(f'Invalid frame in {sidecar}')
            sequences[str(data['sequence_id'])].append((png, data))
        if not sequences:
            continue
        sequence = min(sequences, key=lambda s: (-len({d['frame'] for _, d in sequences[s]}), s))
        selections[phase] = sequence
        for png, data in sequences[sequence]:
            if data['resolution_level'] != 1:
                continue
            if (data['original_width'], data['original_height']) != SOURCE_SIZE:
                raise ValueError(f'Expected original size {SOURCE_SIZE}: {png}')
            captures.append(Capture(png, data['frame'], sequence,
                                    validate_region(data['source_region_xyxy']), data['image_sha256']))
    return sorted(captures, key=lambda c: c.path.as_posix()), selections


def unique_regions(captures):
    """First path in lexical order wins identical-region ties, including retries."""
    by_region = {}
    for capture in sorted(captures, key=lambda c: c.path.as_posix()):
        by_region.setdefault(capture.region, capture)
    return list(by_region.values())


def read_capture(capture):
    raw = capture.path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != capture.sha256:
        raise ValueError(f'Image SHA256 mismatch: {capture.path}')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (540, 960, 3) or image.dtype != np.uint8:
        raise ValueError(f'Expected 960x540, 8-bit RGB PNG: {capture.path}')
    return image


def reconstruct(captures):
    chosen = unique_regions(captures)
    total = np.zeros((1080, 1920, 3), dtype=np.uint32)
    count = np.zeros((1080, 1920), dtype=np.uint32)
    for capture in chosen:
        region = validate_region(capture.region)
        image = read_capture(capture)
        x, y = region[0] // 2, region[1] // 2
        total[y:y+540, x:x+960] += image
        count[y:y+540, x:x+960] += 1
    # Integer round-half-up averaging is deterministic; zero coverage stays black.
    divisor = np.maximum(count, 1)[..., None]
    image = ((total + divisor//2) // divisor).astype(np.uint8)
    mask = (count > 0).astype(np.uint8)*255
    return image, mask, chosen


def create_output(root, output):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if root.is_relative_to(output) or any(output.is_relative_to((root / p).resolve()) for p in PHASES):
        raise ValueError('Output must not contain or be inside source phase directories')
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier result.
    return output


def write_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def build(root, output=None):
    root = Path(root).expanduser().resolve()
    captures, selections = load_captures(root)
    if not captures:
        raise ValueError(f'No L1 captures in selected sequences under {root}')
    output = create_output(root, output if output is not None else root / 'reconstructed_l1')
    for folder in ('images', 'masks', 'metadata'):
        (output / folder).mkdir()
    frames = defaultdict(list)
    for capture in captures:
        frames[capture.frame].append(capture)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'unique_regions', 'coverage_percent',
                                                    'image_path', 'mask_path'])
        writer.writeheader()
        for frame, items in sorted(frames.items()):
            image, mask, chosen = reconstruct(items)
            stem = f'frame_{frame:06d}'
            image_path, mask_path = f'images/{stem}.png', f'masks/{stem}.png'
            write_png(output / image_path, image)
            write_png(output / mask_path, mask)
            coverage = float(np.count_nonzero(mask)*100 / mask.size)
            metadata = dict(frame=frame, available_l1_regions=[list(c.region) for c in chosen],
                unique_regions=len(chosen), coverage_percent=coverage,
                contributing_capture_paths=[str(c.path) for c in chosen],
                available_capture_paths=[str(c.path) for c in items], selected_sequences=selections,
                original_size=list(SOURCE_SIZE), reconstruction_size=list(RECONSTRUCTION_SIZE),
                overlap_method='integer_mean_round_half_up', duplicate_policy='first_lexical_path')
            (output / 'metadata' / f'{stem}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            writer.writerow(dict(frame=frame, unique_regions=len(chosen), coverage_percent=f'{coverage:.6f}',
                                 image_path=image_path, mask_path=mask_path))
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=Path('~/drone-captures'))
    parser.add_argument('--output', type=Path, help='New output directory; default: <capture-root>/reconstructed_l1')
    args = parser.parse_args()
    print(f'Built {build(args.capture_root, args.output)} reconstructed frames')


if __name__ == '__main__':
    main()
