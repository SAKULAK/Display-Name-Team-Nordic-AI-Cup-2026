"""Copy labeled main-sequence L1 captures into an unsplit, provenance-preserving dataset."""

import argparse
import csv
import json
import shutil
import struct

from build_l1_reconstructions import load_captures, PHASES
from validation_dataset_utils import file_sha256, new_output, resolved, validate_yolo, write_yaml

MANIFEST_FIELDS = ('output_filename', 'phase', 'sequence_id', 'source_frame', 'center_x', 'center_y',
                   'source_region_xyxy', 'source_image_path', 'source_label_path', 'image_sha256',
                   'annotation_count', 'deduplicated')


def validate_capture_image(capture):
    digest = file_sha256(capture.path)
    if digest != capture.sha256:
        raise ValueError(f'Capture SHA256 mismatch: {capture.path}')
    with capture.path.open('rb') as stream:
        header = stream.read(24)
    if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR' or
            struct.unpack('>II', header[16:24]) != (960, 540)):
        raise ValueError(f'Expected original 960x540 PNG: {capture.path}')
    return digest


def build_dataset(capture_root, labels_root, output, require_all_labels=False):
    root, labels_root = resolved(capture_root), resolved(labels_root)
    if not labels_root.is_dir():
        raise FileNotFoundError(f'Projected label directory not found: {labels_root}')
    captures, selections = load_captures(root)  # Exactly the reconstruction's selection logic.
    pairs = [(c, labels_root / c.path.relative_to(root).with_suffix('.txt')) for c in captures]
    expected = {label for _, label in pairs}
    orphans = sorted(set(labels_root.rglob('*.txt'))-expected)
    if orphans:
        raise ValueError(f'{len(orphans)} labels do not match selected main-sequence L1 captures; first: {orphans[0]}')
    missing = [c for c, label in pairs if not label.is_file()]
    if missing and require_all_labels:
        raise ValueError(f'{len(missing)} selected captures have no projected TXT; first: {missing[0].path}')
    reviewed_frames = {c.frame for c, label in pairs if label.is_file()}
    incomplete = [c for c in missing if c.frame in reviewed_frames]
    if incomplete:
        raise ValueError(f'Missing projected label for a frame labeled elsewhere: {incomplete[0].path}')
    print(f'Selected sequences: {json.dumps(selections, sort_keys=True)}')
    print(f'Skipping {len(missing)} captures without labels (unreviewed source frames); no negatives invented.')
    prepared, records, canonical = [], [], {}
    for index, (capture, label) in enumerate(pairs):
        if not label.is_file():
            continue
        raw_labels, annotation_count = validate_yolo(label)
        digest = validate_capture_image(capture)
        metadata = json.loads(capture.path.with_suffix('.json').read_text(encoding='utf-8'))
        x, y = metadata['center_x'], metadata['center_y']
        if type(x) is not int or type(y) is not int or (x, y) != (
                (capture.region[0]+capture.region[2])//2, (capture.region[1]+capture.region[3])//2):
            raise ValueError(f'Inconsistent capture center: {capture.path}')
        phase = capture.path.relative_to(root).parts[0]
        key = (digest, raw_labels)  # Label bytes must match as well as image SHA256.
        duplicate = key in canonical
        if not duplicate:
            # Sorted input ordinal disambiguates even identical stems and request retries.
            filename = f'phase_{phase[-2:]}_frame_{capture.frame:06d}_L1_x{x}_y{y}_{index:06d}.png'
            canonical[key] = filename
            prepared.append((capture.path, label, filename))
        filename = canonical[key]
        records.append(dict(output_filename=filename, phase=phase, sequence_id=capture.sequence_id,
            source_frame=capture.frame, center_x=x, center_y=y, source_region_xyxy=json.dumps(capture.region),
            source_image_path=str(capture.path), source_label_path=str(label), image_sha256=digest,
            annotation_count=annotation_count, deduplicated=int(duplicate)))
    if not prepared:
        raise ValueError('No labeled main-sequence L1 captures found')
    output = new_output(output, [labels_root, *(root / phase for phase in PHASES)])
    for kind in ('images', 'labels'):
        (output / kind / 'all').mkdir(parents=True)
    for image, label, filename in prepared:
        shutil.copyfile(image, output / 'images/all' / filename)
        shutil.copyfile(label, output / 'labels/all' / (filename[:-4]+'.txt'))
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    write_yaml(output)
    summary = dict(source_captures=len(records), unique_samples=len(prepared),
                   deduplicated=len(records)-len(prepared), skipped_unreviewed_captures=len(missing))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', default='~/drone-captures')
    parser.add_argument('--labels-root', default='~/drone-captures/projected_l1_labels')
    parser.add_argument('--output', default='~/drone-captures/validation_yolo')
    parser.add_argument('--require-all-labels', action='store_true', help='Fail even for entirely unreviewed frames')
    args = parser.parse_args()
    build_dataset(args.capture_root, args.labels_root, args.output, args.require_all_labels)


if __name__ == '__main__':
    main()
