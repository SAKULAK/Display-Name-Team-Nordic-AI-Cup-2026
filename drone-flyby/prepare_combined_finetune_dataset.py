"""Copy Helsinki and human-labeled validation crops into a source-frame holdout split. No training."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

import yaml

from validation_dataset_utils import (check_class_order, file_sha256, new_output,
                                      resolved, validate_yolo, write_yaml)

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def helsinki_samples(root):
    config = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))
    check_class_order(config)
    samples = []
    for split in ('train', 'val'):
        entry = config.get(split)
        if not isinstance(entry, str):
            raise ValueError('Helsinki data.yaml must use directory-based train/val entries')
        directory = Path(entry)
        directory = directory.resolve() if directory.is_absolute() else (root / directory).resolve()
        if directory != (root / 'images' / split).resolve() or not directory.is_dir():
            raise ValueError(f'Helsinki {split} must be images/{split} under --helsinki-root')
        used_labels = set()
        images = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f'No Helsinki {split} images: {directory}')
        for index, image in enumerate(images):
            relative = image.relative_to(directory)
            label = root / 'labels' / split / relative.with_suffix('.txt')
            if label in used_labels:
                raise ValueError(f'Multiple images share one Helsinki label: {label}')
            used_labels.add(label)
            _, count = validate_yolo(label)
            samples.append(dict(dataset='helsinki', split=split,
                filename=f'helsinki_{split}_{index:07d}{image.suffix.lower()}', image=image, label=label,
                image_sha256=file_sha256(image), annotation_count=count, source_frames=[], provenance=[]))
        extras = set((root / 'labels' / split).rglob('*.txt'))-used_labels
        if extras:
            raise ValueError(f'Orphan Helsinki labels in {split}: {len(extras)}')
    return samples


def validation_samples(root, start, end):
    check_class_order(yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8')))
    groups = defaultdict(list)
    with (root / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        for record in csv.DictReader(stream):
            name = record['output_filename']
            if Path(name).name != name or '/' in name or '\\' in name or not name.endswith('.png'):
                raise ValueError(f'Unsafe validation filename: {name}')
            groups[name].append(record)
    if not groups:
        raise ValueError('Validation manifest is empty')
    actual_images = {p.name for p in (root / 'images/all').iterdir() if p.is_file()}
    actual_labels = {p.name for p in (root / 'labels/all').iterdir() if p.is_file()}
    if actual_images != set(groups) or actual_labels != {Path(n).stem+'.txt' for n in groups}:
        raise ValueError('Validation manifest/image/label file sets must match exactly')
    samples = []
    frame_splits = {}
    for name, records in sorted(groups.items()):
        frames = sorted({int(r['source_frame']) for r in records})
        if any(frame < 0 for frame in frames):
            raise ValueError('Negative validation source frame')
        splits = {'val' if start <= frame <= end else 'train' for frame in frames}
        if len(splits) != 1:
            raise ValueError(f'Deduplicated sample {name} spans holdout boundary: {frames}; '
                             'choose a holdout range keeping these identical samples on one side')
        split = splits.pop()
        for frame in frames:
            if frame in frame_splits and frame_splits[frame] != split:
                raise ValueError(f'Validation source frame leaks across splits: {frame}')
            frame_splits[frame] = split
        image, label = root / 'images/all' / name, root / 'labels/all' / (Path(name).stem+'.txt')
        _, count = validate_yolo(label)
        digest = file_sha256(image)
        if any(r['image_sha256'] != digest or int(r['annotation_count']) != count for r in records):
            raise ValueError(f'Validation sample disagrees with manifest: {name}')
        samples.append(dict(dataset='validation', split=split, filename='validation_'+name,
            image=image, label=label, image_sha256=digest, annotation_count=count,
            source_frames=frames, provenance=records))
    if not any(s['split'] == 'val' for s in samples):
        raise ValueError('Requested holdout range contains no validation samples')
    return samples


def prepare(helsinki_root, validation_root, output, start, end):
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError('Holdout must be an inclusive, nonnegative start..end range')
    helsinki_root, validation_root = resolved(helsinki_root), resolved(validation_root)
    samples = helsinki_samples(helsinki_root) + validation_samples(validation_root, start, end)
    # Also guard against identical image bytes across splits with different labels,
    # or across the two source datasets. Such conflicts cannot honor both assignments.
    hashes = {}
    for sample in samples:
        digest, split = sample['image_sha256'], sample['split']
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f'Identical image bytes would leak across train/val: {sample["image"]}')
        hashes[digest] = split
    output = new_output(output, [helsinki_root, validation_root])
    for kind in ('images', 'labels'):
        for split in ('train', 'val'):
            (output / kind / split).mkdir(parents=True)
    fields = ('output_filename', 'dataset', 'split', 'source_frames', 'source_image_path',
              'source_label_path', 'image_sha256', 'annotation_count', 'validation_provenance')
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            name, split = sample['filename'], sample['split']
            shutil.copyfile(sample['image'], output / 'images' / split / name)
            shutil.copyfile(sample['label'], output / 'labels' / split / (Path(name).stem+'.txt'))
            writer.writerow(dict(output_filename=name, dataset=sample['dataset'], split=split,
                source_frames=json.dumps(sample['source_frames']), source_image_path=str(sample['image']),
                source_label_path=str(sample['label']), image_sha256=sample['image_sha256'],
                annotation_count=sample['annotation_count'], validation_provenance=json.dumps(sample['provenance'])))
    write_yaml(output, split=True)
    summary = dict(helsinki_assignment='Preserved existing images/train and images/val; no resplitting',
                   validation_holdout_inclusive=[start, end],
                   counts=dict(Counter(s['dataset']+'_'+s['split'] for s in samples)))
    (output / 'preparation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helsinki-root', required=True, help='Existing YOLO directory with data.yaml and train/val folders')
    parser.add_argument('--validation-root', default='~/drone-captures/validation_yolo')
    parser.add_argument('--output', default='~/drone-captures/combined_finetune')
    parser.add_argument('--validation-holdout-start-frame', type=int, required=True)
    parser.add_argument('--validation-holdout-end-frame', type=int, required=True)
    args = parser.parse_args()
    prepare(args.helsinki_root, args.validation_root, args.output,
            args.validation_holdout_start_frame, args.validation_holdout_end_frame)


if __name__ == '__main__':
    main()
