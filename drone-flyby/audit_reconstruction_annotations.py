"""Read-only statistics and review flags for reconstruction annotations."""

import argparse
from collections import Counter, defaultdict
import json
from statistics import median, quantiles

from dtos import OBJECT_CLASSES
from project_reconstruction_labels import load_annotations
from validation_dataset_utils import resolved, warn_absent_classes


def audit(frames):
    by_class = defaultdict(list)
    boundaries, duplicates, multiple = [], [], []
    for frame, annotations in sorted(frames.items()):
        counts = Counter(a['object_id'] for a in annotations)
        for name, count in sorted(counts.items()):
            if count > 1:
                multiple.append(dict(frame=frame, object_id=name, count=count))
        boxes = defaultdict(list)
        for index, annotation in enumerate(annotations):
            box = annotation['bbox']
            x1, y1, x2, y2 = box
            by_class[annotation['object_id']].append((frame, index, x2-x1, y2-y1, (x2-x1)*(y2-y1)))
            boxes[tuple(box)].append(annotation['object_id'])
            if x1 == 0 or y1 == 0 or x2 == 1920 or y2 == 1080:
                boundaries.append(dict(frame=frame, annotation_index=index, object_id=annotation['object_id'], bbox=box))
        for box, names in boxes.items():
            if len(names) > 1:
                duplicates.append(dict(frame=frame, bbox=list(box), object_ids=names, count=len(names)))
    classes, outliers = {}, []
    for name in OBJECT_CLASSES:
        records = by_class[name]
        areas = [r[4] for r in records]
        classes[name] = dict(boxes=len(records), unique_frames=len({r[0] for r in records}),
            median_width=median([r[2] for r in records]) if records else None,
            median_height=median([r[3] for r in records]) if records else None,
            median_area=median(areas) if records else None,
            min_area=min(areas, default=None), max_area=max(areas, default=None))
        if len(records) >= 4:
            for column, metric in ((2, 'width'), (3, 'height'), (4, 'area')):
                q1, _, q3 = quantiles([r[column] for r in records], n=4, method='inclusive')
                low, high = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
                for record in records:
                    if not low <= record[column] <= high:
                        outliers.append(dict(frame=record[0], annotation_index=record[1],
                                             object_id=name, metric=metric, value=record[column]))
    return dict(total_reviewed_frames=len(frames), total_boxes=sum(len(a) for a in frames.values()),
                empty_frames=sum(not a for a in frames.values()), per_class=classes,
                boundary_box_count=len(boundaries), boundary_boxes=boundaries,
                suspicious_exact_duplicate_boxes=duplicates, frames_with_multiple_instances=multiple,
                statistical_outliers=outliers, outlier_rule='1.5 IQR per class/metric, inclusive quartiles, n>=4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    args = parser.parse_args()
    report = audit(load_annotations(resolved(args.annotations)))
    print(json.dumps(report, indent=2))
    warn_absent_classes({name: values['boxes'] for name, values in report['per_class'].items()})


if __name__ == '__main__':
    main()
