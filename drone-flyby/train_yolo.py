"""Fine-tune a pretrained YOLO detector on Helsinki."""

import argparse
from pathlib import Path

from prepare_yolo import DEFAULT_OUTPUT, ROOT, convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_OUTPUT / 'data.yaml')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default=None, help='e.g. 0 or cpu; default auto')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', '--run-name', dest='name', default='helsinki')
    for option, default in [('degrees', 180), ('flipud', 0.5), ('fliplr', 0.5),
                            ('scale', 0.5), ('translate', 0.1), ('mosaic', 1.0)]:
        parser.add_argument('--' + option, type=float, default=default)
    args = parser.parse_args()
    if not 0 <= args.degrees <= 180:
        parser.error('--degrees must be in [0, 180]')
    for option in ('flipud', 'fliplr', 'scale', 'translate', 'mosaic'):
        if not 0 <= getattr(args, option) <= 1:
            parser.error(f'--{option} must be in [0, 1]')
    if not args.data.exists() and args.data.resolve() == DEFAULT_OUTPUT / 'data.yaml':
        convert()
    if not args.data.is_file():
        parser.error(f'Dataset YAML does not exist: {args.data}')
    from ultralytics import YOLO
    import yaml
    from dtos import OBJECT_CLASSES
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    names = config['names']
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    if tuple(ordered) != OBJECT_CLASSES:
        raise ValueError('Dataset names must exactly match dtos.OBJECT_CLASSES')
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                project=str(ROOT / 'runs' / 'detect'), name=args.name,
                seed=42, deterministic=True, exist_ok=False,
                degrees=args.degrees, flipud=args.flipud, fliplr=args.fliplr,
                scale=args.scale, translate=args.translate, mosaic=args.mosaic)
    print(f'Trained weights: {model.trainer.best}')


if __name__ == '__main__':
    main()
