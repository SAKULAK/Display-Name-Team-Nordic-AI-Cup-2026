"""User-run continuation of an incomplete local checkpoint; never used by preflight."""

import argparse
from pathlib import Path


def resume(checkpoint, data, run_dir, device='0', batch=4):
    checkpoint, data, run_dir = (Path(p).expanduser().resolve() for p in (checkpoint, data, run_dir))
    if not checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError('Both the local last.pt and prepared data.yaml must exist')
    from teammate_preflight import dataset_paths
    if not all(path.is_dir() for path in dataset_paths(data)):
        raise ValueError('Dataset paths do not resolve on this machine; rebuild or relocate data.yaml')
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    state = model.ckpt
    if not state or state.get('epoch', -1) < 0 or state.get('optimizer') is None:
        raise ValueError('Checkpoint has no resumable epoch/optimizer state. Start a fresh run from best.pt instead.')
    # Explicit string resume avoids an automatic fallback to fresh training.
    model.train(resume=str(checkpoint), data=str(data), device=device, batch=batch,
                workers=0, save_dir=str(run_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    resume(args.checkpoint, args.data, args.run_dir, args.device, args.batch)


if __name__ == '__main__':
    main()
