"""Restart an isolated API for each setting and run the unmodified official evaluator."""
import argparse
import csv
import hashlib
import itertools
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

from detector import parse_bool

ROOT = Path(__file__).resolve().parent
FIELDS = ['model', 'sha256', 'confidence', 'iou', 'agnostic_nms', 'camera_policy',
          'max_det', 'map50', 'elapsed_seconds', 'timestamp', 'average_inference_ms',
          'approx_fps', 'calls', 'scene', 'realtime', 'status', 'log']


def parse_score(text):
    matches = re.findall(r'COCO mAP@0\.50:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not matches:
        raise ValueError('Final official mAP50 not found')
    return float(matches[-1])


def stop_server(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def experiment(model, confidence, iou, agnostic, policy, args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_dir = args.output.parent/'logs'/stamp
    log_dir.mkdir(parents=True)
    row = dict.fromkeys(FIELDS, '')
    row.update(model=str(model), confidence=confidence, iou=iou, agnostic_nms=agnostic,
               camera_policy=policy, max_det=args.max_det, timestamp=stamp,
               scene=args.scene, realtime=args.realtime, log=str(log_dir), status='failed')
    env = os.environ.copy()
    env.update(YOLO_WEIGHTS=str(model), YOLO_CONF=str(confidence), YOLO_IOU=str(iou),
               YOLO_AGNOSTIC_NMS=str(agnostic), YOLO_MAX_DET=str(args.max_det),
               CAMERA_POLICY=policy, YOLO_DEVICE=args.device, YOLO_LOG_EVERY='0',
               PYTHONUNBUFFERED='1')
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'ultralytics', 'numpy', 'opencv-python', 'faster-coco-eval')}
    (log_dir/'environment.json').write_text(json.dumps(dict(python=sys.version, packages=versions,
        configuration={k: v for k, v in env.items() if k.startswith('YOLO_') or k == 'CAMERA_POLICY'}), indent=2))
    started, server = time.perf_counter(), None
    try:
        row['sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        metrics = log_dir/'metrics.json'
        with (log_dir/'server.log').open('w', encoding='utf-8') as server_log:
            try:
                server = subprocess.Popen([sys.executable, str(ROOT/'benchmark_server.py'),
                    '--port', str(port), '--metrics', str(metrics.resolve())], cwd=ROOT, env=env,
                    stdout=server_log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
                deadline = time.monotonic()+args.startup_timeout
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('API exited before readiness; inspect server.log')
                    try:
                        with urlopen(f'http://127.0.0.1:{port}/api', timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('API startup timeout')
                    time.sleep(0.2)
                command = [sys.executable, str(ROOT/'local_evaluator.py'), '--url',
                           f'http://127.0.0.1:{port}/predict', '--scene', args.scene]
                if args.realtime:
                    command.append('--realtime')
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, errors='replace', timeout=args.timeout)
                (log_dir/'evaluator.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
                if result.returncode:
                    raise RuntimeError(f'Evaluator exit code {result.returncode}')
                row['map50'] = parse_score(result.stdout)
                row['status'] = 'ok'
                for label in ('frames unanswered', 'timeouts', 'http errors', 'invalid responses', 'camera moves refused'):
                    match = re.search(re.escape(label) + r'\s+(\d+)', result.stdout)
                    if match and int(match[1]):
                        row['status'] = f'failed: {label}={match[1]}'
            finally:
                if server is not None:
                    stop_server(server)
        server_text = (log_dir/'server.log').read_text(encoding='utf-8')
        if 'Detector failed' in server_text or 'Traceback' in server_text:
            row['status'] = 'failed: API errors; inspect server.log'
        if metrics.exists():
            timing = json.loads(metrics.read_text())
            row['calls'] = timing['calls']
            if timing['calls'] and timing['seconds']:
                row['average_inference_ms'] = 1000*timing['seconds']/timing['calls']
                row['approx_fps'] = timing['calls']/timing['seconds']
        else:
            row['status'] = 'failed: missing latency metrics'
    except Exception as exc:
        row['status'] = f'failed: {exc}'
    row['elapsed_seconds'] = time.perf_counter()-started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--confidence', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.25])
    parser.add_argument('--iou', type=float, nargs='+', default=[0.5])
    parser.add_argument('--agnostic-nms', type=parse_bool, nargs='+', default=[True])
    parser.add_argument('--camera-policies', choices=['hold_full', 'baseline_sweep'], nargs='+', default=['hold_full'])
    parser.add_argument('--max-det', type=int, default=500)
    parser.add_argument('--device', default='0')
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/detector_sweep.csv')
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with args.output.open(newline='') as handle:
            if next(csv.reader(handle), []) != FIELDS:
                parser.error('Existing CSV schema differs; choose a new --output')
    failures = 0
    for model, confidence, iou, agnostic, policy in itertools.product(
            args.models, args.confidence, args.iou, args.agnostic_nms, args.camera_policies):
        row = experiment(model.resolve(), confidence, iou, agnostic, policy, args)
        exists = args.output.exists()
        with args.output.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row), flush=True)
        failures += row['status'] != 'ok'
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
