"""Use --models checkpoint1.pt checkpoint2.pt --confidence 0.10 for matched comparison.

Delegates to the official-evaluator runner, including synchronized inference
latency and approximate FPS. All detector-sweep options remain available.
"""
from benchmark_detector import main

if __name__ == '__main__':
    raise SystemExit(main())
