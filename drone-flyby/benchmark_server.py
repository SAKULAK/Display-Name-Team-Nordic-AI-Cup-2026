"""Private benchmark server: warm up before readiness, persist timing on shutdown."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uvicorn

from detector import get_detector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    args = parser.parse_args()
    detector = get_detector()
    request = SimpleNamespace(view=SimpleNamespace(source_region_xyxy=[0, 0, 3840, 2160]),
                              original_width=3840, original_height=2160)
    for _ in range(3):
        detector.detect(np.zeros((540, 960, 3), dtype=np.uint8), request)
    detector.number_of_calls = 0
    detector.total_inference_seconds = 0.0
    # A lifespan hook runs on graceful shutdown, without changing the public API.
    from contextlib import asynccontextmanager
    from api import app

    @asynccontextmanager
    async def lifespan(app):
        yield
        args.metrics.write_text(json.dumps(dict(calls=detector.number_of_calls,
            seconds=detector.total_inference_seconds)), encoding='utf-8')

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
