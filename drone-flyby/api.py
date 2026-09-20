"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.
"""

import datetime
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from detector import get_detector
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9054

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_detector() loads the model and runs a dummy inference (see
    # detector.py). Forcing that here, before uvicorn accepts connections,
    # is what actually pays the cold-start cost before an attempt instead of
    # on frame 0 -- calling it lazily from inside predict() does not help,
    # since nothing calls it until the first real request arrives anyway.
    logger.info('Warming up detector...')
    started = time.perf_counter()
    get_detector()
    logger.info('Detector warmed up in %.2fs', time.perf_counter() - started)
    yield


app = FastAPI(lifespan=lifespan)
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
