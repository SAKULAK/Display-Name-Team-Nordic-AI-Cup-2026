"""YOLO detection with the baseline camera policy and response protocol."""

import logging
import os
import math
from typing import Dict, List, Optional

import numpy as np

from dtos import (
    MAXIMUM_CENTER_DELTA_PIXELS,
    FULL_FRAME_CENTER,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import decode_view
from detector import get_detector
from tracker import track

logger = logging.getLogger(__name__)


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: report detections and pick the next camera position."""
    # The evaluator tells you when it ignored your last camera command. Reading
    # this beats wondering why the camera never moved.
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            'Camera command from frame %s was ignored: %s',
            feedback.frame,
            feedback.reason,
        )

    image = decode_view(request.view)

    # Never let a modelling error cost you the frame. An empty list still
    # scores the frame; an exception loses it and every detection in it.
    try:
        fresh = detect(image, request)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        fresh = []

    # Fuse this frame's detections with what earlier frames already found.
    # Even an empty ``fresh`` still lets already-tracked objects propagate
    # forward from wherever the camera was pointed before.
    annotations = track(request.sequence_id, request.frame, fresh)

    return DroneFlybyPredictResponseDto(
        # These two must come straight back from the request, unchanged.
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )


def detect(
    image: np.ndarray,
    request: DroneFlybyPredictRequestDto,
) -> List[DroneFlybyPredictionDto]:
    """Run the trained detector and return frame-global predictions."""
    return get_detector().detect(image, request)


### DUMMY CAMERA POLICY ###

# Where the sweep goes next, per sequence. The evaluator sends the camera's
# real position in every request, so this only needs to remember intent.
_sweep_direction: Dict[str, int] = {}


def choose_next_view(request):
    mode = os.environ.get('CAMERA_POLICY', 'hold_full').strip().lower()
    if mode == 'baseline_sweep':
        return baseline_sweep(request)
    if mode == 'l1_cycle':
        return l1_cycle(request)
    if mode != 'hold_full':
        raise ValueError(f'Unknown CAMERA_POLICY: {mode!r}')
    current, constraints = request.view, request.camera_constraints
    if current.resolution_level == 0:
        return None
    # The official L2 constraints exclude L0: return through L1 first.
    target = 0 if 0 in constraints.allowed_resolution_levels else 1
    if target not in constraints.allowed_resolution_levels:
        return None
    bounds = constraints.bounds_for_level(target)
    if bounds is None:
        return None
    if target == 0:
        x, y = FULL_FRAME_CENTER
    else:
        x = min(max(current.center_x, bounds.minimum_center_x), bounds.maximum_center_x)
        y = min(max(current.center_y, bounds.minimum_center_y), bounds.maximum_center_y)
    if not (bounds.minimum_center_x <= x <= bounds.maximum_center_x and
            bounds.minimum_center_y <= y <= bounds.maximum_center_y):
        return None
    exempt = target == 0 and constraints.full_view_reset_exempt_from_delta
    if not exempt and math.hypot(x-current.center_x, y-current.center_y) > constraints.maximum_center_delta:
        return None
    return RequestedViewDto(resolution_level=target, center_x=int(x), center_y=int(y))


def baseline_sweep(
    request: DroneFlybyPredictRequestDto,
) -> Optional[RequestedViewDto]:
    """Sweep sideways at the deepest zoom the camera can reach right now.

    Everything here is read from ``request.camera_constraints`` rather than
    hardcoded, which is the whole trick: honour the constraints you are handed
    and your commands cannot be rejected. Return ``None`` to hold position.
    """
    constraints = request.camera_constraints
    current = request.view
    allowed = [level for level in constraints.allowed_resolution_levels if level > 0]
    if not allowed:
        return None

    # Zoom in one step at a time; L0 cannot reach L2 directly.
    target_level = min(max(allowed), current.resolution_level + 1)
    bounds = constraints.bounds_for_level(target_level)
    if bounds is None:
        return None

    # Coming from the full view there is only one legal centre to start from.
    if current.resolution_level == 0:
        centre_x = (bounds.minimum_center_x + bounds.maximum_center_x) // 2
        centre_y = (bounds.minimum_center_y + bounds.maximum_center_y) // 2
        return RequestedViewDto(
            resolution_level=target_level,
            center_x=int(centre_x),
            center_y=int(centre_y),
        )

    direction = _sweep_direction.setdefault(request.sequence_id, 1)

    # Move as far as this response is allowed to, and no further. The limit
    # belongs to the level the camera is on now, not the one we are going to.
    limit = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS[
        current.resolution_level
    ]
    step = int(limit * 0.9)

    centre_x = current.center_x + direction * step
    if centre_x > bounds.maximum_center_x or centre_x < bounds.minimum_center_x:
        # Turn around at the edge and drop down a row.
        direction = -direction
        _sweep_direction[request.sequence_id] = direction
        centre_x = current.center_x + direction * step

    centre_y = current.center_y

    # Clamp into the legal window. int() matters: these fields are strict ints
    # on the evaluator, so a float here is a validation error.
    centre_x = int(min(max(centre_x, bounds.minimum_center_x), bounds.maximum_center_x))
    centre_y = int(min(max(centre_y, bounds.minimum_center_y), bounds.maximum_center_y))

    return RequestedViewDto(
        resolution_level=target_level,
        center_x=centre_x,
        center_y=centre_y,
    )


def l1_cycle(request: DroneFlybyPredictRequestDto) -> Optional[RequestedViewDto]:
    """Loop the camera around the frame at L1, covering all of it on repeat.

    The four L1 bound corners are exactly the centres of the four 1920x1080
    quadrants that tile the full 3840x2160 frame with no gaps and no overlap
    (L1's region size is exactly half the frame in each dimension). Visiting
    all four covers the whole frame; the four edge midpoints between them are
    just legal-sized hops to get there, since a straight quadrant-to-quadrant
    jump (1920px) exceeds the L1 move limit (1102px, see
    ``camera_constraints.maximum_center_delta``) and takes two hops anyway.

    Measured against the Helsinki annotations, objects cross the whole
    2160px frame at roughly 55-80px/frame, i.e. they are in the frame for
    about 30 frames. This 8-hop loop (4 quadrant dwells + 4 transit hops)
    keeps every point in the frame within one L1 dwell every 8 frames, well
    inside that window, so the tracker in ``tracker.py`` only ever has to
    bridge short gaps rather than a policy that permanently moves on.

    Deliberately stateless (recomputed from ``request.view`` every call, like
    ``baseline_sweep``): if a requested move is ever rejected, the next call
    just resumes from wherever the camera actually is instead of drifting out
    of sync with an internally remembered index.
    """
    constraints = request.camera_constraints
    current = request.view
    bounds = constraints.bounds_for_level(1)
    if bounds is None:
        return None
    min_x, max_x = bounds.minimum_center_x, bounds.maximum_center_x
    min_y, max_y = bounds.minimum_center_y, bounds.maximum_center_y
    mid_x, mid_y = (min_x + max_x) // 2, (min_y + max_y) // 2

    # Clockwise: TL -> top-mid -> TR -> right-mid -> BR -> bottom-mid -> BL -> left-mid -> TL
    waypoints = [
        (min_x, min_y), (mid_x, min_y),
        (max_x, min_y), (max_x, mid_y),
        (max_x, max_y), (mid_x, max_y),
        (min_x, max_y), (min_x, mid_y),
    ]

    if current.resolution_level == 0:
        if 1 not in constraints.allowed_resolution_levels:
            return None
        target_x, target_y = waypoints[0]
    else:
        # Find where we already are in the loop and advance one hop. Exact
        # match is expected since we only ever request waypoint coordinates.
        try:
            current_index = waypoints.index((current.center_x, current.center_y))
            target_x, target_y = waypoints[(current_index + 1) % len(waypoints)]
        except ValueError:
            # Off-loop (e.g. a stale/rejected command): head for the nearest
            # waypoint rather than the geometrically next one.
            target_x, target_y = min(
                waypoints,
                key=lambda point: math.hypot(point[0] - current.center_x, point[1] - current.center_y),
            )

    if math.hypot(target_x - current.center_x, target_y - current.center_y) > constraints.maximum_center_delta:
        return None
    return RequestedViewDto(resolution_level=1, center_x=int(target_x), center_y=int(target_y))
