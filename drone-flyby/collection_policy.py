"""Deterministic collection routes, checked against each request's constraints."""

import logging
import math
import os
from threading import Lock

from dtos import FULL_FRAME_CENTER, RequestedViewDto

logger = logging.getLogger(__name__)
L1_PATH = ((960, 540), (1920, 540), (2880, 540), (2880, 1080),
           (2880, 1620), (1920, 1620), (960, 1620), (960, 1080), (1920, 1080))
L2_INDICES = (tuple((x, 0) for x in range(7)) +
              tuple((x, 1) for x in range(6, 0, -1)) +
              tuple((x, 2) for x in range(1, 7)) +
              tuple((x, 3) for x in range(6, -1, -1)) + ((0, 2), (0, 1)))
L2_PATH = tuple((480+480*x, 270+540*y) for x, y in L2_INDICES)


def nearest(path, position):
    return min(range(len(path)), key=lambda i: (math.dist(path[i], position), i))


def clamp_to_bounds(request, level, position):
    bounds = request.camera_constraints.bounds_for_level(level)
    if bounds is None:
        return position  # Validation below will hold and log the missing bounds.
    return (min(max(position[0], bounds.minimum_center_x), bounds.maximum_center_x),
            min(max(position[1], bounds.minimum_center_y), bounds.maximum_center_y))


def legal_move(request, level, position):
    current, constraints = request.view, request.camera_constraints
    x, y = map(int, position)
    bounds = constraints.bounds_for_level(level)
    reason = None
    if level not in (0, 1, 2) or abs(current.resolution_level-level) > 1:
        reason = 'transition'
    elif level not in constraints.allowed_resolution_levels:
        reason = 'level_not_allowed'
    elif bounds is None:
        reason = 'missing_bounds'
    elif not (bounds.minimum_center_x <= x <= bounds.maximum_center_x and
              bounds.minimum_center_y <= y <= bounds.maximum_center_y):
        reason = 'outside_bounds'
    elif not (level == 0 and constraints.full_view_reset_exempt_from_delta) and not (
            math.dist((current.center_x, current.center_y), (x, y)) <= constraints.maximum_center_delta):
        reason = 'maximum_delta'
    if reason:
        logger.info('COLLECT frame=%s hold reason=%s desired=L%s@(%s,%s)',
                    request.frame, reason, level, x, y)
        return None
    return RequestedViewDto(resolution_level=level, center_x=x, center_y=y)


class CollectionPolicy:
    def __init__(self):
        self.lock = Lock()
        self.context = None
        self.last_key = None
        self.last_command = None
        self.bridge = None

    def choose(self, request):
        level = int(os.environ.get('COLLECT_LEVEL', '0'))
        if level not in (0, 1, 2):
            raise ValueError('COLLECT_LEVEL must be 0, 1, or 2')
        phase = int(os.environ.get('COLLECT_PHASE', '0')) % (28 if level == 2 else 9)
        with self.lock:
            context = (request.sequence_id, level, phase)
            key = (request.frame, request.request_id)
            if context != self.context or (self.last_key is not None and
                    request.frame <= self.last_key[0] and key != self.last_key):
                self.context, self.last_key, self.bridge = context, None, None
            if key == self.last_key:
                return self.last_command
            current = request.view.resolution_level
            position = (request.view.center_x, request.view.center_y)
            target, desired = level, position
            if level == 0:
                if current == 0:
                    target = None
                elif current == 2:
                    target, desired = 1, clamp_to_bounds(request, 1, position)
                else:
                    desired = FULL_FRAME_CENTER
            elif level == 1:
                desired = L1_PATH[phase] if current == 0 else (
                    L1_PATH[(nearest(L1_PATH, position)+1) % 9] if current == 1 else
                    clamp_to_bounds(request, 1, position))
            elif current == 0:
                target, desired = 1, clamp_to_bounds(request, 1, L2_PATH[phase])
            elif current == 1:
                desired = L2_PATH[phase] if self.bridge == position else L2_PATH[nearest(L2_PATH, position)]
            else:
                desired = L2_PATH[(nearest(L2_PATH, position)+1) % 28]
            command = legal_move(request, target, desired) if target is not None else None
            if level == 2 and current == 0:
                self.bridge = (command.center_x, command.center_y) if command else None
            elif current != 1 or command is not None:
                self.bridge = None
            logger.info('COLLECT frame=%s current=L%s@(%s,%s) target=%s phase=%s',
                        request.frame, current, *position,
                        f'L{command.resolution_level}@({command.center_x},{command.center_y})'
                        if command else 'hold', phase)
            self.last_key, self.last_command = key, command
            return command


collection_policy = CollectionPolicy()
