"""Target-only L1 confirmation between authoritative L0 refreshes.

Boxes and motion use frame-global normalized coordinates. Velocity is measured
per source-frame number, not per received request, so skipped frames count.
Configuration and the scoring formula are deliberately local to this module.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from statistics import median
from threading import Lock
from typing import Optional

from dtos import DroneFlybyPredictionDto, FULL_FRAME_CENTER, RequestedViewDto
from utils import clip_bbox_to_frame

logger = logging.getLogger(__name__)

CLASS_PRIORITIES = {
    'hangar': 3.0, 'medium_launcher': 3.0, 'medium_plane': 3.0, 'mine_roller': 3.0,
    'ta-ta': 1.5, 'small_launcher': 1.5,
}
FOCUS_CLASSES = frozenset({'small_launcher', 'ta-ta', 'mine_roller',
                           'hangar', 'medium_launcher', 'medium_plane'})


@dataclass(frozen=True)
class FocusConfig:
    min_l0_frames: int = 3
    max_focus_conf: float = 0.45
    memory_ttl: int = 2  # Inclusive: age >= 3 expires by default.
    confidence_decay: float = 0.95
    confidence_grace_frames: int = 1
    cooldown_refreshes: int = 4
    class_weight: float = 1.0
    uncertainty_weight: float = 0.8
    small_object_weight: float = 0.6
    stale_focus_weight: float = 0.3
    small_object_scale: float = 0.06  # sqrt(normalized area)
    stale_horizon: int = 12          # L0 refreshes
    minimum_focus_score: float = 1.2
    match_distance_per_frame: float = 0.08
    maximum_match_distance: float = 0.25
    motion_min_matches: int = 3
    motion_min_confidence: float = 0.2
    duplicate_iou: float = 0.35
    duplicate_center_fraction: float = 0.25
    cooldown_region_distance: float = 0.035
    outcome_match_distance: float = 0.035
    confirmed_frames: int = 12
    confirmation_confidence: float = 0.70
    confirmation_distance: float = 0.035
    confirmed_penalty: float = 10.0
    velocity_min_iou: float = 0.2
    velocity_match_margin: float = 0.1
    incidental_min_iou: float = 0.65
    incidental_center_fraction: float = 0.25
    incidental_match_margin: float = 0.1
    focus_classes: frozenset[str] = FOCUS_CLASSES
    min_focus_conf: float = 0.0

    @classmethod
    def from_env(cls):
        return cls(
            focus_classes=(frozenset(name.strip() for name in os.environ['FOCUS_CLASSES'].split(',')
                                     if name.strip()) if 'FOCUS_CLASSES' in os.environ else FOCUS_CLASSES),
            min_focus_conf=float(os.environ.get('FOCUS_MIN_CONF', '0.0')),
            min_l0_frames=int(os.environ.get('FOCUS_MIN_L0_FRAMES', '3')),
            max_focus_conf=float(os.environ.get('FOCUS_MAX_CONF', '0.45')),
            minimum_focus_score=float(os.environ.get('FOCUS_MIN_SCORE', '1.2')),
            memory_ttl=int(os.environ.get('FOCUS_MEMORY_TTL', '2')),
            confidence_decay=float(os.environ.get('FOCUS_CONFIDENCE_DECAY', '0.95')),
            confidence_grace_frames=int(os.environ.get('FOCUS_CONFIDENCE_GRACE_FRAMES', '1')),
            confirmed_frames=int(os.environ.get('FOCUS_CONFIRMED_FRAMES', '12')),
            confirmation_confidence=float(os.environ.get('FOCUS_CONFIRM_CONF', '0.70')),
            confirmed_penalty=float(os.environ.get('FOCUS_CONFIRMED_PENALTY', '10')),
        )

    def __post_init__(self):
        if not 0 <= self.max_focus_conf <= 1:
            raise ValueError('FOCUS_MAX_CONF must be in [0, 1]')
        if not 0 <= self.min_focus_conf <= 1:
            raise ValueError('FOCUS_MIN_CONF must be in [0, 1]')
        if self.min_focus_conf > self.max_focus_conf:
            raise ValueError('FOCUS_MIN_CONF must not exceed FOCUS_MAX_CONF')
        if type(self.confirmed_frames) is not int or self.confirmed_frames < 1:
            raise ValueError('FOCUS_CONFIRMED_FRAMES must be a positive integer')
        if not 0 <= self.confirmation_confidence <= 1:
            raise ValueError('FOCUS_CONFIRM_CONF must be in [0, 1]')
        if not math.isfinite(self.confirmed_penalty) or self.confirmed_penalty < 0:
            raise ValueError('FOCUS_CONFIRMED_PENALTY must be finite and nonnegative')
        if type(self.min_l0_frames) is not int or self.min_l0_frames < 1:
            raise ValueError('FOCUS_MIN_L0_FRAMES must be a positive integer')
        if not math.isfinite(self.minimum_focus_score) or self.minimum_focus_score < 0:
            raise ValueError('FOCUS_MIN_SCORE must be finite and nonnegative')
        if type(self.confidence_grace_frames) is not int or self.confidence_grace_frames < 0:
            raise ValueError('Confidence grace frames must be a nonnegative integer')
        if type(self.memory_ttl) is not int:
            raise ValueError('Memory TTL must be an integer')
        if self.memory_ttl < 1 or self.cooldown_refreshes < 1 or self.motion_min_matches < 1:
            raise ValueError('TTL, cooldown and minimum matches must be positive')
        if not 0 < self.confidence_decay <= 1 or self.small_object_scale <= 0 or self.stale_horizon < 1:
            raise ValueError('Invalid decay or scoring scales')


def center(box):
    return (box[0]+box[2])/2, (box[1]+box[3])/2


def intersection(a, b):
    return max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    common = intersection(a, b)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-common
    return common/union if union > 0 else 0.0


def shifted(box, dx, dy, age):
    return clip_bbox_to_frame((box[0]+dx*age, box[1]+dy*age,
                               box[2]+dx*age, box[3]+dy*age))


def match_detections(previous, current, distance_gate):
    """Deterministic greedy one-to-one same-class match, using distance and IoU."""
    candidates = []
    for old_index, old in enumerate(previous):
        for new_index, new in enumerate(current):
            if old.object_id != new.object_id:
                continue
            distance = math.dist(center(old.bbox), center(new.bbox))
            overlap = iou(old.bbox, new.bbox)
            if distance <= distance_gate:
                cost = distance/max(distance_gate, 1e-9) + 0.25*(1-overlap)
                candidates.append((cost, old_index, new_index))
    used_old, used_new, matches = set(), set(), []
    for _, old_index, new_index in sorted(candidates):
        if old_index not in used_old and new_index not in used_new:
            matches.append((old_index, new_index))
            used_old.add(old_index)
            used_new.add(new_index)
    return matches


def unambiguous_pairs(candidates, margin):
    """Accept only reciprocal best matches separated from both runners-up."""
    by_old, by_new = {}, {}
    for cost, old, new in candidates:
        by_old.setdefault(old, []).append((cost, new))
        by_new.setdefault(new, []).append((cost, old))
    for values in list(by_old.values())+list(by_new.values()):
        values.sort()
    result = []
    for old, values in sorted(by_old.items()):
        cost, new = values[0]
        if len(values) > 1 and values[1][0]-cost < margin:
            continue
        reverse = by_new[new]
        if reverse[0][1] != old or (len(reverse) > 1 and reverse[1][0]-cost < margin):
            continue
        result.append((old, new))
    return result


def legal_command(request, level, x, y):
    """Only emit L0/L1 commands allowed by the received bounds and delta."""
    constraints, view = request.camera_constraints, request.view
    if level not in (0, 1) or level not in constraints.allowed_resolution_levels:
        return None
    bounds = constraints.bounds_for_level(level)
    if bounds is None:
        return None
    if level == 0:
        x, y = FULL_FRAME_CENTER
    else:
        x = int(min(max(round(x), bounds.minimum_center_x), bounds.maximum_center_x))
        y = int(min(max(round(y), bounds.minimum_center_y), bounds.maximum_center_y))
    if not (bounds.minimum_center_x <= x <= bounds.maximum_center_x and
            bounds.minimum_center_y <= y <= bounds.maximum_center_y):
        return None
    exempt = level == 0 and constraints.full_view_reset_exempt_from_delta
    if not exempt and math.hypot(x-view.center_x, y-view.center_y) > constraints.maximum_center_delta:
        return None
    return RequestedViewDto(resolution_level=level, center_x=int(x), center_y=int(y))


def return_to_full(request):
    if request.view.resolution_level == 0:
        return None
    return legal_command(request, 0, *FULL_FRAME_CENTER)


@dataclass
class Track:
    track_id: int
    detection: DroneFlybyPredictionDto
    observed_frame: int


@dataclass
class FocusRecord:
    track_id: int
    object_id: str
    bbox: list
    frame: int
    refresh: int


@dataclass(frozen=True)
class FocusSnapshot:
    frame: int
    annotations: tuple
    target_index: int
    edge_velocities: tuple


@dataclass
class Confirmation:
    object_id: str
    bbox: list
    position_frame: int
    confirmed_frame: int
    confidence: float


@dataclass
class SequenceState:
    sequence_id: str
    tracks: list = field(default_factory=list)
    last_full_detections: list = field(default_factory=list)
    previous_full_detections: list = field(default_factory=list)
    last_full_frame: Optional[int] = None
    previous_full_frame: Optional[int] = None
    dx: float = 0.0
    dy: float = 0.0
    motion_frame_gap: int = 0
    focus_target: Optional[Track] = None
    focus_snapshot: Optional[FocusSnapshot] = None
    snapshot_used: bool = False
    confirmations: list = field(default_factory=list)
    performing_focus_frame: bool = False
    last_focused_class: Optional[str] = None
    last_focused_region: Optional[list] = None
    cooldowns: list = field(default_factory=list)
    full_refreshes: int = 0
    l0_frames_since_focus: int = 0
    next_track_id: int = 0
    last_frame: Optional[int] = None
    last_request_id: Optional[str] = None
    last_decision: object = None
    focus_operations: int = 0
    remembered_emitted: int = 0


@dataclass
class FocusDecision:
    annotations: list
    requested_view: Optional[RequestedViewDto]
    memory_emitted: int = 0
    expired: int = 0
    replaced: int = 0
    cleared: int = 0


class FocusPolicy:
    def __init__(self, config=None):
        self._config = config
        # Exactly one active sequence. A change of ID discards all old state.
        self.states = {}
        self.active_sequence = None
        self.lock = Lock()

    @property
    def config(self):
        # Read once on first focus use, so invalid focus settings cannot affect hold_full.
        if self._config is None:
            self._config = FocusConfig.from_env()
        return self._config

    def _state(self, request):
        if request.sequence_id != self.active_sequence:
            self.states.clear()
            self.active_sequence = request.sequence_id
        state = self.states.get(request.sequence_id)
        if state is None or (state.last_frame is not None and request.frame <= state.last_frame and
                             (request.frame, request.request_id) != (state.last_frame, state.last_request_id)):
            state = SequenceState(request.sequence_id)
            self.states[request.sequence_id] = state
        return state

    def begin_request(self, request):
        """Reset sequence state even when the subsequent detector call fails."""
        with self.lock:
            state = self._state(request)
            state.performing_focus_frame = request.view.resolution_level == 1
            if state.performing_focus_frame:
                # Even a failed detector on L1 begins a new full-view waiting period.
                state.l0_frames_since_focus = 0

    def process(self, request, live):
        with self.lock:
            state = self._state(request)
            if (state.last_frame, state.last_request_id) == (request.frame, request.request_id):
                return state.last_decision
            state.performing_focus_frame = request.view.resolution_level == 1
            if request.view.resolution_level == 0:
                decision = self._full_frame(state, request, list(live)[:500])
            elif request.view.resolution_level == 1:
                decision = self._focus_frame(state, request, list(live)[:500])
            else:
                # No L2 implementation or commands; unexpected input remains stateless.
                state.tracks.clear()
                state.focus_target = None
                state.focus_snapshot = None
                logger.warning('frame=%s unexpected L%s in focus_l1', request.frame, request.view.resolution_level)
                decision = FocusDecision(list(live)[:500], return_to_full(request))
            state.last_frame, state.last_request_id = request.frame, request.request_id
            state.last_decision = decision
            return decision

    def _motion(self, state, live, frame):
        state.dx = state.dy = 0.0
        state.motion_frame_gap = 0
        if state.last_full_frame is None:
            return
        gap = frame-state.last_full_frame
        if gap <= 0:
            return
        state.motion_frame_gap = gap
        old = [d for d in state.last_full_detections if d.confidence >= self.config.motion_min_confidence]
        new = [d for d in live if d.confidence >= self.config.motion_min_confidence]
        gate = min(self.config.maximum_match_distance, self.config.match_distance_per_frame*gap)
        pairs = match_detections(old, new, gate)
        if len(pairs) >= self.config.motion_min_matches:
            state.dx = median((center(new[j].bbox)[0]-center(old[i].bbox)[0])/gap for i, j in pairs)
            state.dy = median((center(new[j].bbox)[1]-center(old[i].bbox)[1])/gap for i, j in pairs)

    def _project(self, track, state, frame):
        age = frame-track.observed_frame
        if age < 0 or age > self.config.memory_ttl:
            return None
        box = shifted(track.detection.bbox, state.dx, state.dy, age)
        if box is None:
            return None
        return DroneFlybyPredictionDto(object_id=track.detection.object_id, bbox=list(box),
            confidence=float(track.detection.confidence*self.config.confidence_decay**
                             max(0, age-self.config.confidence_grace_frames)))

    def _individual_velocities(self, state, live, frame):
        velocities = [None]*len(live)
        if state.last_full_frame is None or frame <= state.last_full_frame:
            return tuple(velocities)
        gap = frame-state.last_full_frame
        gate = min(self.config.maximum_match_distance, self.config.match_distance_per_frame*gap)
        candidates = []
        for i, previous in enumerate(state.last_full_detections):
            expected = shifted(previous.bbox, state.dx, state.dy, gap)
            if expected is None:
                continue
            for j, current in enumerate(live):
                if previous.object_id != current.object_id:
                    continue
                overlap = iou(expected, current.bbox)
                distance = math.dist(center(expected), center(current.bbox))
                ratios = [(current.bbox[k+2]-current.bbox[k]) /
                          (previous.bbox[k+2]-previous.bbox[k]) for k in (0, 1)]
                if (overlap >= self.config.velocity_min_iou and distance <= gate and
                        all(0.5 <= r <= 2.0 for r in ratios)):
                    candidates.append((1-overlap+distance/max(gate, 1e-9), i, j))
        for i, j in unambiguous_pairs(candidates, self.config.velocity_match_margin):
            velocity = tuple((b-a)/gap for a, b in zip(state.last_full_detections[i].bbox, live[j].bbox))
            next_box = [b+v for b, v in zip(live[j].bbox, velocity)]
            if next_box[0] < next_box[2] and next_box[1] < next_box[3]:
                velocities[j] = velocity
        return tuple(velocities)

    def _live_tracks(self, state, live, frame):
        old, detections = [], []
        for track in state.tracks:
            projected = self._project(track, state, frame)
            if projected is not None:
                old.append(track)
                detections.append(projected)
        pairs = match_detections(detections, live, self.config.match_distance_per_frame)
        identifiers = {j: old[i].track_id for i, j in pairs}
        result = []
        for index, detection in enumerate(live):
            if index not in identifiers:
                identifiers[index] = state.next_track_id
                state.next_track_id += 1
            result.append(Track(identifiers[index], detection, frame))
        return result

    def _focus_age(self, state, track, frame):
        ages = []
        for record in state.cooldowns:
            box = shifted(record.bbox, state.dx, state.dy, frame-record.frame)
            same_region = (box is not None and record.object_id == track.detection.object_id and
                           math.dist(center(box), center(track.detection.bbox)) <= self.config.cooldown_region_distance)
            if record.track_id == track.track_id or same_region:
                ages.append(state.full_refreshes-record.refresh)
        return min(ages) if ages else None

    def focus_score(self, detection, focus_age):
        # score = class_weight*priority + uncertainty_weight*(1-confidence)
        #       + small_weight*max(0,1-sqrt(area)/small_object_scale)
        #       + stale_weight*min(refreshes_since_focus/stale_horizon,1).
        # Never-focused targets get the full stale bonus; cooldown is a hard gate.
        x1, y1, x2, y2 = detection.bbox
        small = max(0.0, 1-math.sqrt((x2-x1)*(y2-y1))/self.config.small_object_scale)
        stale = 1.0 if focus_age is None else min(1.0, focus_age/self.config.stale_horizon)
        return (self.config.class_weight*CLASS_PRIORITIES.get(detection.object_id, 0.0) +
                self.config.uncertainty_weight*(1-detection.confidence) +
                self.config.small_object_weight*small + self.config.stale_focus_weight*stale)

    def _full_frame(self, state, request, live):
        # This refresh is authoritative, including when the previous focus was refused.
        state.focus_snapshot = None
        state.snapshot_used = False
        self._motion(state, live, request.frame)
        edge_velocities = self._individual_velocities(state, live, request.frame)
        self._advance_confirmations(state, live, request.frame)
        expired = sum(self._project(t, state, request.frame) is None for t in state.tracks)
        old_ids = {t.track_id for t in state.tracks}
        new_tracks = self._live_tracks(state, live, request.frame)
        replaced = sum(t.track_id in old_ids for t in new_tracks)
        cleared = len(state.tracks)-expired-replaced
        state.tracks = new_tracks
        state.previous_full_detections = state.last_full_detections
        state.previous_full_frame = state.last_full_frame
        state.last_full_detections = live
        state.last_full_frame = request.frame
        state.full_refreshes += 1
        state.l0_frames_since_focus += int(bool(live))
        state.cooldowns = [r for r in state.cooldowns
                           if state.full_refreshes-r.refresh <= max(self.config.stale_horizon, self.config.cooldown_refreshes)]
        state.focus_target = None
        candidates = []
        eligible_scores = []
        confirmed_count = 0
        class_filtered = confidence_filtered = persistence_filtered = 0
        for index, track in enumerate(state.tracks):
            if track.detection.object_id not in self.config.focus_classes:
                class_filtered += 1
                continue
            if not self.config.min_focus_conf <= track.detection.confidence <= self.config.max_focus_conf:
                confidence_filtered += 1
                continue
            # Reliable matching uses only the immediately previous authoritative L0,
            # with global motion and the actual source-frame gap.
            if track.detection.confidence < 0.15 and edge_velocities[index] is None:
                persistence_filtered += 1
                continue
            age = self._focus_age(state, track, request.frame)
            if age is not None and age <= self.config.cooldown_refreshes:
                continue
            score = self.focus_score(track.detection, age)
            if self._is_confirmed(state, track.detection):
                score -= self.config.confirmed_penalty
                confirmed_count += 1
            eligible_scores.append(score)
            if score < self.config.minimum_focus_score:
                continue
            x, y = center(track.detection.bbox)
            command = legal_command(request, 1, x*request.original_width, y*request.original_height)
            if command is not None:
                candidates.append((score, track.track_id, track, command))
        command, score = None, 0.0
        if state.l0_frames_since_focus < self.config.min_l0_frames:
            skip = 'minimum_l0_frames'
        elif candidates:
            score, _, state.focus_target, command = min(candidates, key=lambda c: (-c[0], c[1]))
            skip = 'none'
        elif not live:
            skip = 'no_detections'
        elif not eligible_scores:
            skip = ('class_filter' if class_filtered == len(live) else
                    'confidence_filter' if class_filtered+confidence_filtered == len(live) else
                    'persistence_filter' if persistence_filtered else 'cooldown')
        elif max(eligible_scores) < self.config.minimum_focus_score:
            skip = 'no_target_above_threshold'
        else:
            skip = 'camera_constraints'
        target = state.focus_target.detection if state.focus_target else None
        if command is not None:
            target_index = next(i for i, t in enumerate(state.tracks) if t.track_id == state.focus_target.track_id)
            state.focus_snapshot = FocusSnapshot(request.frame,
                tuple(d.model_copy(deep=True) for d in live), target_index, edge_velocities)
            state.focus_target = Track(state.focus_target.track_id,
                state.focus_snapshot.annotations[target_index], request.frame)
        logger.info('frame=%s L0 detections=%s focus=%s conf=%.3f center=%s score=%.3f '
                    'memory=0 expired=%s replaced=%s cleared=%s dx=%.3f dy=%.3f '
                    'l0_frames=%s min_l0_frames=%s skip=%s threshold=%.3f best_score=%.3f confirmed_penalized=%s '
                    'class_filtered=%s confidence_filtered=%s max_focus_conf=%.3f '
                    'min_focus_conf=%.3f focus_classes=%s',
                    request.frame, len(live), target.object_id if target else 'none',
                    target.confidence if target else 0.0,
                    (command.center_x, command.center_y) if command else None, score,
                    expired, replaced, cleared, state.dx*request.original_width, state.dy*request.original_height,
                    state.l0_frames_since_focus, self.config.min_l0_frames, skip,
                    self.config.minimum_focus_score, max(eligible_scores, default=0.0), confirmed_count,
                    class_filtered, confidence_filtered, self.config.max_focus_conf,
                    self.config.min_focus_conf, ','.join(sorted(self.config.focus_classes)))
        return FocusDecision(live, command, expired=expired, replaced=replaced, cleared=cleared)

    def _duplicate(self, a, b):
        if a.object_id != b.object_id:
            return False
        diagonal = min(math.hypot(d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1]) for d in (a, b))
        return (iou(a.bbox, b.bbox) >= self.config.duplicate_iou or
                math.dist(center(a.bbox), center(b.bbox)) <= max(0.002, diagonal*self.config.duplicate_center_fraction))

    def _strong_spatial_match(self, old, fresh):
        if iou(old.bbox, fresh.bbox) < self.config.incidental_min_iou:
            return False
        diagonal = min(math.hypot(d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1]) for d in (old, fresh))
        return math.dist(center(old.bbox), center(fresh.bbox)) <= diagonal*self.config.incidental_center_fraction

    @staticmethod
    def _possibly_same_object(a, b):
        # Class-agnostic association guards against claiming a saved neighbor.
        diagonal = min(math.hypot(d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1]) for d in (a, b))
        return (iou(a.bbox, b.bbox) >= 0.1 or
                math.dist(center(a.bbox), center(b.bbox)) <= max(0.002, diagonal*0.5))

    def _advance_confirmations(self, state, live, frame):
        records, predictions = [], []
        for record in state.confirmations:
            if frame-record.confirmed_frame > self.config.confirmed_frames:
                continue
            box = shifted(record.bbox, state.dx, state.dy, frame-record.position_frame)
            if box is not None:
                records.append(record)
                predictions.append(DroneFlybyPredictionDto(object_id=record.object_id,
                    bbox=list(box), confidence=float(record.confidence)))
        matches = {i: j for i, j in match_detections(predictions, live, self.config.confirmation_distance)}
        for index, record in enumerate(records):
            # Follow the spatial trajectory, without refreshing the L1 confirmation date.
            record.bbox = list(live[matches[index]].bbox if index in matches else predictions[index].bbox)
            record.position_frame = frame
        state.confirmations = records

    def _is_confirmed(self, state, detection):
        return any(record.object_id == detection.object_id and
                   math.dist(center(record.bbox), center(detection.bbox)) <= self.config.confirmation_distance
                   for record in state.confirmations)

    def _confirm(self, state, target, matched, frame):
        if (target is None or matched is None or matched.object_id != target.detection.object_id or
                matched.confidence < self.config.confirmation_confidence):
            return
        self._advance_confirmations(state, [], frame)
        state.confirmations = [r for r in state.confirmations if not
            (r.object_id == matched.object_id and math.dist(center(r.bbox), center(matched.bbox))
             <= self.config.confirmation_distance)]
        state.confirmations.append(Confirmation(matched.object_id, list(matched.bbox),
            frame, frame, float(matched.confidence)))

    def _focus_frame(self, state, request, live):
        state.l0_frames_since_focus = 0
        target = state.focus_target
        if target is not None:
            state.cooldowns.append(FocusRecord(target.track_id, target.detection.object_id,
                list(target.detection.bbox), target.observed_frame, state.full_refreshes))
            state.last_focused_class = target.detection.object_id
            state.last_focused_region = list(target.detection.bbox)
        state.focus_target = None
        state.focus_operations += 1
        source = request.view.source_region_xyxy
        region = (source[0]/request.original_width, source[1]/request.original_height,
                  source[2]/request.original_width, source[3]/request.original_height)
        observed_live = [d for d in live if intersection(d.bbox, region) > 0]
        snapshot = state.focus_snapshot if not state.snapshot_used else None
        status = 'ready' if snapshot is not None else ('consumed' if state.snapshot_used else 'missing')
        original_count = len(snapshot.annotations) if snapshot else 0
        gap = request.frame-snapshot.frame if snapshot else 0
        propagated, clipped_out = {}, 0
        individual_velocity = global_fallback = 0
        if snapshot is not None:
            for index, detection in enumerate(snapshot.annotations):
                velocity = snapshot.edge_velocities[index] if gap == 1 else None
                if velocity is not None:
                    box = clip_bbox_to_frame(tuple(b+v for b, v in zip(detection.bbox, velocity)))
                else:
                    # Individual edge prediction is strictly next-source-frame only.
                    # Preserve the existing global fallback for gaps in received frames.
                    box = shifted(detection.bbox, state.dx, state.dy, gap)
                if box is None:
                    clipped_out += 1
                else:
                    individual_velocity += int(velocity is not None)
                    global_fallback += int(velocity is None)
                    propagated[index] = DroneFlybyPredictionDto(object_id=detection.object_id,
                        bbox=list(box), confidence=float(detection.confidence))
            state.snapshot_used = True
        predicted_target = propagated.get(snapshot.target_index) if snapshot else None
        # A target match must not claim a different snapshot object's location.
        target_live = []
        if predicted_target is not None and intersection(predicted_target.bbox, region) > 0:
            for fresh in observed_live:
                overlap = iou(predicted_target.bbox, fresh.bbox)
                distance = math.dist(center(predicted_target.bbox), center(fresh.bbox))
                if any(index != snapshot.target_index and
                       self._possibly_same_object(old, fresh) and
                       (iou(old.bbox, fresh.bbox) >= overlap or
                        math.dist(center(old.bbox), center(fresh.bbox)) <= distance)
                       for index, old in propagated.items()):
                    continue
                target_live.append(fresh)
        matched = (self._match_target(target, target_live, state, request, predicted_target.bbox)
                   if predicted_target is not None else None)
        same_class = matched is not None and matched.object_id == predicted_target.object_id
        reclassified = bool(matched is not None and not same_class and
            matched.confidence >= self.config.confirmation_confidence and
            self._strong_spatial_match(predicted_target, matched))
        updated = bool(same_class or reclassified)
        removed = bool(predicted_target is not None and matched is None and
                       predicted_target.confidence <= 0.15)
        accepted = None
        if updated:
            accepted = predicted_target.model_copy(update={
                'object_id': matched.object_id, 'confidence': matched.confidence})
            propagated[snapshot.target_index] = accepted
        elif removed:
            del propagated[snapshot.target_index]
        annotations = list(propagated.values())[:500]
        kept = len(annotations)-int(updated)
        state.remembered_emitted += kept
        command = return_to_full(request)
        self._confirm(state, target, accepted, request.frame)
        logger.info('FOCUS_BRIDGE frame=%s snapshot_count=%s live_count=%s snapshot_kept=%s '
                    'target_updated=%s target_removed=%s snapshot_replaced=%s final_count=%s '
                    'snapshot_clipped_out=%s snapshot_overflow=0 source_frame_gap=%s snapshot_status=%s '
                    'individual_velocity=%s global_fallback=%s live_added=0',
                    request.frame, original_count, len(live), kept, int(updated), int(removed),
                    int(updated), len(annotations), clipped_out, gap, status,
                    individual_velocity, global_fallback)
        logger.info('frame=%s L1 memory_count=%s memory_expired=0 memory_replaced=%s '
                    'dx_per_frame=%.3f dy_per_frame=%.3f frame_gap=%s returning=%s',
                    request.frame, kept, int(updated), state.dx*request.original_width,
                    state.dy*request.original_height, state.motion_frame_gap, 'L0' if command else 'blocked')
        self._log_outcome(target, matched, request, kept, accepted, removed, reclassified)
        return FocusDecision(annotations, command, kept, replaced=int(updated),
                             cleared=clipped_out+int(removed))

    def _match_target(self, target, live, state, request, predicted_box=None):
        if target is None:
            return None
        box = (predicted_box if predicted_box is not None else
               shifted(target.detection.bbox, state.dx, state.dy, request.frame-target.observed_frame))
        candidates = []
        if box is not None:
            gate = max(0.002, min(self.config.outcome_match_distance,
                                 0.5*math.hypot(box[2]-box[0], box[3]-box[1])))
            for detection in live:
                overlap = iou(box, detection.bbox)
                distance = math.dist(center(box), center(detection.bbox))
                if overlap >= 0.1 or distance <= gate:
                    candidates.append((-overlap, distance, -detection.confidence,
                                       detection.object_id, tuple(detection.bbox), detection))
        return min(candidates, key=lambda item: item[:-1])[-1] if candidates else None

    def _log_outcome(self, target, matched, request, memory_count, accepted, removed, reclassified):
        previous = target.detection if target is not None else None
        area = ((previous.bbox[2]-previous.bbox[0])*(previous.bbox[3]-previous.bbox[1]) *
                request.original_width*request.original_height) if previous is not None else None
        logger.info('FOCUS_RESULT frame=%s target_track=%s target=%s L0_class=%s L0_conf=%s '
                    'L0_area_px2=%s L1_class=%s L1_conf=%s matched=%s class_changed=%s '
                    'confidence_increased=%s memory_count=%s same_class=%s confidence_changed=%s '
                    'target_removed=%s reclassification_accepted=%s', request.frame,
                    target.track_id if target else 'none', previous.object_id if previous else 'none',
                    previous.object_id if previous else 'none',
                    f'{previous.confidence:.4f}' if previous else 'none',
                    f'{area:.3f}' if area is not None else 'none',
                    matched.object_id if matched else 'none', f'{matched.confidence:.4f}' if matched else 'none',
                    str(matched is not None).lower(),
                    str(reclassified).lower(),
                    str(accepted is not None and accepted.confidence > previous.confidence).lower(), memory_count,
                    str(matched is not None and matched.object_id == previous.object_id).lower(),
                    str(accepted is not None and accepted.confidence != previous.confidence).lower(),
                    str(removed).lower(), str(reclassified).lower())


focus_policy = FocusPolicy()
