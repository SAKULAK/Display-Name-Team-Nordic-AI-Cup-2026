"""Cross-frame object memory.

A response only reflects the current camera crop unless something carries
detections forward into frames where the camera is pointed elsewhere. That is
what makes zooming additive instead of lossy: ``baseline_sweep`` (0.022) lost
almost everything a zoomed-out ``hold_full`` (0.388/0.366 held-out) already
had, purely because it only ever reported the current crop while being scored
against the whole frame. This module is the fix, independent of any camera
policy: it fuses fresh per-frame detections with tracks carried forward from
earlier frames, so a response degrades gracefully instead of forgetting
everything outside the current view.

Motion model, measured from the supplied Helsinki annotations (see the
frame-to-frame center-delta dump): per-object frame-to-frame displacement is
not a single shared translation (different classes drift at different
lateral rates, -11px to +13.5px/frame in this scene) and it is not exactly
constant velocity either (forward displacement drifts from ~55px/frame to
~80px/frame over 20 frames for the same object, a mild perspective/parallax
effect, not a sudden jump). A per-track EMA of the observed velocity,
re-estimated on every fresh match and extrapolated linearly between
observations, tracks this well enough over the short gaps a missed detection
or an unlooked-at region actually produces; accuracy decays with distance
from the last real observation, which is exactly why confidence decays too.

Boxes are frame-global normalized xyxy throughout, matching what a response
sends. Propagated boxes are always re-clipped with ``clip_bbox_to_frame``:
one invalid box fails the entire response, so a track that has drifted off
the edge must be dropped, not clamped into a degenerate sliver.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Tuple

from dtos import DroneFlybyPredictionDto
from utils import clip_bbox_to_frame

Bbox = Tuple[float, float, float, float]

# How much predicted-box overlap with a fresh detection counts as "the same
# object", within one class. Detections at IoU 0.50 or better score as a hit,
# so 0.30 leaves room for a track that has drifted a bit without either
# matching two different real objects or spawning a duplicate track for one.
IOU_MATCH_THRESHOLD = 0.20

# Same-class NMS applied to the tracker's own output every frame. The
# evaluator does no NMS server-side and counts overlapping duplicates as
# false positives (see README). Duplicates happen here even within a single
# fresh detection set -- a tiny object (e.g. small_launcher, ~5px wide once
# downsampled to a 960x540 L0 view) lets the raw detector emit multiple
# candidate boxes too jittery to clear its own NMS IoU (0.5), and each
# unmatched leftover then seeds its own track next frame. Left unsuppressed
# these compound into parallel "ghost chains": confirmed by inspecting one
# frame's actual output, where a single real large_launcher showed up as six
# co-existing tracks at evenly-decaying confidence, one per several missed
# real frames, because greedy matching only ever reconnects the fresh
# detection to whichever of the several stale duplicates currently has the
# best IoU, leaving the others to decay in parallel instead of merging.
DEDUP_IOU_THRESHOLD = 0.20

# Confidence applied per elapsed frame a track goes unconfirmed. Chosen so a
# track surviving ~15 missed frames (a plausible zoom-elsewhere gap) is still
# above a normal detection threshold, while one going much longer fades out
# rather than lingering as a stale false positive.
CONFIDENCE_DECAY_PER_FRAME = 0.93

# Below this, a propagated detection is more likely wrong than right; drop it
# rather than spend one of the 500 annotation slots on noise. Set just under
# the detector's own accept threshold (YOLO_CONF=0.15, found by sweeping):
# a borderline-confidence false positive decays past it in ~3 missed frames
# (ln(0.12/0.15)/ln(0.93)), while a real high-confidence detection (0.7-0.9
# typical for this checkpoint) survives 20+ missed frames, which is the
# actual point of a floor this low. Originally 0.05, tuned against
# YOLO_CONF=0.05; that combination let one-off noise decay for ~15 missed
# frames before expiring, and detections/frame climbed to a steady ~50
# against ~16 real objects in the scene -- confirmed by watching the
# server's per-frame detection counts, not just the score.
MINIMUM_CONFIDENCE = 0.12

# Hard cap regardless of decayed confidence, in case a track saturates near 1.0
# and would otherwise take many frames to decay below the confidence floor.
MAXIMUM_MISSED_FRAMES = 40

# Weight on the newest observed velocity in the running estimate. Below 1 so a
# single noisy re-detection cannot swing the extrapolation on its own.
VELOCITY_SMOOTHING = 0.5

# Weight on this frame's median observed velocity in the running scene-motion
# estimate, used to seed brand-new tracks. Measured on the Helsinki
# annotations, per-frame displacement is large relative to object size (small
# objects can move by several times their own box size in one frame — see
# module docstring), so a new track seeded at velocity zero fails its very
# next match attempt and spawns a duplicate "ghost" of the same object
# instead of updating it. Seeding from the dominant shared motion (mostly
# drone ego-motion) fixes the common case; individual object drift still
# gets picked up as soon as that track is matched once.
SCENE_VELOCITY_SMOOTHING = 0.3

MAXIMUM_ANNOTATIONS = 500


def _center(bbox: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _suppress_duplicates(tracks: List['Track']) -> List['Track']:
    """Greedy same-class NMS, highest confidence first.

    Standalone from the fresh-detection/track matching above: matching only
    ever pairs one detection with one existing track, so it cannot merge two
    tracks that both already exist and happen to overlap (parallel ghost
    chains -- see DEDUP_IOU_THRESHOLD's comment). This is the cleanup pass
    that actually collapses those, run every frame over the full surviving
    set. ``tracks`` must already be confidence-sorted descending.
    """
    kept: List[Track] = []
    for track in tracks:
        if any(
            other.object_id == track.object_id
            and _iou(track.bbox, other.bbox) >= DEDUP_IOU_THRESHOLD
            for other in kept
        ):
            continue
        kept.append(track)
    return kept


@dataclass
class Track:
    object_id: str
    bbox: Bbox
    confidence: float
    last_frame: int
    velocity: Tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    # True once matched at least once. Until then its velocity is only the
    # scene-motion seed, not evidence from this object's own history.
    confirmed: bool = False

    def propagate(self, to_frame: int) -> Optional[Bbox]:
        """Where this track would be at ``to_frame``, or None if off-frame."""
        elapsed = to_frame - self.last_frame
        x1, y1, x2, y2 = self.bbox
        dx, dy = self.velocity
        shift_x, shift_y = dx * elapsed, dy * elapsed
        return clip_bbox_to_frame((x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y))


class SequenceTracker:
    """Per-sequence track memory: fuse fresh detections into carried state."""

    def __init__(self):
        self.tracks: List[Track] = []
        self.scene_velocity: Tuple[float, float] = (0.0, 0.0)

    def update(
        self,
        frame: int,
        detections: List[DroneFlybyPredictionDto],
    ) -> List[DroneFlybyPredictionDto]:
        propagated: Dict[int, Optional[Bbox]] = {
            index: track.propagate(frame) for index, track in enumerate(self.tracks)
        }

        # Greedy IoU matching, same class only, highest IoU first.
        candidates = []
        for detection_index, detection in enumerate(detections):
            for track_index, track in enumerate(self.tracks):
                if track.object_id != detection.object_id:
                    continue
                predicted = propagated[track_index]
                if predicted is None:
                    continue
                iou = _iou(tuple(detection.bbox), predicted)
                if iou >= IOU_MATCH_THRESHOLD:
                    candidates.append((iou, detection_index, track_index))
        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_detections, matched_tracks = set(), set()
        observed_velocities: List[Tuple[float, float]] = []
        for iou, detection_index, track_index in candidates:
            if detection_index in matched_detections or track_index in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_index)
            detection = detections[detection_index]
            track = self.tracks[track_index]

            elapsed = max(1, frame - track.last_frame)
            old_center = _center(track.bbox)
            new_center = _center(tuple(detection.bbox))
            observed_velocity = (
                (new_center[0] - old_center[0]) / elapsed,
                (new_center[1] - old_center[1]) / elapsed,
            )
            observed_velocities.append(observed_velocity)
            # A track's own history dominates once it has one; scene motion
            # only matters for tracks that have never been matched before.
            blend = VELOCITY_SMOOTHING if track.confirmed else 1.0
            track.velocity = (
                blend * observed_velocity[0] + (1 - blend) * track.velocity[0],
                blend * observed_velocity[1] + (1 - blend) * track.velocity[1],
            )
            track.bbox = tuple(detection.bbox)
            track.confidence = float(detection.confidence)
            track.last_frame = frame
            track.missed_frames = 0
            track.confirmed = True

        if observed_velocities:
            observed_velocities.sort(key=lambda v: v[1])
            median = observed_velocities[len(observed_velocities) // 2]
            self.scene_velocity = (
                SCENE_VELOCITY_SMOOTHING * median[0]
                + (1 - SCENE_VELOCITY_SMOOTHING) * self.scene_velocity[0],
                SCENE_VELOCITY_SMOOTHING * median[1]
                + (1 - SCENE_VELOCITY_SMOOTHING) * self.scene_velocity[1],
            )

        surviving: List[Track] = []
        for track_index, track in enumerate(self.tracks):
            if track_index in matched_tracks:
                surviving.append(track)
                continue
            predicted = propagated[track_index]
            elapsed = max(1, frame - track.last_frame)
            if predicted is None:
                continue
            track.bbox = predicted
            track.last_frame = frame
            track.missed_frames += elapsed
            track.confidence *= CONFIDENCE_DECAY_PER_FRAME ** elapsed
            # Pull the velocity estimate towards the current scene motion
            # each missed frame, rather than propagating forever on whatever
            # was observed at the last real match. Real motion in this scene
            # drifts (~1px/frame^2, see module docstring); a frozen velocity
            # falls further behind the longer a track goes unconfirmed, and
            # this is measured to matter: l1_cycle's biggest holdout losses
            # were exactly on multi-frame-propagated tracks.
            track.velocity = (
                SCENE_VELOCITY_SMOOTHING * self.scene_velocity[0]
                + (1 - SCENE_VELOCITY_SMOOTHING) * track.velocity[0],
                SCENE_VELOCITY_SMOOTHING * self.scene_velocity[1]
                + (1 - SCENE_VELOCITY_SMOOTHING) * track.velocity[1],
            )
            if track.confidence < MINIMUM_CONFIDENCE or track.missed_frames > MAXIMUM_MISSED_FRAMES:
                continue
            surviving.append(track)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            surviving.append(
                Track(
                    object_id=detection.object_id,
                    bbox=tuple(detection.bbox),
                    confidence=float(detection.confidence),
                    last_frame=frame,
                    velocity=self.scene_velocity,
                )
            )

        surviving.sort(key=lambda track: track.confidence, reverse=True)
        self.tracks = _suppress_duplicates(surviving)[:MAXIMUM_ANNOTATIONS]

        return [
            DroneFlybyPredictionDto(
                object_id=track.object_id,
                bbox=list(track.bbox),
                confidence=max(0.0, min(1.0, track.confidence)),
            )
            for track in self.tracks
        ]


_sequence_trackers: Dict[str, SequenceTracker] = {}
_lock = Lock()


def track(
    sequence_id: str,
    frame: int,
    detections: List[DroneFlybyPredictionDto],
) -> List[DroneFlybyPredictionDto]:
    """Fuse this frame's fresh detections into the sequence's carried memory."""
    with _lock:
        tracker = _sequence_trackers.setdefault(sequence_id, SequenceTracker())
        return tracker.update(frame, detections)
