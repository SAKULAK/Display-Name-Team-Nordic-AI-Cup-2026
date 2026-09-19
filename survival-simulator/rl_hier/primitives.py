"""Movement execution only; no energy/age/threat-based strategic mode switching."""
from enum import IntEnum
import math
import numpy as np
from src.utils.DTOs import ActionRequest


class Mode(IntEnum):
    EXPLORE = 0
    FORAGE = 1
    CAMP = 2
    EVADE = 3
    REGROUP = 4
    CONSERVE = 5


MODE_NAMES = tuple(m.name for m in Mode)


def validate_action(values):
    values = np.asarray(values, dtype=np.float32)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or values[0] != int(values[0]) or not 0 <= values[0] < len(Mode)
            or np.any(values[1:3] < 0) or np.any(values[1:3] > 1)
            or values[3] not in (0., 1.)):
        raise ValueError("Expected [integer mode 0..5, intensity 0..1, residual_unit 0..1, spawn 0/1]")
    return values


def nearest(agent, kind):
    candidates = [o for o in agent.get("observations", ()) if o.get("type") == kind
                  and math.isfinite(o.get("distance", float("nan")))
                  and math.isfinite(o.get("angle", float("nan"))) and o["distance"] >= 0]
    return min(candidates, key=lambda o: (o["distance"], o["angle"]), default=None)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def execute(agent, values, divisor=5, spawn_allowed=True):
    """Read only one public ObservationResponse; return an official legal action.

    Heading itself provides exploration continuity. No target memory, absolute
    coordinates, environment handle, or team summary is available here.
    """
    mode, intensity, unit, spawn = validate_action(values)
    mode, residual = Mode(int(mode)), float(2 * unit - 1)
    sprint = float(agent["sprint_speed"])
    distance = float(intensity) * sprint
    angle = residual * math.pi
    turn = angle / divisor
    target = None
    if mode == Mode.FORAGE:
        target = nearest(agent, "Fruit") or nearest(agent, "Tree")
    elif mode == Mode.CAMP:
        target = nearest(agent, "Tree") or nearest(agent, "Fruit")
    elif mode == Mode.EVADE:
        target = nearest(agent, "Predator")
    elif mode == Mode.REGROUP:
        target = nearest(agent, "Agent")
    if target is not None:
        base = target["angle"] + (math.pi if mode == Mode.EVADE else 0.)
        if mode == Mode.CAMP and target["distance"] <= 30.:
            # A quiet anchor vicinity: tangential movement instead of crossing
            # the center. Residual chooses orbit side and changes the heading.
            base += math.copysign(math.pi / 2, residual if residual else 1.)
            distance *= .15
        elif mode != Mode.EVADE:
            distance = float(intensity) * min(sprint, max(0., target["distance"] - (20. if mode == Mode.CAMP else 0.)))
        angle = wrap(base + residual * math.pi / 3)
        turn = angle / divisor
    elif mode == Mode.CONSERVE:
        distance = float(intensity) * min(sprint * .2, float(agent["speed"]) * .5)
        angle = residual * math.pi / 4
        turn = residual * math.pi / (8 * divisor)
    # Missing targets retain exactly EXPLORE's movement and turn.
    return ActionRequest(agent_id=agent["agent_id"], move_distance=distance,
                         move_direction=wrap(angle), turn_angle=turn,
                         spawn_agent=bool(spawn) and spawn_allowed)
