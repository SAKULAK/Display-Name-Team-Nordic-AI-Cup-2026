"""Fixed-size encoding of public observations; no action recommendations."""
import math
from collections import deque
import numpy as np

BIOMES = ("forest", "grassland", "desert", "swamp", "river", "unknown")
SELF_DIM = 16
GLOBAL_DIM = 12
BLOCKS = (("Fruit", 3, 4), ("Tree", 3, 4), ("Predator", 2, 6), ("Agent", 3, 6), ("Edge", 3, 7))
LOCAL_DIM = SELF_DIM + sum(count * width for _, count, width in BLOCKS)
OBS_DIM = LOCAL_DIM + GLOBAL_DIM


def fraction(a):
    return float(np.clip(a["energy"] / max(a["max_energy"], 1e-6), 0, 1))


def edge_features(o):
    (x1, y1), (x2, y2) = o["coords"]
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    t = max(0., min(1., -(x1 * dx + y1 * dy) / length2)) if length2 else 0.
    x, y = x1 + t * dx, y1 + t * dy
    distance, angle = math.hypot(x, y), math.atan2(y, x)
    orientation = math.atan2(dy, dx)
    return distance, [distance / 400, math.sin(angle), math.cos(angle),
                      math.sin(orientation), math.cos(orientation), math.sqrt(length2) / 400, 1.]


class Encoder:
    def __init__(self, recent_seconds=5.):
        self.recent_seconds = recent_seconds
        self.previous_ids = None
        self.events = deque()

    def observe_population(self, agents, sim_time):
        ids = {a["agent_id"] for a in agents}
        births = deaths = 0
        if self.previous_ids is not None:
            births = len(ids - self.previous_ids)
            deaths = len(self.previous_ids - ids)
            if births or deaths:
                self.events.append((sim_time, births, deaths))
        self.previous_ids = ids
        while self.events and self.events[0][0] < sim_time - self.recent_seconds:
            self.events.popleft()
        return births, deaths

    def global_features(self, agents, sim_time, horizon):
        fractions = [fraction(a) for a in agents]
        ages = [a["age"] for a in agents]
        n = len(agents)
        return np.asarray([
            n / 20., np.mean(fractions) if n else 0., np.median(fractions) if n else 0.,
            min(fractions, default=0.), np.mean(ages) / 120 if n else 0.,
            sum(a < 30 for a in ages) / max(n, 1), sum(a > 80 for a in ages) / max(n, 1),
            sum(e[1] for e in self.events) / 20., sum(e[2] for e in self.events) / 20.,
            sum(o.get("type") == "Predator" for a in agents for o in a["observations"]) / 20.,
            sim_time / 3000., horizon / 3000.], dtype=np.float32)

    def local_features(self, a, sim_time):
        biome = a["biome"] if a["biome"] in BIOMES else "unknown"
        features = [a["energy"] / 1000., fraction(a), a["age"] / 120., a["speed"] / 20.,
                    a["sprint_speed"] / 40., a["hearing_radius"] / 100., a["vision_range"] / 400.,
                    a["vision_angle"] / (math.pi / 2), a["max_energy"] / 1000., sim_time / 3000.]
        features += [float(b == biome) for b in BIOMES]
        for kind, count, width in BLOCKS:
            observations = [o for o in a["observations"] if o.get("type") == kind]
            if kind == "Edge":
                ordered = sorted((edge_features(o) for o in observations), key=lambda item: (item[0], item[1]))
                rows = [row for _, row in ordered[:count]]
            else:
                ordered = sorted(observations, key=lambda o: (o["distance"], o["angle"], o.get("rel_dir", 0)))[:count]
                rows = []
                for o in ordered:
                    row = [o["distance"] / 400., math.sin(o["angle"]), math.cos(o["angle"])]
                    if kind in ("Predator", "Agent"):
                        row += [math.sin(o.get("rel_dir", 0)), math.cos(o.get("rel_dir", 0))]
                    rows.append(row + [1.])
            features += [v for row in rows for v in row]
            features += [0.] * ((count - len(rows)) * width)
        return np.asarray(features, dtype=np.float32)

    def encode(self, agents, sim_time, horizon, only_ids=None):
        context = self.global_features(agents, sim_time, horizon)
        chosen = sorted((a for a in agents if only_ids is None or a["agent_id"] in only_ids), key=lambda a: a["agent_id"])
        rows = [np.concatenate((self.local_features(a, sim_time), context)) for a in chosen]
        observations = np.stack(rows) if rows else np.empty((0, OBS_DIM), dtype=np.float32)
        if not np.isfinite(observations).all() or not np.isfinite(context).all():
            raise ValueError("Nonfinite public observation")
        return dict(ids=[a["agent_id"] for a in chosen], observations=observations,
                    context=context, sim_time=sim_time, horizon=horizon)
