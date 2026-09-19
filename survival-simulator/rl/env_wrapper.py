"""Official simulator adapter with five-tick macro decisions and newborn inference.

Only this adapter imports the simulator, lazily at reset. There are no behavioral
rules: actions are executed as sampled, bounded neural-policy outputs.
"""
import math
import numpy as np
from src.utils.DTOs import ActionRequest
from rl.observation import Encoder, fraction
from rl.metrics import EpisodeMetrics


class SurvivalEnv:
    def __init__(self, config, core_factory=None):
        self.config = config
        self.core_factory = core_factory
        self.core = None

    def reset(self, seed, horizon):
        if self.core_factory is None:
            from src.core import SimulationCore
            factory = SimulationCore
        else:
            factory = self.core_factory
        self.core = factory(seed=seed, dt=self.config.dt)
        self.horizon = horizon
        self.encoder = Encoder()
        self.metrics = EpisodeMetrics(seed, horizon)
        # Public state accessor at t=0; observations may be empty until first tick.
        agents = [self.core.env.get_agent_state(a.agent_id) for a in self.core.env.agents]
        agents = [a for a in agents if a is not None]
        self.state = dict(score=self.core.env.score, sim_time=self.core.env.time, observations=agents)
        self.encoder.observe_population(agents, self.state["sim_time"])
        self.metrics.observe(agents)
        self.active = False
        return self.packet()

    def packet(self, only_ids=None):
        return self.encoder.encode(self.state["observations"], self.state["sim_time"], self.horizon, only_ids)

    def potential(self):
        agents = self.state["observations"]
        return sum(fraction(a) for a in agents) / len(agents) if agents else 0.

    def begin(self, actions):
        if self.active:
            raise RuntimeError("A macro decision is already in progress")
        self.active = True
        self.tick = 0
        self.actions = {}
        self.divisors = {}
        self.score_before = self.state["score"]
        self.potential_before = self.potential()
        self.births_before, self.deaths_before = self.metrics.births, self.metrics.deaths
        self.metrics.decisions += 1
        self._accept(actions)
        return self._advance()

    def continue_with(self, actions):
        if not self.active:
            raise RuntimeError("No pending newborn inference")
        self._accept(actions)
        return self._advance()

    def _accept(self, actions):
        for aid, values in actions.items():
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("Action must be four finite unit-interval values")
            if values[3] not in (0., 1.):
                raise ValueError("Spawn action must be Bernoulli 0/1")
            if aid in self.actions:
                raise ValueError("Cannot replace a selected action mid-repeat")
            self.actions[aid] = values
            self.divisors[aid] = self.config.action_repeat - self.tick

    def _advance(self):
        while self.tick < self.config.action_repeat:
            agents = self.state["observations"]
            missing = {a["agent_id"] for a in agents} - self.actions.keys()
            if missing:
                return dict(kind="need_actions", packet=self.packet(missing), spawn_mask=float(self.tick == 0))
            requests = []
            for a in agents:
                aid = a["agent_id"]
                move, direction, turn, spawn = self.actions[aid]
                request = ActionRequest(agent_id=aid, move_distance=float(move * a["sprint_speed"]),
                                        move_direction=float((2 * direction - 1) * math.pi),
                                        turn_angle=float((2 * turn - 1) * math.pi / self.divisors[aid]),
                                        spawn_agent=bool(spawn) and self.tick == 0)
                requests.append((aid, request))
            self.state = self.core.step(requests)
            self.state["observations"] = [a for a in self.state["observations"] if a is not None]
            self.tick += 1
            self.metrics.ticks += 1
            births, deaths = self.encoder.observe_population(self.state["observations"], self.state["sim_time"])
            self.metrics.observe(self.state["observations"], births, deaths)
            extinct = not self.state["observations"]
            horizon_reached = self.state["sim_time"] + 1e-8 >= self.horizon
            if extinct or horizon_reached:
                return self._finish(True, extinct)
        return self._finish(False, False)

    def _finish(self, done, extinct):
        self.active = False
        official = self.state["score"] - self.score_before
        discount = self.config.gamma ** (self.tick / self.config.action_repeat)
        potential_after = 0. if done else self.potential()
        reward = official + self.config.energy_shaping * (discount * potential_after - self.potential_before)
        if done:
            reward += -self.config.extinction_penalty if extinct else self.config.horizon_bonus
        self.metrics.official_reward += official
        self.metrics.training_reward += reward
        return dict(kind="transition", packet=self.packet(), reward=reward, official_reward=official,
                    done=done, discount=discount, ticks=self.tick,
                    births=self.metrics.births - self.births_before, deaths=self.metrics.deaths - self.deaths_before,
                    peak_population=max(self.metrics.populations, default=0),
                    episode=self.metrics.report(self.state) if done else None)
