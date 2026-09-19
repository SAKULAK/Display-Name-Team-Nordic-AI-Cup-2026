"""Five-tick high-level decisions, with public feedback for primitive execution."""
from rl.env_wrapper import SurvivalEnv as BaselineEnv
from rl_hier.primitives import execute, validate_action


class SurvivalEnv(BaselineEnv):
    # Inherit the exact baseline actor observations and centralized critic context.
    def _accept(self, actions):
        for aid, values in actions.items():
            if aid in self.actions:
                raise ValueError("Cannot replace a selected action mid-repeat")
            self.actions[aid] = validate_action(values).copy()
            self.divisors[aid] = self.config.action_repeat - self.tick

    def _advance(self):
        while self.tick < self.config.action_repeat:
            agents = self.state["observations"]
            missing = {a["agent_id"] for a in agents} - self.actions.keys()
            if missing:
                return dict(kind="need_actions", packet=self.packet(missing), spawn_mask=float(self.tick == 0))
            requests = [(a["agent_id"], execute(a, self.actions[a["agent_id"]],
                         self.divisors[a["agent_id"]], self.tick == 0)) for a in agents]
            self.state = self.core.step(requests)
            self.state["observations"] = [a for a in self.state["observations"] if a is not None]
            self.tick += 1
            self.metrics.ticks += 1
            births, deaths = self.encoder.observe_population(self.state["observations"], self.state["sim_time"])
            self.metrics.observe(self.state["observations"], births, deaths)
            extinct = not self.state["observations"]
            if extinct or self.state["sim_time"] + 1e-8 >= self.horizon:
                return self._finish(True, extinct)
        return self._finish(False, False)
