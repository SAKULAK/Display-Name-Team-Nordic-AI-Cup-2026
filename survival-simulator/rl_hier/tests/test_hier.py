import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl.observation import Encoder, LOCAL_DIM, GLOBAL_DIM, OBS_DIM
from rl.execution import PolicyExecution
from rl_hier.config import Config, Curriculum, VALIDATION_SEEDS, worker_seed
from rl_hier.policy import SharedPolicy
from rl_hier.primitives import Mode, execute
from rl_hier.env_wrapper import SurvivalEnv
from rl_hier.rollout import Collector, prepare_batch, population_weights
from rl_hier.checkpoint import save_checkpoint, load_checkpoint, save_periodic
from rl_hier.validation import preserve_training_rng, process_validation, validation_plan
from rl_hier.metrics import action_statistics, validation_rank
from rl_hier.train import initialize_training, parse_args, restore_collector

torch.set_num_threads(1)


def agent(aid=0, **changes):
    row = dict(agent_id=aid, energy=150., max_energy=500., age=0., speed=10., sprint_speed=20.,
               hearing_radius=50., vision_range=200., vision_angle=math.pi / 3,
               biome="forest", observations=[])
    row.update(changes)
    return row


class FakeCore:
    """Picklable scripted test fixture; never constructs a real simulation."""
    def __init__(self, seed=0, dt=.1, child_tick=None, death_tick=None):
        self.dt, self.child_tick, self.death_tick = dt, child_tick, death_tick
        self.rng = random.Random(seed)
        self.data = {0: agent()}
        self.requests = []
        self.env = self
        self.time = self.score = 0.
        self.agents = [SimpleNamespace(agent_id=0)]

    def get_agent_state(self, aid):
        return copy.deepcopy(self.data.get(aid))

    def step(self, actions):
        self.requests.append([action.model_copy(deep=True) for _, action in actions])
        tick = len(self.requests)
        if tick == self.child_tick:
            self.data[1] = agent(1)
        if tick == self.death_tick:
            self.data.clear()
        self.time += self.dt
        self.score += self.dt
        self.agents = [SimpleNamespace(agent_id=i) for i in self.data]
        return dict(score=self.score, sim_time=self.time, observations=copy.deepcopy(list(self.data.values())))


def newborn_core(**kwargs):
    return FakeCore(**kwargs, child_tick=1)


class FakePool:
    def __init__(self, config):
        self.envs = {i: SurvivalEnv(config, newborn_core) for i in range(config.workers)}
        self.deadline = float("inf")
        self.closed = False

    def exchange(self, commands):
        from rl_hier.snapshot import dumps, loads
        result = {}
        for wid, (command, payload) in commands.items():
            env = self.envs[wid]
            if command == "reset":
                result[wid] = env.reset(**payload)
            elif command == "begin":
                result[wid] = env.begin(payload)
            elif command == "continue":
                result[wid] = env.continue_with(payload)
            elif command == "snapshot":
                result[wid] = dumps(env)
            elif command == "restore":
                self.envs[wid] = loads(payload)
                result[wid] = self.envs[wid].packet()
        return result

    def close(self):
        self.closed = True


class PrimitiveTests(unittest.TestCase):
    def test_all_modes_finite_legal(self):
        seen = [dict(type=kind, distance=10., angle=.3) for kind in ("Fruit", "Tree", "Predator", "Agent")]
        for mode in Mode:
            for intensity in (0., .4, 1.):
                for residual_unit in (0., .5, 1.):
                    action = execute(agent(observations=seen), [mode, intensity, residual_unit, 1])
                    self.assertTrue(all(math.isfinite(x) for x in (action.move_distance, action.move_direction, action.turn_angle)))
                    self.assertTrue(0 <= action.move_distance <= 20)
                    self.assertTrue(-math.pi <= action.move_direction <= math.pi)
                    self.assertTrue(abs(action.turn_angle) <= math.pi / 5)

    def test_missing_entities_exact_explore_fallback(self):
        expected = execute(agent(), [Mode.EXPLORE, .7, .2, 1])
        for mode in (Mode.FORAGE, Mode.CAMP, Mode.EVADE, Mode.REGROUP):
            self.assertEqual(execute(agent(), [mode, .7, .2, 1]), expected)

    def test_forage_prefers_fruit_then_tree(self):
        tree = dict(type="Tree", distance=10., angle=1.)
        fruit = dict(type="Fruit", distance=60., angle=-1.)
        self.assertAlmostEqual(execute(agent(observations=[tree, fruit]), [1, 1, .5, 0]).move_direction, -1.)
        self.assertAlmostEqual(execute(agent(observations=[tree]), [1, 1, .5, 0]).move_direction, 1.)

    def test_camp_vicinity_and_approach(self):
        close = execute(agent(observations=[dict(type="Tree", distance=10., angle=0.)]), [2, 1, .5, 0])
        far = execute(agent(observations=[dict(type="Tree", distance=100., angle=0.)]), [2, 1, .5, 0])
        self.assertAlmostEqual(close.move_direction, math.pi / 2)
        self.assertLess(close.move_distance, far.move_distance)
        self.assertEqual(far.move_direction, 0.)

    def test_evade_nearest_and_regroup_bearing(self):
        seen = [dict(type="Predator", distance=d, angle=a) for d, a in ((50, 1), (10, 0))]
        self.assertAlmostEqual(abs(execute(agent(observations=seen), [3, 1, .5, 0]).move_direction), math.pi)
        seen = [dict(type="Agent", distance=40, angle=-.7)]
        self.assertAlmostEqual(execute(agent(observations=seen), [4, 1, .5, 0]).move_direction, -.7)

    def test_conserve_and_no_strategic_energy_rules(self):
        regular = execute(agent(), [0, 1, .8, 1])
        conserve = execute(agent(), [5, 1, .8, 1])
        self.assertLess(conserve.move_distance, regular.move_distance)
        for mode in Mode:
            self.assertEqual(execute(agent(), [mode, .4, .2, 1]),
                             execute(agent(energy=0, age=120), [mode, .4, .2, 1]))

    def test_reject_invalid_actions(self):
        for values in ([6, .5, .5, 0], [1.5, .5, .5, 0], [0, -1, .5, 0], [0, .5, float("nan"), 0], [0, .5, .5, .1]):
            with self.assertRaises(ValueError):
                execute(agent(), values)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.policy = SharedPolicy((16, 16))
        self.obs = torch.zeros(12, OBS_DIM)

    def test_individual_and_combined_logprob_and_entropy(self):
        actions, old = self.policy.act(self.obs)
        actions = torch.as_tensor(actions)
        mode, continuous, spawn = self.policy.distributions(self.obs)
        actual, entropy = self.policy.evaluate_actions(self.obs, actions, torch.ones(12))
        expected = mode.log_prob(actions[:, 0].long()) + continuous.log_prob(actions[:, 1:3]).sum(-1)
        expected += spawn.log_prob(actions[:, 3]) - math.log(2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(entropy, mode.entropy() + continuous.entropy().sum(-1) + spawn.entropy() + math.log(2))
        torch.testing.assert_close(actual.detach(), torch.as_tensor(old))
        self.assertTrue(((actions[:, 0] >= 0) & (actions[:, 0] < 6)).all())
        self.assertTrue((actions[:, 0] == actions[:, 0].round()).all())
        with torch.no_grad():
            self.policy.actor[-1].bias.add_(torch.linspace(-.05, .05, 11))
        new, _ = self.policy.evaluate_actions(self.obs, actions, torch.ones(12))
        ratio = (new - torch.as_tensor(old)).exp()
        self.assertTrue(torch.isfinite(ratio).all())
        # Synthetic backward only; no optimizer update or training.
        (-new.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.policy.actor.parameters()))

    def test_deterministic_heads(self):
        actions, _ = self.policy.act(self.obs, deterministic=True)
        mode, continuous, spawn = self.policy.distributions(self.obs)
        np.testing.assert_array_equal(actions[:, 0], mode.probs.argmax(-1))
        np.testing.assert_allclose(actions[:, 1:3], continuous.mean.detach())
        np.testing.assert_array_equal(actions[:, 3], (spawn.probs >= .5).detach())

    def test_masked_spawn_has_no_density_or_entropy(self):
        actions, _ = self.policy.act(self.obs, spawn_mask=np.zeros(12))
        self.assertTrue((actions[:, 3] == 0).all())
        first = self.policy.evaluate_actions(self.obs, torch.as_tensor(actions), torch.zeros(12))
        with torch.no_grad():
            self.policy.actor[-1].bias[-1] += 10
        second = self.policy.evaluate_actions(self.obs, torch.as_tensor(actions), torch.zeros(12))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_fixed_seed_stream_reproducibility_and_isolation(self):
        before = torch.get_rng_state().clone()
        left, right = PolicyExecution(self.policy, "stochastic", 44), PolicyExecution(self.policy, "stochastic", 44)
        for _ in range(3):
            a, lp = left.act(self.obs)
            b, rp = right.act(self.obs)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(lp, rp)
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_actor_requires_exact_baseline_observation_size(self):
        self.assertEqual(OBS_DIM, 103)
        self.assertEqual(self.policy.actor[0].in_features, OBS_DIM)
        self.assertEqual(self.policy.critic[0].in_features, GLOBAL_DIM)
        for size in (LOCAL_DIM, OBS_DIM + 1):
            with self.assertRaises(ValueError):
                self.policy.act(np.zeros((1, size), np.float32))
        self.assertEqual(self.policy.act(np.zeros((1, OBS_DIM), np.float32))[0].shape, (1, 4))

    def test_actor_observation_and_critic_context_match_baseline(self):
        env = SurvivalEnv(Config(), FakeCore)
        first = env.reset(17, 300)
        env.state["observations"].append(agent(1, energy=400, age=80,
            observations=[dict(type="Predator", distance=25., angle=.2)]))
        env.encoder.observe_population(env.state["observations"], env.state["sim_time"])
        second = env.packet({0})
        np.testing.assert_array_equal(first["observations"][:, :LOCAL_DIM], second["observations"][:, :LOCAL_DIM])
        self.assertFalse(np.array_equal(first["context"], second["context"]))
        self.assertEqual(first["observations"].shape, (1, OBS_DIM))
        self.assertEqual(first["context"].shape, (GLOBAL_DIM,))
        for ids in (None, {0}, {1}, set()):
            actual = env.packet(ids)
            expected = Encoder.encode(env.encoder, env.state["observations"],
                                      env.state["sim_time"], env.horizon, ids)
            self.assertEqual(actual["ids"], expected["ids"])
            np.testing.assert_array_equal(actual["observations"], expected["observations"])
            np.testing.assert_array_equal(actual["context"], expected["context"])
            np.testing.assert_array_equal(actual["observations"][:, LOCAL_DIM:],
                np.repeat(actual["context"][None, :], len(actual["ids"]), axis=0))
        # Extra simulator-only fields are not added to either public encoding.
        before = env.packet()
        env.state["observations"][0].update(x=999., y=-999., hidden_predator_energy=400.)
        after = env.packet()
        np.testing.assert_array_equal(before["observations"], after["observations"])
        np.testing.assert_array_equal(before["context"], after["context"])

    def test_hierarchical_metrics(self):
        actions, _ = self.policy.act(self.obs)
        batch = dict(actions=actions, observations=self.obs.numpy(), weights=np.ones(12), spawn_masks=np.ones(12))
        row = action_statistics(self.policy, batch)
        self.assertAlmostEqual(sum(row[f"mode_{m.name.lower()}_fraction"] for m in Mode), 1.)
        self.assertTrue(all(np.isfinite(v) for v in row.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_stochastic_rng_isolation(self):
        self.policy.to("cuda")
        before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        left = PolicyExecution(self.policy, "stochastic", 77)
        right = PolicyExecution(self.policy, "stochastic", 77)
        np.testing.assert_array_equal(left.act(self.obs)[0], right.act(self.obs)[0])
        torch.testing.assert_close(torch.get_rng_state(), before_cpu)
        torch.testing.assert_close(torch.cuda.get_rng_state(), before_cuda)


class EnvironmentTests(unittest.TestCase):
    def test_newborn_shared_policy_and_population_neutral_team_batch(self):
        config = Config(workers=2)
        collector = Collector(FakePool(config), SharedPolicy((8, 8)), config)
        collector.reset(range(2), 300)
        transitions, episodes = collector.step(300)
        self.assertFalse(episodes)
        batch = prepare_batch(transitions, collector.values(), config)
        for t in transitions:
            self.assertEqual(t["ticks"], 5)
            self.assertEqual(len(t["actors"]["actions"]), 2)
            np.testing.assert_array_equal(t["actors"]["spawn_masks"], [1, 0])
        self.assertAlmostEqual(float(batch["weights"].sum()), 2.)
        self.assertAlmostEqual(float(population_weights(100).sum()), 1., places=5)
        self.assertEqual(batch["contexts"].shape[1], GLOBAL_DIM)
        self.assertEqual(batch["observations"].shape[1], OBS_DIM)

    def test_five_ticks_and_first_tick_spawn_only(self):
        env = SurvivalEnv(Config(), newborn_core)
        env.reset(0, 300)
        result = env.begin({0: [0, .3, .6, 1]})
        self.assertEqual(result["kind"], "need_actions")
        self.assertEqual(result["spawn_mask"], 0.)
        result = env.continue_with({1: [0, .3, .6, 1]})
        self.assertEqual(result["ticks"], 5)
        self.assertTrue(env.core.requests[0][0].spawn_agent)
        self.assertFalse(any(a.spawn_agent for tick in env.core.requests[1:] for a in tick))
        self.assertAlmostEqual(env.core.requests[1][1].turn_angle, (.6 * 2 - 1) * math.pi / 4, places=6)
        self.assertAlmostEqual(result["reward"], .5)

    def test_terminal_reward_and_discount(self):
        for extinct in (False, True):
            env = SurvivalEnv(Config(), lambda **kw: FakeCore(**kw, death_tick=1 if extinct else None))
            env.reset(0, .1)
            result = env.begin({0: [0, 0, .5, 0]})
            self.assertTrue(result["done"])
            self.assertAlmostEqual(result["reward"], .1 + (-5 if extinct else 5))
            self.assertAlmostEqual(result["discount"], .9995 ** .2)

    def test_snapshot_resume_live_fake_episode_and_seed_stream(self):
        config = Config(workers=1)
        policy = SharedPolicy((8, 8))
        collector = Collector(FakePool(config), policy, config)
        collector.reset([0], 300)
        collector.step(300)
        snapshots = collector.pool.exchange({0: ("snapshot", None)})
        restored = Collector(FakePool(config), policy, config, collector.episode_indices)
        restore_collector(restored, dict(worker_snapshots=snapshots), 300)
        np.testing.assert_array_equal(collector.packets[0]["observations"], restored.packets[0]["observations"])
        self.assertEqual(restored.episode_indices, [1])
        rng = torch.get_rng_state()
        left, _ = collector.step(300)
        torch.set_rng_state(rng)
        right, _ = restored.step(300)
        np.testing.assert_array_equal(left[0]["actors"]["actions"], right[0]["actors"]["actions"])
        self.assertEqual(left[0]["reward"], right[0]["reward"])
        collector.reset([0], 900)
        restored.reset([0], 900)
        self.assertEqual(restored.pool.envs[0].metrics.seed, worker_seed(config.base_seed, 0, 1, 1))

    def test_surface_and_simulator_object_snapshot_one_tick(self):
        # Tiny official object fixture, no generated full game or episodes.
        import pygame
        from src.core import SimulationCore
        from src.elements.environment import Environment
        from rl_hier.snapshot import dumps, loads
        core = SimulationCore.__new__(SimulationCore)
        core.rng, core.dt, core.seed = random.Random(7), .1, 7
        core.env = Environment(160, 160, 400, core.rng)
        core.env.spawn_agent(x=80, y=80)
        core.env.agents_dict = {a.agent_id: a for a in core.env.agents}
        env = SurvivalEnv(Config(), FakeCore)
        env.reset(7, .1)
        env.core = core
        env.state = dict(score=0., sim_time=0., observations=[core.env.get_agent_state(a.agent_id) for a in core.env.agents])
        blob = dumps(env)
        restored = loads(blob)
        self.assertIs(restored.core.rng, restored.core.env.rng)
        self.assertEqual(core.rng.getstate(), restored.core.rng.getstate())
        self.assertEqual(pygame.image.tobytes(core.env.biome_surface, "RGBA"),
                         pygame.image.tobytes(restored.core.env.biome_surface, "RGBA"))
        aid = core.env.agents[0].agent_id
        # One tick per branch verifies restored official mechanics can execute.
        left = env.begin({aid: [0, 0, .5, 0]})
        right = restored.begin({aid: [0, 0, .5, 0]})
        self.assertEqual(left["reward"], right["reward"])
        self.assertEqual(left["ticks"], 1)
        np.testing.assert_array_equal(left["packet"]["observations"], right["packet"]["observations"])


class CheckpointValidationTests(unittest.TestCase):
    def test_checkpoint_full_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--output-dir", tmp, "--device", "cpu", "--workers", "1"])
            config, policy, optimizer, curriculum, state = initialize_training(args)
            # A single synthetic scalar update verifies nonempty Adam slots only.
            optimizer.zero_grad()
            policy.value(torch.ones(1, GLOBAL_DIM)).square().sum().backward()
            optimizer.step()
            curriculum.advance(decisions=7, seconds=8)
            state.update(curriculum=curriculum.state_dict(), training_step=3, episode_indices=[9])
            save_periodic(tmp, policy, optimizer, config, state)
            path = Path(tmp) / "latest.pt"
            expected = (random.random(), np.random.random(), torch.rand(3))
            restored = initialize_training(parse_args(["--resume", str(path), "--max-hours", "1"]))
            rc, rp, ro, rcur, rs = restored
            actual = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            torch.testing.assert_close(actual[2], expected[2])
            for k, v in policy.state_dict().items():
                torch.testing.assert_close(v, rp.state_dict()[k])
            self.assertEqual(len(ro.state), len(optimizer.state))
            for original, loaded in zip(optimizer.state.values(), ro.state.values()):
                for key in original:
                    torch.testing.assert_close(original[key], loaded[key])
            self.assertEqual(rcur.state_dict(), curriculum.state_dict())
            self.assertEqual(rs["episode_indices"], [9])
            self.assertEqual(rs["training_step"], 3)
            self.assertTrue(list(Path(tmp).glob("checkpoint_step3_*.pt")))
            with self.assertRaises(ValueError):
                initialize_training(parse_args(["--resume", str(path), "--workers", "2"]))

    def test_baseline_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.pt"
            torch.save(dict(version=1), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_old_91_input_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old_hierarchy.pt"
            torch.save(dict(format="rl_hier_v1_local91_categorical6_beta2_bernoulli"), path)
            with self.assertRaisesRegex(ValueError, "compatible hierarchical checkpoint"):
                load_checkpoint(path)

    def test_fair_pc_config_and_evaluation_seed_defaults(self):
        from rl_hier.evaluate import parse_args as evaluation_args
        config_path = Path(__file__).resolve().parents[1] / "config_pc_overnight.json"
        pc = Config(**json.loads(config_path.read_text(encoding="utf-8")))
        for config in (Config(), pc):
            self.assertEqual(config.workers, 6)
            self.assertEqual(config.rollout_steps, 128)
            self.assertEqual(config.validation_policy_seed_base, 20260918)
            self.assertEqual(config.validation_policy_rollouts, 3)
        args = evaluation_args(["--checkpoint", "unused.pt", "--policy-mode", "stochastic"])
        self.assertEqual(args.policy_seed, 20260918)

    def test_rng_context_restores_on_exception(self):
        random.seed(20)
        np.random.seed(20)
        torch.manual_seed(20)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            with preserve_training_rng():
                random.random(), np.random.random(), torch.rand(2)
                raise RuntimeError("interrupted")
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2])

    def test_complete_and_partial_validation_ranking_without_sweeps(self):
        config = Config()
        plan = list(validation_plan(config))
        self.assertEqual(len(plan), 15)
        self.assertEqual(len({p["policy_seed"] for p in plan}), 15)
        self.assertEqual([p["policy_seed"] for p in plan], list(range(20260918, 20260918 + 15)))
        self.assertEqual(set(p["environment_seed"] for p in plan), set(VALIDATION_SEEDS))
        episodes = [dict(seed=p["environment_seed"], horizon=300., complete=1,
                         horizon_reached=1, survival_time=300., official_score=290., **p) for p in plan]
        saved = []
        state = dict(training_step=1, environment_ticks=5, best_ranks={})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(process_validation(episodes[:-1], config, state, 0, 300., tmp, saved.append))
            self.assertEqual(saved, [])
            self.assertTrue(process_validation(episodes, config, state, 0, 300., tmp, saved.append))
            self.assertEqual(len(saved), 2)
            duplicate = episodes[:-1] + [episodes[0]]
            self.assertFalse(process_validation(duplicate, config, state, 0, 300., tmp, saved.append))
            self.assertEqual(len(saved), 2)
            self.assertEqual(state["validation_rank"], (1., 300., 300., 290.))
            interrupted = copy.deepcopy(episodes)
            interrupted[-1]["complete"] = 0
            self.assertFalse(process_validation(interrupted, config, state, 0, 300., tmp, saved.append))
            self.assertEqual(len(saved), 2)
        self.assertEqual(len(validation_rank(episodes, "deterministic", 1)), 4)

    def test_fixed_contract_and_output_protection(self):
        for changes in (dict(energy_shaping=.01), dict(extinction_penalty=1), dict(dt=.2), dict(action_repeat=1), dict(output_dir="rl/runs/unsafe")):
            with self.assertRaises(ValueError):
                Config(**changes)

    def test_worker_handshake_no_simulator_reset_or_torch(self):
        from rl_hier.workers import WorkerPool
        with WorkerPool(Config(workers=1)) as pool:
            response = pool.exchange({0: ("ping", None)})[0]
        self.assertFalse(response["torch_loaded"])
        self.assertEqual(response["threads"], "1")

    def test_interrupted_collection_keeps_consistent_checkpoint(self):
        from rl_hier.train import main
        class InterruptedPool(FakePool):
            def exchange(self, commands):
                if any(command == "begin" for command, _ in commands.values()):
                    raise KeyboardInterrupt
                return super().exchange(commands)
        with tempfile.TemporaryDirectory() as tmp:
            pool = InterruptedPool(Config(workers=1))
            with patch("rl_hier.workers.WorkerPool", return_value=pool), \
                    patch("rl_hier.ppo.update", side_effect=AssertionError("must not train")):
                main(["--output-dir", tmp, "--workers", "1", "--device", "cpu", "--max-hours", ".01"])
            state = load_checkpoint(Path(tmp) / "latest.pt")["state"]
            self.assertEqual(state["training_step"], 0)
            self.assertEqual(state["decision_steps"], 0)
            self.assertEqual(state["episode_indices"], [0])
            self.assertTrue(pool.closed)

    def test_requested_stop_skips_validation_without_workers(self):
        from rl_hier.evaluate import validate
        with patch("rl_hier.workers.WorkerPool", side_effect=AssertionError("must not simulate")):
            self.assertEqual(validate(SharedPolicy((8, 8)), Config(), 300, should_stop=lambda: True), [])


if __name__ == "__main__":
    unittest.main()
