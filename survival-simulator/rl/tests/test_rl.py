import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl.config import Config, Curriculum, worker_seed, VALIDATION_SEEDS, TRAIN_SEED_LIMIT
from rl.env_wrapper import SurvivalEnv
from rl.observation import Encoder, BLOCKS, SELF_DIM, OBS_DIM, GLOBAL_DIM, edge_features
from rl.policy import SharedPolicy
from rl.rollout import gae, population_weights, prepare_batch, Collector
from rl.ppo import weighted_loss
from rl.checkpoint import save_checkpoint, load_checkpoint
from rl.metrics import validation_rank

torch.set_num_threads(1)


def agent(aid=0, **changes):
    a = dict(agent_id=aid, energy=150., max_energy=500., age=0., speed=10., sprint_speed=20.,
             hearing_radius=50., vision_range=200., vision_angle=math.pi / 3, biome="forest", observations=[])
    a.update(changes)
    return a


class FakeCore:
    """Scripted fixture; deliberately not a real simulator run."""
    def __init__(self, seed=0, dt=.1, child_tick=None, death_tick=None):
        self.dt, self.child_tick, self.death_tick = dt, child_tick, death_tick
        self.data = {0: agent()}
        self.requests = []
        self.env = SimpleNamespace(time=0., score=0.)
        self.env.get_agent_state = lambda aid: copy.deepcopy(self.data.get(aid))
        self.env.agents = [SimpleNamespace(agent_id=0)]

    def step(self, actions):
        self.requests.append([a.model_copy(deep=True) for _, a in actions])
        tick = len(self.requests)
        if tick == self.child_tick:
            self.data[1] = agent(1, energy=75.)
        if tick == self.death_tick:
            self.data.clear()
        self.env.time += self.dt
        self.env.score += self.dt
        self.env.agents = [SimpleNamespace(agent_id=i) for i in self.data]
        return dict(score=self.env.score, sim_time=self.env.time, num_agents=len(self.data),
                    observations=copy.deepcopy(list(self.data.values())))


class FakePool:
    def __init__(self, config):
        self.envs = {i: SurvivalEnv(config, lambda **kw: FakeCore(**kw, child_tick=1)) for i in range(config.workers)}
        self.seeds = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def exchange(self, commands):
        results = {}
        for i, (kind, payload) in commands.items():
            if kind == "reset":
                self.seeds.append(payload["seed"])
                results[i] = self.envs[i].reset(**payload)
            elif kind == "begin":
                results[i] = self.envs[i].begin(payload)
            else:
                results[i] = self.envs[i].continue_with(payload)
        return results


class EncodingTests(unittest.TestCase):
    def test_fixed_shapes_and_all_missing_masks(self):
        encoder = Encoder()
        a = agent()
        packet = encoder.encode([a], 0, 300)
        self.assertEqual(packet["observations"].shape, (1, OBS_DIM))
        self.assertEqual(packet["context"].shape, (GLOBAL_DIM,))
        offset = SELF_DIM
        for _, count, width in BLOCKS:
            np.testing.assert_array_equal(packet["observations"][0, offset:offset + count * width], 0)
            offset += count * width
        self.assertEqual(encoder.encode([], 0, 300)["observations"].shape, (0, OBS_DIM))

    def test_nearest_sort_and_masks(self):
        a = agent(observations=[dict(type="Fruit", distance=d, angle=0.) for d in (100, 30, 20, 10)])
        encoded = Encoder().local_features(a, 0)
        fruits = encoded[SELF_DIM:SELF_DIM + 12].reshape(3, 4)
        np.testing.assert_allclose(fruits[:, 0], np.array([10, 20, 30]) / 400)
        np.testing.assert_array_equal(fruits[:, -1], 1)

    def test_edges_closest_point_and_degenerate_segment(self):
        distance, features = edge_features(dict(coords=((-5, 10), (5, 10))))
        self.assertEqual(distance, 10)
        self.assertAlmostEqual(features[1], 1)
        self.assertEqual(edge_features(dict(coords=((3, 4), (3, 4))))[0], 5)

    def test_global_recent_events_expire(self):
        e = Encoder(recent_seconds=5)
        self.assertEqual(e.observe_population([agent()], 0), (0, 0))
        self.assertEqual(e.observe_population([agent(), agent(1)], 1), (1, 0))
        self.assertEqual(e.observe_population([agent(1)], 2), (0, 1))
        context = e.global_features([agent(1)], 2, 300)
        self.assertAlmostEqual(context[7], .05)
        self.assertAlmostEqual(context[8], .05)
        e.observe_population([agent(1)], 8)
        self.assertFalse(e.events)

    def test_no_input_mutation(self):
        a = agent(observations=[dict(type="Predator", distance=50, angle=.7, rel_dir=-.4)])
        before = copy.deepcopy(a)
        Encoder().encode([a], 1, 300)
        self.assertEqual(a, before)


class ActionRepeatTests(unittest.TestCase):
    def test_five_ticks_spawn_once_and_total_turn(self):
        env = SurvivalEnv(Config(), FakeCore)
        env.reset(0, 300)
        result = env.begin({0: np.array([.75, .25, .75, 1])})
        self.assertEqual(result["ticks"], 5)
        actions = [tick[0] for tick in env.core.requests]
        self.assertEqual([a.spawn_agent for a in actions], [True, False, False, False, False])
        self.assertTrue(all(a.move_distance == 15 for a in actions))
        self.assertTrue(all(abs(a.move_direction + math.pi / 2) < 1e-6 for a in actions))
        self.assertAlmostEqual(sum(a.turn_angle for a in actions), math.pi / 2, places=6)
        self.assertAlmostEqual(result["official_reward"], .5)
        self.assertAlmostEqual(result["reward"], .5)

    def test_newborn_immediately_requests_shared_actor(self):
        env = SurvivalEnv(Config(), lambda **kw: FakeCore(**kw, child_tick=1))
        env.reset(0, 300)
        result = env.begin({0: [.5, .5, .5, 0]})
        self.assertEqual(result["kind"], "need_actions")
        self.assertEqual(result["packet"]["ids"], [1])
        self.assertEqual(result["spawn_mask"], 0)
        result = env.continue_with({1: [.3, .5, 1., 1]})
        self.assertEqual(result["ticks"], 5)
        self.assertEqual(result["births"], 1)
        newborn = [a for tick in env.core.requests for a in tick if a.agent_id == 1]
        self.assertEqual(len(newborn), 4)
        self.assertFalse(any(a.spawn_agent for a in newborn))
        self.assertAlmostEqual(sum(a.turn_angle for a in newborn), math.pi, places=6)

    def test_extinction_stops_repeat_immediately(self):
        env = SurvivalEnv(Config(), lambda **kw: FakeCore(**kw, death_tick=2))
        env.reset(0, 300)
        result = env.begin({0: [.5, .5, .5, 0]})
        self.assertTrue(result["done"])
        self.assertEqual(result["ticks"], 2)
        self.assertEqual(result["deaths"], 1)
        self.assertAlmostEqual(result["reward"], .2 - 5)
        self.assertAlmostEqual(result["official_reward"], .2)

    def test_horizon_bonus_and_potential_terminal_zero(self):
        c = Config(energy_shaping=.01)
        env = SurvivalEnv(c, FakeCore)
        env.reset(0, .3)
        result = env.begin({0: [.5, .5, .5, 0]})
        self.assertTrue(result["done"])
        self.assertEqual(result["ticks"], 3)
        self.assertAlmostEqual(result["reward"], .3 + 5 - .003)
        self.assertEqual(result["episode"]["horizon_reached"], 1)

    def test_invalid_actions_rejected(self):
        env = SurvivalEnv(Config(), FakeCore)
        env.reset(0, 300)
        with self.assertRaises(ValueError):
            env.begin({0: [2, .5, .5, 0]})


class PolicyAndLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.policy = SharedPolicy((16, 16))

    def test_action_bounds_correct_log_prob_and_spawn_mask(self):
        observations = np.zeros((100, OBS_DIM), np.float32)
        masks = np.ones(100, np.float32)
        masks[:10] = 0
        actions, stored = self.policy.act(observations, spawn_mask=masks)
        self.assertTrue(np.all(actions >= 0) and np.all(actions <= 1))
        self.assertTrue(np.all(actions[:10, 3] == 0))
        self.assertTrue(set(actions[:, 3]) <= {0., 1.})
        physical_angles = (2 * actions[:, 1:3] - 1) * math.pi
        self.assertTrue(np.all(np.abs(physical_angles) <= math.pi))
        log_prob, entropy = self.policy.evaluate_actions(torch.from_numpy(observations), torch.from_numpy(actions), torch.from_numpy(masks))
        np.testing.assert_allclose(stored, log_prob.detach().numpy())
        self.assertTrue(torch.isfinite(entropy).all())

    def test_one_actor_handles_variable_population(self):
        parameters = tuple(id(p) for p in self.policy.actor.parameters())
        for n in (0, 1, 5, 12):
            actions, _ = self.policy.act(np.zeros((n, OBS_DIM), np.float32))
            self.assertEqual(actions.shape, (n, 4))
            self.assertEqual(parameters, tuple(id(p) for p in self.policy.actor.parameters()))

    def test_deterministic_evaluation(self):
        obs = np.zeros((3, OBS_DIM), np.float32)
        a, _ = self.policy.act(obs, deterministic=True)
        b, _ = self.policy.act(obs, deterministic=True)
        np.testing.assert_array_equal(a, b)

    def test_gae_does_not_cross_episode_boundary(self):
        adv, targets = gae([1., 2.], [0., 0.], [False, True], [.9, .9], 100., 1.)
        np.testing.assert_allclose(adv, [2.8, 2.])
        np.testing.assert_allclose(targets, adv)

    def test_population_weights_equalize_team_decisions(self):
        weights = np.concatenate((population_weights(1), population_weights(4)))
        losses = torch.tensor([2., 6., 6., 6., 6.])
        weights = torch.tensor(weights)
        result = weighted_loss(losses, weights, weights.mean())
        self.assertAlmostEqual(result.item(), 4.)
        self.assertAlmostEqual(population_weights(30).sum(), 1., places=6)

    def test_collector_batches_newborns_and_gae_targets(self):
        config = Config(workers=2)
        pool = FakePool(config)
        collector = Collector(pool, self.policy, config)
        collector.reset(range(2), 300)
        with patch.object(self.policy, "act", wraps=self.policy.act) as inference:
            transitions, episodes = collector.step(300)
        self.assertEqual(inference.call_count, 2)
        self.assertTrue(all(call.args[0].shape[0] == 2 for call in inference.call_args_list))
        self.assertEqual(len(transitions), 2)
        self.assertFalse(episodes)
        for t in transitions:
            self.assertEqual(len(t["actors"]["actions"]), 2)
            np.testing.assert_array_equal(t["actors"]["spawn_masks"], [1., 0.])
        batch = prepare_batch(transitions, collector.values(), config)
        self.assertEqual(batch["observations"].shape, (4, OBS_DIM))
        self.assertAlmostEqual(batch["weights"].sum(), 2.)
        self.assertTrue(np.isfinite(batch["advantages"]).all())

    def test_checkpoint_round_trip_rng_actor_critic_optimizer(self):
        config = Config(hidden_sizes=(16, 16))
        optimizer = torch.optim.Adam(self.policy.parameters())
        # Synthetic tensor-only optimizer fixture, not PPO or environment training.
        loss = self.policy.value(torch.zeros((1, GLOBAL_DIM))).sum()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            state = dict(training_step=3, environment_ticks=20, curriculum_stage=1)
            save_checkpoint(path, self.policy, optimizer, config, state)
            expected_rng = (random.random(), np.random.rand(), torch.rand(1).item())
            restored = SharedPolicy((16, 16))
            restored_optimizer = torch.optim.Adam(restored.parameters())
            payload = load_checkpoint(path, restored, restored_optimizer, restore_rng=True)
            actual_rng = (random.random(), np.random.rand(), torch.rand(1).item())
            self.assertEqual(expected_rng, actual_rng)
            self.assertEqual(payload["state"], state)
            for a, b in zip(self.policy.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


class DisciplineTests(unittest.TestCase):
    def test_windows_spawn_handshake_no_simulation_or_torch(self):
        from rl.workers import WorkerPool
        with WorkerPool(Config(workers=2)) as pool:
            replies = pool.exchange({0: ("ping", None), 1: ("ping", None)})
            self.assertEqual(len({r["pid"] for r in replies.values()}), 2)
            self.assertTrue(all(r["pid"] != os.getpid() for r in replies.values()))
            self.assertTrue(all(not r["torch_loaded"] for r in replies.values()))
            self.assertTrue(all(r["threads"] == "1" for r in replies.values()))
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_validation_uses_only_reserved_seeds_with_fake_environment(self):
        from rl.evaluate import validate
        config = Config(workers=1)
        pool = FakePool(config)
        policy = SharedPolicy((16, 16))
        before = copy.deepcopy(policy.state_dict())
        with patch("rl.workers.WorkerPool", return_value=pool):
            episodes = validate(policy, config, .2)
        self.assertEqual(pool.seeds, list(config.validation_seeds))
        self.assertEqual(len(episodes), 5)
        self.assertTrue(all(e["horizon_reached"] for e in episodes))
        for key, value in before.items():
            torch.testing.assert_close(value, policy.state_dict()[key])

    def test_ctrl_c_before_collection_still_checkpoints_without_training(self):
        from rl.train import main
        class InterruptedPool:
            closed = False
            def exchange(self, commands):
                raise KeyboardInterrupt
            def close(self):
                self.closed = True
        pool = InterruptedPool()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(dict(hidden_sizes=[16, 16], output_dir=directory, workers=1)), encoding="utf-8")
            with patch("rl.workers.WorkerPool", return_value=pool), patch("rl.ppo.update", side_effect=AssertionError("must not train")):
                main(["--config", str(config_path), "--max-hours", "0.01"])
            payload = load_checkpoint(Path(directory) / "latest.pt")
            self.assertEqual(payload["state"]["training_step"], 0)
            self.assertEqual(payload["state"]["environment_steps"], 0)
            self.assertTrue(list(Path(directory).glob("checkpoint_*.pt")))
            self.assertTrue(pool.closed)

    def test_unique_worker_seeds_and_disjoint_validation(self):
        seeds = [worker_seed(17, w, episode, 8) for episode in range(100) for w in range(8)]
        self.assertEqual(len(set(seeds)), 800)
        self.assertFalse(set(seeds) & set(VALIDATION_SEEDS))
        with self.assertRaises(ValueError):
            worker_seed(TRAIN_SEED_LIMIT - 1, 1, 0, 2)

    def test_curriculum_promotion_performance_fallback_and_restore(self):
        c = Config(curriculum_window=2, curriculum_max_decisions=10, curriculum_max_seconds=100)
        curriculum = Curriculum(c)
        episodes = [dict(horizon=300., horizon_reached=1)] * 2
        self.assertTrue(curriculum.advance(episodes=episodes))
        self.assertEqual(curriculum.horizon, 900)
        self.assertTrue(curriculum.advance(decisions=10))
        self.assertEqual(curriculum.horizon, 1800)
        restored = Curriculum(c)
        restored.load_state_dict(curriculum.state_dict())
        self.assertEqual(restored.horizon, 1800)
        self.assertTrue(restored.advance(seconds=100))
        self.assertEqual(restored.horizon, 3000)
        self.assertFalse(restored.advance(decisions=100))

    def test_validation_ranking_not_training_reward(self):
        a = [dict(complete=1, horizon_reached=1, survival_time=300, official_score=290)]
        b = [dict(complete=1, horizon_reached=0, survival_time=100, official_score=1000)]
        self.assertGreater(validation_rank(a), validation_rank(b))
        self.assertIsNone(validation_rank([dict(complete=0)]))

    def test_existing_simulator_and_controllers_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        expected = json.loads((Path(__file__).parent / "protected_files.sha256.json").read_text(encoding="utf-8"))
        for relative, digest in expected.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)

    def test_worker_module_has_no_torch_import(self):
        path = Path(__file__).resolve().parents[1] / "workers.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(not alias.name.startswith("torch") for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("torch"))


if __name__ == "__main__":
    unittest.main()
