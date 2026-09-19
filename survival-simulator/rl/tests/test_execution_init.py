"""Synthetic tensors/checkpoints and scripted environments only; no real runs."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rl.config import Config, Curriculum, worker_seed
from rl.policy import SharedPolicy
from rl.observation import OBS_DIM
from rl.execution import PolicyExecution, rollout_seeds
from rl.checkpoint import save_checkpoint, load_checkpoint, initialize_weights
from rl.train import initialize_training, parse_args as train_args
from rl.evaluate import parse_args as evaluate_args, evaluate_episode, prepare_output, aggregate, main as evaluate_main
from rl.env_wrapper import SurvivalEnv
from rl.tests.test_rl import FakeCore

torch.set_num_threads(1)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.policy = SharedPolicy((16, 16))
        self.obs = np.zeros((256, OBS_DIM), np.float32)

    def test_deterministic_is_exact_old_actor_path(self):
        expected = self.policy.act(self.obs, deterministic=True)
        actual = PolicyExecution(self.policy).act(self.obs)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        with torch.no_grad():
            self.policy.actor[-1].bias[-1] = 0.
        self.assertTrue(np.all(PolicyExecution(self.policy).act(self.obs)[0][:, 3] == 1))

    def test_sampling_is_exact_native_beta_and_bernoulli_path(self):
        with torch.no_grad():
            self.policy.actor[-1].bias[-1] = -1.  # p<0.5; deterministic spawn is always false.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            expected = self.policy.act(self.obs, deterministic=False)
        actual = PolicyExecution(self.policy, "stochastic", 123).act(self.obs)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(actual[0][:, :3].std(axis=0) > .02))
        self.assertGreater(actual[0][:, 3].sum(), 0)
        self.assertLess(actual[0][:, 3].sum(), len(self.obs))
        self.assertFalse(PolicyExecution(self.policy).act(self.obs)[0][:, 3].any())

    def test_private_stream_reproducibility_and_progression(self):
        a, b = (PolicyExecution(self.policy, "stochastic", 44) for _ in range(2))
        first = a.act(self.obs)[0]
        np.testing.assert_array_equal(first, b.act(self.obs)[0])
        second = a.act(self.obs)[0]
        np.testing.assert_array_equal(second, b.act(self.obs)[0])
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, PolicyExecution(self.policy, "stochastic", 45).act(self.obs)[0]))

    def test_ambient_and_environment_rngs_untouched(self):
        environment_rng = random.Random(2147483748)
        env_before, py_before = environment_rng.getstate(), random.getstate()
        np_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
        PolicyExecution(self.policy, "stochastic", 12345).act(self.obs)
        self.assertEqual(env_before, environment_rng.getstate())
        self.assertEqual(py_before, random.getstate())
        np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])
        self.assertEqual(np_before[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_before, torch.get_rng_state())

    def test_rng_restored_even_on_actor_exception(self):
        before = torch.get_rng_state().clone()
        with patch.object(self.policy, "act", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                PolicyExecution(self.policy, "stochastic", 5).act(self.obs)
        torch.testing.assert_close(before, torch.get_rng_state())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_private_stream_and_ambient_state(self):
        policy = self.policy.to("cuda")
        before = torch.cuda.get_rng_state().clone()
        a, b = (PolicyExecution(policy, "stochastic", 24) for _ in range(2))
        first = a.act(self.obs)[0]
        np.testing.assert_array_equal(first, b.act(self.obs)[0])
        np.testing.assert_array_equal(a.act(self.obs)[0], b.act(self.obs)[0])
        torch.testing.assert_close(before, torch.cuda.get_rng_state())

    def test_fake_episode_newborn_sampling_reproducible(self):
        captures = []
        def factory(config):
            env = SurvivalEnv(config, lambda **kw: FakeCore(**kw, child_tick=1))
            captures.append(env)
            return env
        with patch("rl.env_wrapper.SurvivalEnv", side_effect=factory):
            first = evaluate_episode(self.policy, Config(), 9001, 1., policy_mode="stochastic", policy_seed=8)
            second = evaluate_episode(self.policy, Config(), 9001, 1., policy_mode="stochastic", policy_seed=8)
        self.assertEqual(captures[0].core.requests, captures[1].core.requests)
        self.assertEqual(first["policy_mode"], "stochastic")
        self.assertEqual(first["policy_seed"], 8)
        self.assertEqual(first["official_score"], second["official_score"])

    def test_old_cli_defaults_and_seed_mapping(self):
        args = evaluate_args(["--checkpoint", "example.pt"])
        self.assertEqual(args.policy_mode, "deterministic")
        self.assertEqual(args.policy_rollouts, 1)
        pairs = list(rollout_seeds([100, 200], "stochastic", 1000, 5))
        self.assertEqual(pairs, [(env, seed) for env in (100, 200) for seed in range(1000, 1005)])
        self.assertEqual(list(rollout_seeds([100])), [(100, None)])

    def test_old_csv_preserved_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            path.write_text("seed,official_score,survival_time\n9,123,120\n", encoding="utf-8")
            prepare_output(path)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0], dict(seed="9", official_score="123", survival_time="120",
                                           policy_mode="deterministic", policy_seed=""))
        stats = aggregate([dict(complete=1, survival_time=t, official_score=t+1, horizon_reached=int(t==20)) for t in (10, 20)])
        self.assertEqual(stats["mean_survival"], 15)
        self.assertEqual(stats["median_survival"], 15)
        self.assertEqual(stats["min_survival"], 10)
        self.assertEqual(stats["max_survival"], 20)
        self.assertEqual(stats["horizon_completion_rate"], .5)


class InitFromTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.pt"
        self.config = Config(hidden_sizes=(16, 16), workers=2, base_seed=17,
                             output_dir=str(self.root / "old_run"), max_hours=42, device="cpu")
        torch.manual_seed(777)
        self.policy = SharedPolicy((16, 16))
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        # Construct a nonempty Adam-state fixture without any optimizer/training step.
        for p in self.policy.parameters():
            optimizer.state[p] = dict(step=torch.tensor(7.), exp_avg=torch.ones_like(p), exp_avg_sq=torch.ones_like(p))
        curriculum = Curriculum(self.config)
        curriculum.stage, curriculum.stage_decisions, curriculum.stage_seconds = 2, 91, 321.
        self.old_state = dict(training_step=622, environment_ticks=2390000, decision_steps=478000,
                              episodes_completed=80, wall_time=50000., births=90, deaths=88,
                              population_max=11, best_ranks={"1800": [0, 430, 450]}, episode_indices=[40, 40],
                              curriculum=curriculum.state_dict(), curriculum_stage=2)
        save_checkpoint(self.source, self.policy, optimizer, self.config, self.old_state)
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.new_config = self.root / "new.json"
        self.new_config.write_text(json.dumps(dict(hidden_sizes=[16, 16], base_seed=1000000, device="cpu",
                                                  output_dir=str(self.root / "fresh"))), encoding="utf-8")

    def args(self, extra=()):
        return train_args(["--init-from", str(self.source), "--config", str(self.new_config),
                           "--workers", "6", "--learning-rate", "1e-4", "--start-horizon", "3000", *extra])

    def test_init_loads_both_weights_only_and_records_provenance(self):
        config, policy, optimizer, curriculum, state = initialize_training(self.args())
        for name in ("actor", "critic"):
            for expected, actual in zip(getattr(self.policy, name).parameters(), getattr(policy, name).parameters()):
                torch.testing.assert_close(expected, actual)
        self.assertFalse(optimizer.state)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        for name in ("training_step", "environment_ticks", "decision_steps", "wall_time", "births", "deaths", "episodes_completed"):
            self.assertEqual(state[name], 0)
        self.assertEqual(state["episode_indices"], [0] * 6)
        self.assertEqual(config.workers, 6)
        self.assertEqual(config.base_seed, 1000000)
        self.assertEqual(config.max_hours, 6)
        self.assertEqual(config.output_dir, str(self.root / "fresh"))
        self.assertEqual(curriculum.horizon, 3000)
        self.assertEqual(curriculum.stage_decisions, 0)
        self.assertEqual(curriculum.stage_seconds, 0)
        self.assertFalse(curriculum.recent)
        self.assertEqual(worker_seed(config.base_seed, 0, state["episode_indices"][0], config.workers), 1000000)
        self.assertEqual(state["initialized_from_training_step"], 622)
        self.assertEqual(state["initialized_from_environment_ticks"], 2390000)
        self.assertEqual(state["initialized_from"], str(self.source.resolve()))
        output = self.root / "fresh_checkpoint.pt"
        save_checkpoint(output, policy, optimizer, config, dict(state, curriculum=curriculum.state_dict()))
        self.assertEqual(load_checkpoint(output)["state"]["initialized_from_training_step"], 622)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_fresh_rng_same_as_ordinary_new_run_not_source(self):
        args = self.args()
        initialize_training(args)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        args.init_from = None
        initialize_training(args)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2])

    def test_resume_retains_optimizer_counters_curriculum_and_rng(self):
        config, policy, optimizer, curriculum, state = initialize_training(train_args(["--resume", str(self.source)]))
        self.assertEqual(state, self.old_state)
        self.assertTrue(optimizer.state)
        self.assertEqual(config.workers, 2)
        self.assertEqual(config.base_seed, 17)
        self.assertEqual(curriculum.state_dict(), self.old_state["curriculum"])
        saved = load_checkpoint(self.source)
        torch.testing.assert_close(torch.get_rng_state(), saved["rng"]["torch"])
        for a, b in zip(policy.parameters(), self.policy.parameters()):
            torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "worker count"):
            initialize_training(train_args(["--resume", str(self.source), "--workers", "6"]))

    def test_ordinary_fresh_start_remains_first_horizon(self):
        *_, curriculum, state = initialize_training(train_args(["--config", str(self.new_config)]))
        self.assertEqual(curriculum.horizon, 300)
        self.assertNotIn("initialized_from", state)

    def test_ambiguous_or_invalid_start_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train_args(["--resume", "x", "--init-from", "y"])
            with self.assertRaises(SystemExit):
                train_args(["--resume", "x", "--start-horizon", "3000"])
        with self.assertRaisesRegex(ValueError, "start-horizon"):
            initialize_training(self.args(["--start-horizon", "1234"]))

    def test_incompatible_critic_does_not_partially_load_actor(self):
        saved = load_checkpoint(self.source)
        saved["critic"]["0.weight"] = torch.zeros((7, 7))
        bad = self.root / "bad.pt"
        torch.save(saved, bad)
        model = SharedPolicy((16, 16))
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "Incompatible critic architecture"):
            initialize_weights(bad, model)
        for key in before:
            torch.testing.assert_close(before[key], model.state_dict()[key])

    def test_init_refuses_source_directory_or_nonempty_output(self):
        with self.assertRaisesRegex(ValueError, "empty output"):
            initialize_training(self.args(["--output-dir", str(self.root)]))

    def test_evaluation_cli_writes_ten_separate_rollouts(self):
        output = self.root / "evaluated.csv"
        def fake_episode(policy, config, seed, horizon, **kwargs):
            return dict(seed=seed, complete=1, survival_time=100., official_score=101., horizon_reached=0,
                        policy_mode=kwargs["policy_mode"], policy_seed=kwargs["policy_seed"])
        with patch("rl.evaluate.evaluate_episode", side_effect=fake_episode) as run, contextlib.redirect_stdout(io.StringIO()) as printed:
            evaluate_main(["--checkpoint", str(self.source), "--seeds", "2147483748", "2147483749",
                           "--policy-mode", "stochastic", "--policy-rollouts", "5", "--policy-seed", "1000",
                           "--output", str(output), "--device", "cpu"])
        self.assertEqual(run.call_count, 10)
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10)
        self.assertEqual([int(r["policy_seed"]) for r in rows], list(range(1000, 1005)) * 2)
        self.assertTrue(all(r["policy_mode"] == "stochastic" for r in rows))
        self.assertIn("mean_survival", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
