import copy
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from test_rl import FakePool
from rl.config import Config, VALIDATION_SEEDS, worker_seed
from rl.evaluate import validate
from rl.metrics import percentile25, validation_rank
from rl.policy import SharedPolicy
from rl.rollout import Collector
from rl.validation import validation_plan, complete_sweep, process_validation, rank_key


def config(**kw):
    return Config(**dict(dict(validation_policy_mode='stochastic', validation_policy_rollouts=3,
                             validation_policy_seed_base=1234), **kw))


def episodes(cfg, values=None, horizon=300.):
    plan = list(validation_plan(cfg))
    values = values or [100.] * len(plan)
    return [dict(p, seed=p['environment_seed'], horizon=horizon, complete=True,
                 survival_time=v, official_score=10., horizon_reached=False)
            for p, v in zip(plan, values)]


class RobustValidationTests(unittest.TestCase):
    def test_rollouts_reproducible_actions_and_seeds(self):
        policy = SharedPolicy((16, 16))
        cfg = config()
        traces = []
        class TrackingPool(FakePool):
            def exchange(self, commands):
                for kind, payload in commands.values():
                    if kind in ('begin', 'continue'):
                        traces[-1].append(np.stack(list(payload.values())).copy())
                return super().exchange(commands)
        results = []
        for _ in range(2):
            traces.append([])
            with patch('rl.workers.WorkerPool', TrackingPool):
                results.append(validate(policy, cfg, .2))
        self.assertEqual(len(results[0]), 15)
        self.assertEqual([e['policy_seed'] for e in results[0]], list(range(1234, 1249)))
        self.assertEqual([e['seed'] for e in results[0]], [s for s in VALIDATION_SEEDS for _ in range(3)])
        self.assertTrue(complete_sweep(results[0], cfg, .2))
        for a, b in zip(traces[0], traces[1]):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(traces[0][0], traces[0][2]))

    def test_training_rng_workers_and_next_seeds_unchanged(self):
        cfg = config(workers=2)
        policy = SharedPolicy((16, 16))
        live = FakePool(cfg)
        collector = Collector(live, policy, cfg)
        collector.reset(range(2), 300.)
        for i, env in live.envs.items():
            env.core.rng = random.Random(500 + i)
        worker_states = [env.core.rng.getstate() for env in live.envs.values()]
        packets = copy.deepcopy(collector.packets)
        indices = list(collector.episode_indices)
        next_seeds = [worker_seed(cfg.base_seed, i, indices[i], 2) for i in range(2)]
        if torch.cuda.is_available():
            torch.cuda.init()
        py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        class NoisyPool(FakePool):
            def exchange(self, commands):
                random.random(); np.random.random(); torch.rand(3)
                if cuda:
                    torch.rand(3, device='cuda')
                return super().exchange(commands)
        with patch('rl.workers.WorkerPool', NoisyPool):
            validate(policy, cfg, .2)
        self.assertEqual(py, random.getstate())
        np.testing.assert_equal(npstate, np.random.get_state())
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else []):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(worker_states, [env.core.rng.getstate() for env in live.envs.values()])
        self.assertEqual(indices, collector.episode_indices)
        for i in packets:
            np.testing.assert_array_equal(packets[i]['observations'], collector.packets[i]['observations'])
        collector.reset(range(2), 300.)
        self.assertEqual(live.seeds[-2:], next_seeds)

    def test_deadline_skip_and_exception_restore(self):
        policy = SharedPolicy((16, 16))
        with patch('rl.workers.WorkerPool') as pool:
            self.assertEqual(validate(policy, config(), .2, deadline=0), [])
            pool.assert_not_called()
        state = torch.get_rng_state()
        def failure(*args):
            torch.rand(1)
            raise RuntimeError('fixture failure')
        with patch('rl.workers.WorkerPool', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                validate(policy, config(), .2)
        self.assertTrue(policy.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_deadline_truncates_completed_prefix(self):
        clock = [0.]
        class ExpiringPool(FakePool):
            def exchange(self, commands):
                if any(kind == 'reset' for kind, _ in commands.values()) and self.seeds:
                    clock[0] = 2.
                    raise TimeoutError('deadline')
                return super().exchange(commands)
        cfg = config()
        with patch('rl.workers.WorkerPool', ExpiringPool), patch('rl.evaluate.time.monotonic', side_effect=lambda: clock[0]):
            rows = validate(SharedPolicy((16,16)), cfg, .2, deadline=1.)
        self.assertEqual(len(rows), 1)
        self.assertFalse(complete_sweep(rows, cfg, .2))

    def test_percentile_and_rank_tiebreaks(self):
        self.assertEqual(percentile25([300, 0, 200, 100]), 75)
        self.assertEqual(percentile25([25, 30, 40, 500, 600]), 30)
        self.assertEqual(percentile25([12]), 12)
        cfg = config(validation_policy_rollouts=1)
        def rank(vals, score=10):
            rows = episodes(cfg, vals)
            for e in rows:
                e['official_score'] = score
            return validation_rank(rows, 'stochastic', 5)
        self.assertGreater(rank([100,110,120,130,140]), rank([25,30,40,500,600]))
        self.assertGreater(rank([10,20,40,50,60]), rank([10,20,30,50,60]))
        self.assertGreater(rank([10,20,30,40,50], 11), rank([10,20,30,40,50], 10))
        self.assertEqual(len(validation_rank(episodes(Config()))), 3)

    def test_csv_metadata_partial_and_stage_selection(self):
        cfg = config()
        state = dict(training_step=20, environment_ticks=500, best_ranks={}, legacy='retained')
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            best = root / 'best_validation.pt'
            best.write_bytes(b'existing checkpoint')
            (root / 'validation.csv').write_text('training_step,seed,official_score,legacy_metric\n10,2147483648,5,kept\n')
            saved = []
            def save(name):
                saved.append((name, copy.deepcopy(state)))
            original = copy.deepcopy(state)
            self.assertFalse(process_validation(episodes(cfg)[:-1], cfg, state, 0, 300., root, save))
            self.assertEqual(state, original)
            self.assertEqual(best.read_bytes(), b'existing checkpoint')
            self.assertFalse((root / 'validation_summary.csv').exists())
            self.assertEqual(saved, [])
            self.assertTrue(process_validation(episodes(cfg), cfg, state, 0, 300., root, save))
            self.assertEqual([s[0] for s in saved], ['best_validation.pt', 'best_validation_stage0.pt'])
            metadata = saved[0][1]
            for key in ('policy_mode','policy_rollouts','policy_seed_base','completion_rate','q25_survival',
                        'median_survival','mean_survival','mean_official_score','rank'):
                self.assertIn('validation_' + key, metadata)
            self.assertEqual(metadata['legacy'], 'retained')
            with (root / 'validation_summary.csv').open() as f:
                row = next(csv.DictReader(f))
            required = 'training_step environment_ticks curriculum_stage horizon policy_mode policy_rollouts num_environment_seeds episode_count completion_rate q25_survival median_survival mean_survival min_survival max_survival mean_official_score median_official_score validation_rank'.split()
            self.assertTrue(set(required) <= row.keys())
            self.assertEqual(row['episode_count'], '15')
            with (root / 'validation.csv').open() as f:
                old = next(csv.DictReader(f))
            self.assertEqual(old['legacy_metric'], 'kept')
            self.assertEqual(old['validation_step'], '10')
            self.assertEqual(old['policy_mode'], 'deterministic')
            self.assertFalse(process_validation(episodes(cfg), cfg, state, 0, 300., root, save))
            self.assertTrue(process_validation(episodes(cfg, horizon=900.), cfg, state, 1, 900., root, save))
            self.assertEqual(saved[-1][0], 'best_validation_stage1.pt')
            self.assertEqual(len(state['best_ranks']), 2)

    def test_full_coverage_and_protocol_scope(self):
        cfg = config()
        rows = episodes(cfg)
        self.assertFalse(complete_sweep(rows[:-1] + [rows[0]], cfg, 300.))
        self.assertFalse(complete_sweep(rows, cfg, 900.))
        rows[0]['complete'] = False
        self.assertFalse(complete_sweep(rows, cfg, 300.))
        self.assertNotEqual(rank_key(cfg, 300.), rank_key(replace(cfg, validation_policy_rollouts=5), 300.))
        self.assertNotEqual(rank_key(cfg, 300.), rank_key(replace(cfg, validation_policy_seed_base=99), 300.))
        self.assertNotEqual(rank_key(cfg, 300.), rank_key(Config(), 300.))

    def test_legacy_default_and_config(self):
        cfg = Config()
        self.assertEqual((cfg.validation_policy_mode, cfg.validation_policy_rollouts), ('deterministic', 1))
        self.assertEqual(rank_key(cfg, 300.), '300.0')
        with self.assertRaisesRegex(ValueError, 'redundant'):
            Config(validation_policy_rollouts=2)
        with patch('rl.workers.WorkerPool', FakePool):
            rows = validate(SharedPolicy((16,16)), cfg, .2)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(e['policy_seed'] is None for e in rows))
        self.assertEqual(validation_rank(rows), (1, .2, .2))
        state = dict(training_step=1, environment_ticks=1, best_ranks={'300.0': (1,300,10)})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self.assertFalse(process_validation(episodes(cfg), cfg, state, 0, 300., tmp,
                                               lambda name: self.fail('old better rank overwritten')))
        hpc = json.loads(Path('rl/config_hpc_finetune.json').read_text())
        self.assertNotIn('workers', hpc)
        self.assertEqual(Config(**hpc).validation_policy_rollouts, 3)


if __name__ == '__main__':
    unittest.main()
