# Execution modes and fresh initialization

The actor, critic architecture, observations, rewards, curriculum promotion
rules, and action distributions are unchanged. The overnight checkpoint remains
read-only. Its inspected metadata is training step 622, 2,385,975 environment
ticks, and hidden sizes (256, 256).

## Evaluation

`--policy-mode deterministic` remains the default: Beta means and Bernoulli
probability >=0.5, exactly as before. `--policy-mode stochastic` calls the same
`SharedPolicy.act(..., deterministic=False)` path used during training, including
the existing newborn spawn mask. It adds no noise, temperature, clipping, or
probability adjustment. Training-time validation defaults to deterministic; see [robust stochastic validation](ROBUST_VALIDATION.md) for configurable stochastic sweeps.

`--policy-seed` defaults to 0 for stochastic execution and is independent of the
environment seed. `--policy-rollouts` defaults to 1. For every environment seed,
rollout index i uses `base_policy_seed + i`, starting at i=0. For example, two
environment seeds with five rollouts and base policy seed 1000 produce ten CSV
rows, using 1000..1004 separately for each environment. The private stream resets
at episode start and advances across all decisions, including newborn inference.

PyTorch distribution sampling does not accept a dedicated Generator argument.
`PolicyExecution` therefore initializes private CPU/CUDA generator states and
temporarily installs them with `torch.random.fork_rng` around each actor call.
The private states advance, while ambient Torch states are restored afterward,
including on exceptions. Python, NumPy, and the simulator's random.Random are
not seeded or consumed by this helper. This is a serial evaluator facility, not
a thread-safe global-RNG swap for concurrent actor threads. Reproducibility is
expected with the same checkpoint, observations, seed, device, PyTorch build,
and runtime; bit-identical results across CPU/GPU or library versions are not
promised.

Evaluation CSVs retain every metric and append `policy_mode` and `policy_seed`.
Deterministic rows have a blank policy seed. Existing legacy evaluation CSVs are
atomically extended with those columns, tagging historical rows deterministic.
Separate output paths remain preferable when comparing experiments. Each
rollout is a separate row. Aggregate output reports completed-episode mean,
median, minimum, maximum survival, mean official score, and horizon completion
rate; interrupted episodes are saved but excluded from that summary.

## Initialization versus resume

`--init-from PATH` loads actor and critic parameters with strict key/shape checks
on both modules before loading either. It does not inherit source config,
optimizer, counters, wall time, curriculum counters, worker episode indices,
RNG, seed stream, output paths, or rollout buffers. The new config is formed
from fresh Config defaults, optional `--config`, and CLI overrides. Specify a
new `--base-seed` and output directory for the new experiment. The source
checkpoint's architecture must match the new config; no architecture is silently
inferred or changed.

`--start-horizon 3000` selects that exact value in the new config's `horizons`.
The stage's counters start at zero, and normal promotion rules remain unchanged.
Without the option, ordinary fresh training still starts at 300 seconds.
`--learning-rate 1e-4` overrides config learning_rate; the default stays 3e-4.

`--resume` preserves existing behavior, including optimizer/counters/RNG and
curriculum restoration, worker-count/base-seed restrictions, and fresh simulator
episodes at the saved next episode indices. Combining resume with init-from is
an argparse error. Combining resume with start-horizon is also rejected because
resume must restore its saved curriculum.

Init-from requires an empty/new output directory separate from the source
checkpoint directory. This prevents subsequent latest/best saves from replacing
the source. Checkpoint `state` records provenance fields `initialized_from`
(resolved source path), `initialized_from_training_step`, and
`initialized_from_environment_ticks`. Those values are provenance only, not the
new experiment's counters. They persist through subsequent saves/resume.

## Windows PowerShell commands — user-run only

Run from `survival-simulator`. Use the same device for closely reproducible
comparisons. These commands were prepared, not executed.

```powershell
$checkpoint = 'rl/runs/sanity/best_validation.pt'
$pairedSeeds = 2147483748..2147483752

# A. Deterministic evaluation: five environment seeds.
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint $checkpoint --seeds $pairedSeeds --policy-mode deterministic --output rl/runs/eval_deterministic.csv

# B. Stochastic evaluation of the SAME seeds: 25 episodes total.
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint $checkpoint --seeds $pairedSeeds --policy-mode stochastic --policy-rollouts 5 --policy-seed 1000 --output rl/runs/eval_stochastic_5.csv

# C. Twenty additional environment seeds: 20 stochastic episodes.
# These are outside the training partition and default validation seeds.
# If you have already evaluated them elsewhere, choose another unused range.
$freshSeeds = 2147483848..2147483867
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint $checkpoint --seeds $freshSeeds --policy-mode stochastic --policy-rollouts 1 --policy-seed 2000 --output rl/runs/eval_stochastic_fresh20.csv

# D. New short run: six workers, fresh Adam, full horizon, LR=1e-4.
# 0.1 hours is a six-minute budget, including startup/shutdown reserve.
rl\.venv\Scripts\python.exe -m rl.train --init-from $checkpoint --start-horizon 3000 --workers 6 --learning-rate 1e-4 --base-seed 1000000 --max-hours 0.1 --device auto --output-dir rl/runs/init_full_short
```

The short run may finish before scheduled validation. Its final `latest.pt` is
still saved, but no new `best_validation.pt` is claimed without validation.

E. Future Linux/HPC shell template, from the same project directory after copying
the checkpoint and installing the environment. Set worker count to suit the
scheduler allocation and use a fresh output directory for each experiment:

```bash
WORKERS=16
python -m rl.train \
  --init-from rl/runs/sanity/best_validation.pt \
  --start-horizon 3000 --workers "$WORKERS" \
  --learning-rate 1e-4 --base-seed 2000000 \
  --max-hours 6 --device auto --output-dir rl/runs/hpc_full
```

## Verification

42 lightweight tests passed, including CPU and CUDA policy-stream isolation,
exact comparison to existing deterministic/stochastic actor paths, scripted
newborn rollouts, ten-row multi-rollout CSV output, old-CSV migration, fresh
init-from state, strict architecture failures without partial loads, unchanged
resume behavior, and protected official-file hashes. Tests use synthetic
checkpoints and scripted fake environments; no real evaluation, training,
benchmark, or PPO sweep was run. The overnight checkpoint and RL architecture,
encoding, reward, curriculum, and PPO source files were checked by SHA-256.
