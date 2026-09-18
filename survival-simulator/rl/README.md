# Experimental shared-policy PPO

For deterministic/stochastic evaluation, isolated policy seeds, multiple rollouts,
and fresh checkpoint initialization, see [EXECUTION_MODES.md](EXECUTION_MODES.md).
Deterministic evaluation and existing resume behavior remain the defaults.

This is an isolated behavioral-learning experiment. All new implementation,
configuration, dependencies, tests, and run artifacts live under `rl/`. It does
not import an existing rule controller and does not modify official simulation
files. There are no hard-coded foraging, fleeing, spawning, tree-camping, pairing,
or decoy rules. No performance improvement is claimed before training/evaluation.

## Windows PowerShell commands

Run these from `survival-simulator`. Use Python 3.12 for the repository's pinned
simulator dependencies. Commands use the environment executable directly, so
PowerShell activation/execution-policy changes are unnecessary.

### 1. Dependencies

```powershell
python -m venv rl\.venv
rl\.venv\Scripts\python.exe -m pip install --upgrade pip
rl\.venv\Scripts\python.exe -m pip install -r rl\requirements.txt
rl\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For NVIDIA acceleration, install the Windows CUDA wheel appropriate for your
driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/).
For example, on a machine supporting its CUDA 12.8 wheel:

```powershell
rl\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use, the explicit alternative is:

```powershell
rl\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

`--device auto` uses CUDA only when the installed wheel and hardware expose it.
The assistant's test environment installed CPU PyTorch; replace that wheel if
you want to train on an NVIDIA GPU. On Linux use `python3 -m venv rl/.venv` and
`rl/.venv/bin/python`; the Python module commands are otherwise identical.

### 2. Quick smoke test (no real simulation or training)

```powershell
rl\.venv\Scripts\python.exe -m unittest discover -s rl/tests -v
rl\.venv\Scripts\python.exe -m rl.train --help
```

Tests use tiny tensor fixtures and a scripted fake environment. They do not
instantiate `SimulationCore`, collect real episodes, benchmark workers, run PPO
updates, or evaluate trained policies. The checkpoint test performs one synthetic
critic optimizer step solely to verify optimizer-state serialization.

### 3. Worker throughput benchmark (user-run)

```powershell
rl\.venv\Scripts\python.exe -m rl.benchmark --workers 1 2 4 6 8 10 --seconds 60
```

This uses random actions with the official simulator, no neural model or training.
It reports aggregate ticks/sec, separate initial startup time, and sampled total
parent+worker resident memory when psutil is available. Episode-reset costs after
startup count toward throughput. Choose your training worker count from these
results; more workers can be slower or exceed memory. The official environment
still creates its offscreen terrain surfaces in headless mode; bypassing those
would alter RNG consumption and is intentionally not attempted.

### 4. Six-hour training run (user-run)

```powershell
rl\.venv\Scripts\python.exe -m rl.train --config rl\config_pc.json --workers 4 --max-hours 6 --device auto --output-dir rl/runs/pc
```

Replace `--workers 4` with your measured choice. Other settings can be overridden
in a JSON config; see `Config` in `config.py`. Resume with the same worker count:

```powershell
rl\.venv\Scripts\python.exe -m rl.train --resume rl/runs/pc/latest.pt --max-hours 6
```

`--max-hours` is the budget for this invocation, including startup, validation,
and updates. A shutdown margin (30 seconds by default, smaller for short runs)
reserves time for final checkpointing and process cleanup. Parent collection and
validation worker waits honor the deadline; PPO checks it between minibatches.
Extremely slow filesystem checkpoint writes can still exceed that margin.
Ctrl+C/SIGTERM triggers final saving and bounded worker shutdown where possible.

### 5. Evaluate the best checkpoint (user-run)

```powershell
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/pc/best_validation.pt --output rl/runs/pc/evaluation.csv
```

Default evaluation is at the full 3,000-second horizon on five reserved seeds.
`best_validation.pt` is created only after a complete scheduled validation; it
will not exist after a very short smoke training session. Evaluate `latest.pt`
explicitly if no complete validation has happened yet. For fresh final test seeds:

```powershell
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/pc/best_validation.pt --seeds 2147483748 2147483749 2147483750 --output rl/runs/pc/heldout.csv
```

### 6. Render one evaluation episode (user-run)

```powershell
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/pc/best_validation.pt --seeds 2147483648 --render --output rl/runs/pc/visual-evaluation.csv
```

Rendering requires exactly one seed and is intended only for inspection. It uses
the existing simulator renderer, restores lazy vision-cache changes, and never
feeds wall-clock frame time into the physics. Closing the window marks the run
incomplete. Bulk training and throughput runs never initialize a display.

## Policy and observation design

One shared actor processes each agent's 103-feature vector: 91 local features
and 12 global features. Local state includes normalized energy, energy fraction,
age, movement/sensing traits, max energy, biome one-hot, and simulation progress.
It encodes nearest 3 fruits, 3 trees, 2 predators, 3 agents, and 3 edges, each
with a presence mask. Angles use sine/cosine; nearest-edge encoding uses the
closest point, segment orientation, and length. Missing blocks are all zeros.

Global context uses only current public agent states and five seconds of public
population-ID history: population, mean/median/min energy fraction, mean age,
young/old fractions, recent observed births/deaths, count of predator observations,
full-horizon progress, and the current curriculum horizon fraction. Predator
observations may duplicate the same predator; the feature is not a hidden census.
IDs are used only for action routing and temporal bookkeeping, not actor inputs.

The team critic takes the 12-feature global summary. It predicts one team value
per environment decision. This avoids artificial termination/value discontinuities
when one agent dies while the species survives. Both actor and critic default to
two 256-unit Tanh layers. CUDA inference and updates live only in the parent
process; simulator workers never own a neural model or create GPU contexts.

The actor emits three independent Beta variables and a Bernoulli spawn choice.
Beta variables map to move fraction and two angles by affine transforms.
Movement is `fraction * sprint_speed`; angles are in [-pi, pi]. Log densities
include the two 2*pi affine Jacobians; there is no Gaussian action clipping.
Deterministic evaluation uses Beta means and Bernoulli probability >=0.5.

## Five-tick transitions and population weighting

Each regular decision repeats for five official 0.1-second ticks. Movement is
repeated; total selected rotation is split evenly over those ticks. Spawn is
attempted only at the first macro-decision tick. Extinction stops immediately.

If a newborn appears during the repeat, the worker requests another centralized
inference batch. That newborn acts starting on its next simulator tick, using
the same actor. Its chosen total turn is divided over the remaining ticks; its
spawn dimension is masked until the next macro decision. The masked Bernoulli
contributes neither log probability nor entropy. No baby uses a fallback rule.

All acting agents, including newborn samples within the interval, share its
team reward and team GAE target. Conditional newborn actions are part of that
same macro-transition. If there are N actor samples, their actor, value, and
entropy losses each have weight 1/N. We normalize advantages over team decisions
before expanding them to agents. Minibatches use a fixed whole-rollout weight
normalizer, so random minibatches estimate equal-environment-decision loss rather
than implicitly favoring large populations. GAE never links trajectories by
individual agent ID.

Defaults: gamma=0.9995 per 0.5-second decision, lambda=0.95, PPO clip=0.2,
6 epochs, value coefficient=0.5, entropy coefficient=0.01, gradient norm=0.5,
target KL=0.03. An early extinction uses gamma^(actual_ticks/5). Curriculum
horizons are episodic task terminals with a survival bonus, not bootstrapped
time-limit truncations.

## Rewards, curriculum, validation, and checkpoints

Training reward is official score delta, plus configurable extinction penalty
(-5 default), horizon bonus (+5 default), and optional energy potential shaping
(disabled by default). Optional shaping is `coefficient * (discount * Phi(next)
- Phi(current))`, where Phi is mean energy fraction and terminal Phi is zero.
The coefficient is capped at 0.1. There are no explicit population, birth,
trait, movement-direction, food-proximity, or terrain rewards. Official and
training rewards are logged separately.

Only episode horizon changes: 300 -> 900 -> 1800 -> 3000 seconds. Promotion
requires at least 70% horizon completion over 20 recent episodes at the current
horizon, or 200,000 environment decisions / 5,400 wall-clock seconds in that
stage. A promoted horizon applies on future resets; running episodes keep their
original task. No predator mechanics, starting environment, or time step changes.

Training seeds are `base + episode_index * workers + worker_id`, strictly below
2**31. Fixed validation seeds are 2147483648 through 2147483652 and never enter
that training stream. Validation runs approximately every 30 minutes, with one
additional CPU simulator worker and deterministic centralized inference.
Checkpoint ranking is lexicographic: completion rate, median survival, official
mean score. Only complete five-seed sweeps qualify. Rankings are maintained per
curriculum horizon so unlike tasks are not compared; `best_validation.pt` tracks
the latest stage with completed validation, with stage-specific copies retained.

About every 15 minutes and on exit, saves include `latest.pt` and a timestamped
checkpoint. Each contains separate actor and critic weights, optimizer, counters,
curriculum state, config, next per-worker episode indices, and Python/NumPy/Torch
CPU/CUDA RNG state where available. Writes use atomic replacement. Load only
trusted local checkpoints, since practical RNG metadata uses Python serialization.

Resume restores learning state but starts fresh simulator episodes with unused
seeds. It does not serialize official simulator objects or unfinished on-policy
rollout buffers, so it is not a bit-identical continuation of the interrupted
trajectory. Worker count and base seed must remain fixed. Logs append on resume;
use a fresh output directory for an unrelated experiment.

## Metrics and validation limits

`training.csv` records wall time, ticks, decision/update counts, ticks/sec,
episodes, official/shaped rewards, completed-episode survival, current population
and energy statistics, cumulative births/deaths, loss/entropy/KL/clip statistics,
explained variance, and curriculum stage. `episodes.csv` and `validation.csv`
contain episode-level score, survival, completion, final/peak/min population,
observed births/deaths, mean/median energy, and wall time. TensorBoard is used
when its optional dependency is installed.

Birth/death counts use public IDs observed at each underlying tick. A newborn
born and killed entirely within one tick may not appear. Predator-related deaths
and isolated fruit-score contribution remain blank: step-boundary observations
cannot reliably separate those causes. No inferred cause is silently reported
as a measured fact. Energy summaries in training.csv describe the current agents
at rollout end; episode summaries aggregate observed agent-ticks.

Unit tests cover observation dimensions/masks, distributions/log probabilities,
action repeat/spawn masking, newborns, extinction, score/shaping separation,
worker seed partition, equal-team loss weights, centralized collection, GAE,
checkpoint/RNG/optimizer round-trip, curriculum, validation ranking, and SHA-256
protection of the existing simulator/controller files. The protected-file
snapshot is tied to this repository revision; deliberate upstream simulator
updates require reviewing and regenerating it.

All 25 lightweight tests passed on native Windows with Python 3.12.7 and CPU
PyTorch 2.14.0. This includes real spawn-process handshakes (no simulator reset),
confirmation that workers did not import torch, and a mocked Ctrl+C final-save
test. Existing simulator/controller hashes also matched. CLI help and Python
syntax checks passed.

Only lightweight tests and static checks were run during implementation. Real
Windows worker throughput, CUDA operation, trained behavior, and survival quality
remain to be measured by the user. This implementation has no trained checkpoint.

See [robust stochastic validation](ROBUST_VALIDATION.md) for checkpoint ranking, RNG isolation, reporting, and PC/HPC commands.
