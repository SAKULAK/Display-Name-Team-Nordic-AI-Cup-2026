# Experimental hierarchical PPO

This experiment tests whether learned behavioral primitives improve the speed of
learning long-horizon team survival. It starts from scratch. No learning or
survival improvement is claimed before the user runs the experiment.

All implementation and default run artifacts live in `rl_hier/`. The existing
`rl/` and official `src/` are read-only dependencies. There are no simulator
patches, monkeypatches, reward heuristics, or pretrained baseline initialization.

## Run on the PC

Run from `survival-simulator`, using a Python environment with the existing
simulator and PyTorch dependencies. The existing `rl/.venv` can be used as an
interpreter without changing its packages or baseline files. `device=auto` uses
CUDA if that environment exposes it. For a separate environment, install
`rl_hier/requirements.txt`; no dependency installation is performed by the code.

Exact overnight command using the existing Windows environment:

```powershell
cd C:\Users\takac\nordicai\Display-Name-Team-Nordic-AI-Cup-2026\survival-simulator
rl\.venv\Scripts\python.exe -B -m rl_hier.train --config rl_hier/config_pc_overnight.json
```

With the intended environment's Python on PATH, the equivalent is:

```powershell
python -B -m rl_hier.train --config rl_hier/config_pc_overnight.json
```

`-B` avoids writing import caches into the read-only baseline and simulator trees.
The PC config uses 6 simulator workers, automatic device selection, 128 rollout
macrosteps per worker, 1,024-row minibatches, 15-minute checkpoints, hourly
validation, 3 stochastic rollouts on each of the 5 reserved seeds, and an 8-hour
invocation budget. Validation uses one additional simulator worker while the
training workers remain paused. Rollout length matches the baseline PC config.
No worker throughput or memory benchmark has been
run for this implementation.

Resume for another eight hours, with the original worker count and seed streams:

```powershell
rl\.venv\Scripts\python.exe -B -m rl_hier.train --resume rl_hier/runs/pc_overnight/latest.pt --max-hours 8
```

Evaluate at 3,000 seconds with the same reserved seeds and stochastic protocol:

```powershell
rl\.venv\Scripts\python.exe -B -m rl_hier.evaluate --checkpoint rl_hier/runs/pc_overnight/best_validation.pt --horizon 3000 --policy-mode stochastic --policy-rollouts 3 --policy-seed 20260918 --output rl_hier/runs/pc_overnight/evaluation_stochastic.csv
```

Deterministic evaluation:

```powershell
rl\.venv\Scripts\python.exe -B -m rl_hier.evaluate --checkpoint rl_hier/runs/pc_overnight/best_validation.pt --horizon 3000 --policy-mode deterministic --output rl_hier/runs/pc_overnight/evaluation_deterministic.csv
```

Use `latest.pt` if no full validation sweep has finished yet. Optional `--seeds`
accepts an explicit evaluation seed list. `--render` requires exactly one seed.
Evaluate baseline and hierarchy using the same horizon, environment seeds,
policy mode, and rollout count. Compare wall time as well as simulator ticks;
primitive execution and snapshotting add overhead. Reserved validation seeds
are model-selection data; use a separate seed list for final held-out results.

Training supports `--workers`, `--max-hours`, `--device`, `--output-dir`,
`--learning-rate`, `--validation-seconds`, `--rollout-steps`, and `--base-seed`.
JSON sets other configuration fields. Fresh runs require an empty output
directory. Output paths inside `rl/` or `src/` are rejected. There is no
`--init-from` option, and low-level PPO checkpoints are rejected.

## Architecture and observation boundary

One shared actor controls every agent, including newborns. Actor and centralized
team critic each use two 256-unit Tanh hidden layers. The actor has six
Categorical logits, four Beta concentration outputs (two alpha/beta pairs), and
one Bernoulli spawn logit. Concentrations are `softplus(output) + 1`.

The actor receives the baseline's exact **103-element public observation**:
91 local public features concatenated with the same 12-element public team
summary. Normalization, ordering, nearest-entity blocks, and missing-entity masks
are unchanged. Local features include self traits, energy, age, biome, simulation
progress, and nearest 3 fruits, 3 trees, 2 predators, 3 teammates, and 3 relative
obstacle edges. No absolute coordinates, global team positions, hidden predator
state, or other privileged information is added. Agent IDs route actions only.

The critic receives the same 12-element public team summary as the baseline:
population and energy/age summaries, recent births/deaths, aggregated perceived
predator counts, progress, and horizon. As in the baseline, this public summary
is also included in the actor input. Actor observations, critic information,
and the PC rollout length therefore match the baseline; action abstraction is
the intended experimental difference.

## Learned actions and primitive execution

Each sample is stored as `[mode, movement_intensity, residual_unit, spawn]`.
The two Beta samples remain in their original unit coordinates. Execution maps
`steering_residual = 2 * residual_unit - 1`; it never reconstructs samples from
physical movement requests. The actor learns when to choose each mode, its
movement intensity, its residual steering, and whether to attempt reproduction.
Continuous parameters are shared across modes, rather than six independent
parameter sets.

| Mode | Scripted execution of the learned choice |
| --- | --- |
| EXPLORE | Intensity times sprint distance; residual controls relative heading over the full `[-pi, pi]` range and gradual turning. The physical heading carries continuity; there is no target rule or additional memory. |
| FORAGE | Approach the nearest perceived fruit, otherwise nearest perceived tree. Cap approach distance at target distance. Residual can offset heading by up to 60 degrees. |
| CAMP | Use the nearest perceived tree, otherwise fruit. Approach toward a 20-unit stand-off. Within 30 units, use slow tangential movement at 15% intensity scale; residual chooses orbit side and perturbs heading. |
| EVADE | Move away from the nearest currently perceived predator, with learned intensity and up to 60 degrees of steering offset. |
| REGROUP | Approach the nearest currently perceived teammate, with learned intensity and steering offset. |
| CONSERVE | Cap full-intensity distance at `min(0.2 * sprint_speed, 0.5 * speed)` and use slow residual-controlled scanning. PPO decides when to select it. |

FORAGE, CAMP, EVADE, and REGROUP fall back exactly to EXPLORE when no relevant
target is currently perceived. Targets are reacquired from each tick's public
entity list; CAMP does not remember an invisible resource or use an absolute
anchor. Its vicinity distances are execution parameters, not energy/age-based
strategy rules. No primitive decides when to spawn, conserve, flee, or breed.
There are no carrying-capacity, age-replacement, or predator-exhaustion rules.

The policy chooses once every five official `dt=0.1` ticks (0.5 seconds).
Mode and continuous samples remain fixed through that interval. The primitive
recomputes its movement bearing from fresh public observations each tick. Each
tick's turn request is its chosen relative turn divided by five (or by the
newborn's remaining tick count). Targeted turns therefore use feedback rather
than promising one fixed net rotation across the entire interval.

Actions are official `ActionRequest` objects. Movement distance uses the same
`intensity * sprint_speed` scale as the baseline. **The official simulator
applies this distance per tick, without multiplying by `dt`.** Its movement
costs, biome modifiers, collision handling, reproductive energy checks,
predators, resources, and aging remain untouched.

Spawn is attempted only on the first tick of a macrodecision. A newborn requests
shared-policy inference immediately for its remaining ticks. Its spawn head is
masked to zero until the next regular decision, contributing neither spawn
log probability nor spawn entropy during the masked interval.

## PPO, reward, and curriculum

Joint log probability is Categorical mode log probability plus both Beta log
densities minus `log(2)` for the residual transform, plus the eligible Bernoulli
spawn log probability. Joint entropy includes the corresponding three heads
and residual Jacobian. PPO ratios compare exactly the original stored samples;
fallback behavior does not relabel a sampled mode. Deterministic execution uses
Categorical argmax, Beta means, and Bernoulli probability `>= 0.5`. Stochastic
execution samples the same distributions as training.

The unchanged baseline collector and PPO update provide centralized team values,
team reward, team GAE, and population-neutral losses. Each team macrostep has
total sample weight one, split across its acting agents including newborns.
Advantages are normalized over team steps before expansion. The fixed rollout
weight denominator avoids population-dependent minibatch normalization.

Defaults remain gamma 0.9995 per macrostep, lambda 0.95, clip 0.2, 6 epochs,
minibatch 1,024, value coefficient 0.5, entropy coefficient 0.01, gradient norm
0.5, target KL 0.03, and Adam learning rate 0.0003 with epsilon 0.00001. As in the
baseline update, the value-error expression itself includes a factor of 0.5.
Early terminal steps use `gamma ** (actual_ticks / 5)`.

Reward is fixed to:

```text
official score delta - 5 if extinct
official score delta + 5 if the horizon is completed alive
official score delta otherwise
```

Energy shaping is locked to zero. There are no added population, spawn, eating,
mode-selection, predator-distance, or survival-shaping rewards. Official score
already includes elapsed time, fruit energy contribution, and predation
penalties; those official terms remain intact.

The baseline curriculum is reused unchanged: 300, 900, 1,800, and 3,000 seconds.
Promotion requires 70% completion over 20 recent current-horizon episodes, or
200,000 team decisions / 5,400 stage wall-clock seconds. New horizons apply on
worker reset; active episodes keep their horizon. The baseline wall-clock
accounting includes time spent on updates and validation between collections.

## Checkpoints and resume

Atomic `latest.pt`, `checkpoint_stepN_TIMESTAMP.pt`, `best_validation.pt`, and
stage-specific best checkpoints contain actor/critic weights, optimizer slots,
Python/NumPy/Torch CPU/CUDA RNG states, configuration, curriculum history and
timers, counters, per-worker next episode indices, validation scheduling state,
and compressed live worker snapshots. Simulator RNG aliases, observations,
encoder history, episode metrics, spatial grids, objects, and render surfaces
are serialized without editing their classes. Surfaces use a private pickler
with an RGBA reducer; it does not change global serialization registrations.

Snapshots are taken at completed PPO update boundaries, where no pending
on-policy rollout buffer must be restored. Intervals are therefore approximate:
a slow rollout/update can delay a scheduled checkpoint. Ctrl+C/SIGTERM requests
a stop after the current partial rollout has been processed, then saves a
consistent boundary. If a worker times out or collection/update fails midway,
the trainer preserves the last durable checkpoint instead of pairing a partial
learner update with inconsistent worker state. Work since that checkpoint can
be lost. Logs written after that saved step can remain in CSV/TensorBoard;
on recovery, use training-step counters to identify repeated log regions, or
resume to a fresh `--output-dir`.

Resume restores live episodes rather than resetting them to unused seeds. An
initial checkpoint made before any worker reset legitimately has no live episode
and starts the initial seed stream. Resume keeps training semantics, worker
count, seed base, and rollout size fixed; runtime budgets, output, learning rate,
device, and validation protocol can be overridden. `--max-hours` is a fresh
budget for each invocation. Best ranks are separated by horizon and validation
protocol. A source SHA-256 stamp rejects changed experiment, baseline, or
simulator Python code. Use the same dependencies for resume. Checkpoints contain
Python serialization and must be trusted local files.

The 103-input actor uses checkpoint format v2. Earlier hierarchical checkpoints
with a 91-input actor are incompatible and are rejected; start a fresh run.

This preserves learning state and simulator episode state, but is **not a claim
of bit-identical long-horizon trajectories across processes/devices**. The
official simulator uses identity-hashed objects in sets, whose iteration order
can change after deserialization, and CUDA/platform behavior can differ. Making
those mechanics deterministic would require changes outside this experiment's
authorized scope. Full-size snapshot cost and long-duration resume behavior have
not been benchmarked. Surface snapshots and full world state increase checkpoint
size; snapshots are fetched one worker at a time to limit transient memory.

## Validation and diagnostics

Training seeds are `base_seed + episode_index * workers + worker_id`, below
`2**31`. Reserved validation seeds are 2147483648 through 2147483652. Stochastic
policy seeds are `seed_base + environment_index * rollouts + rollout_index`,
with default seed base **20260918**, matching the baseline robust-validation
protocol. Evaluation uses the same schedule and CLI seed default. Each stochastic execution has a private
Torch CPU/CUDA sampling stream; the full validation scope also restores Python,
NumPy, and Torch states, including exceptional exits. Training workers remain
paused and their RNGs are independent.

Only an exactly complete sweep is eligible to replace a best checkpoint.
Missing, interrupted, duplicate, or wrong-horizon entries cannot qualify.
Both policy modes rank higher lexicographically by:

```text
(completion_rate, q25_survival, median_survival, mean_official_score)
```

`validation.csv`, `validation_summary.csv`, and `validation_by_seed.csv` record
the protocol and episode/aggregate outcomes, including completion rate,
mean/median/q25/min/max survival, and mean/median official score. Best snapshots
are retained per curriculum stage; the generic best file represents the most
recent stage/protocol with an improved full sweep, not a cross-horizon ranking.

`training.csv` retains baseline reward, score, population, energy, birth/death,
survival, throughput, PPO loss/KL/entropy, value-fit, and curriculum diagnostics.
It adds raw per-agent and equal-team-step mode fractions, Categorical entropy,
eligible spawn probability/action frequency, and mean/std/min/max intensity and
physical residual. Probabilities and entropy describe the collecting policy
before its PPO update. `modes_by_stage.csv` attributes the same hierarchical
metrics to the actual episode horizon, including workers finishing an earlier
stage. Optional TensorBoard receives training metrics. `episodes.csv` retains
the baseline episode reports. No diagnostic is a reward or actor input.

## Lightweight verification

```powershell
rl\.venv\Scripts\python.exe -B -m unittest discover -s rl_hier/tests -v
rl\.venv\Scripts\python.exe -B -m rl_hier.train --help
rl\.venv\Scripts\python.exe -B -m rl_hier.evaluate --help
```

Tests use small tensors and scripted fake workers. One test builds a tiny
160-by-160 official world fixture and executes one tick on each of the original
and restored branches. There is a worker ping test with no environment reset,
and one synthetic critic optimizer step solely to check nonempty Adam-state
serialization. No PPO training, full simulation episodes, throughput benchmarks,
or validation sweeps are run by these tests. CUDA RNG isolation is checked when
CUDA is available. Long-run strategy quality, full-size memory cost, and robust
survival remain user-run experimental questions.
