# Robust stochastic validation

Old configs default to deterministic validation with one rollout and the existing
(completion rate, median survival, mean official score) rank. Repeated deterministic
validation rollouts are rejected. No actor, action distributions, observations,
rewards, physics, or PPO updates are changed.

`config_hpc_finetune.json` uses `validation_policy_mode: "stochastic"`,
`validation_policy_rollouts: 3`, and `validation_policy_seed_base: 20260918`.
It keeps the existing PPO settings except learning rate 1e-4, uses five reserved
validation environment seeds, and omits workers. Supply workers from CLI and use
`--init-from` and `--start-horizon 3000` as below. `config_pc.json` is unchanged.

For environment index i and rollout index j (both zero based), the policy seed is
`base + i * rollouts + j`. Every rollout resets the environment to the same seed
for that environment index. The existing PolicyExecution helper samples the
learned Beta movement and Bernoulli spawn distributions without extra noise or
thresholding. It installs private Torch RNG states around action calls, including
newborn actions. The whole sweep additionally saves/restores Python, NumPy, CPU
Torch and initialized CUDA RNG states, including exceptional exits. Validation
uses a separate worker; live training workers and collector episode indices are
never passed to it. Reproducibility assumes the same checkpoint, config, device,
and software stack; CPU and CUDA need not produce identical samples.

With multiple stochastic rollouts, higher lexicographic rank wins:
`(completion_rate, q25_survival, median_survival, mean_official_score)`.
Completion means reaching the configured horizon alive, not merely finishing an
evaluation episode. Extinction is a completed evaluation with horizon_reached=0.
For sorted survival values, q25 is linear interpolation at zero-based index
`(n-1)*0.25`; for [0,100,200,300] it is 75. Single stochastic rollouts retain the
three-part rank. All comparisons are scoped by horizon and validation protocol
(mode, count, seeds and policy seed base). Default deterministic validation keeps
historical horizon keys and three-part checkpoint ranks for resume compatibility.
Changing stochastic validation settings starts a separate ranking history.

Validation runs synchronously: training waits and sweeps cannot overlap.
`validation_seconds` remains 1800 in the example; `--validation-seconds` can
override it for short tests. Scheduling is checked between PPO updates, so this is
not an exact start-time guarantee. The existing global shutdown deadline applies
to the validation worker and newborn continuation requests. Completed episode
rows from a truncated sweep are retained; an unfinished episode is discarded.
Only exact full plan coverage is eligible for summaries or best checkpoints.
A short job may produce latest.pt without finishing a validation sweep.
Five seeds with three rollouts means 15 episodes; five rollouts means 25 episodes,
about 67% more episodes. Actual cost depends on survival duration.

`validation.csv` preserves historical columns and adds validation_step (the
training update number), environment_seed, rollout_index, policy_mode and
policy_seed. Legacy rows are backfilled with deterministic mode, rollout 0, and
blank policy seed; their existing metrics remain intact. `validation_summary.csv`
has one row per full sweep, including training counters, horizon/protocol, all
survival and score statistics and the rank. `validation_by_seed.csv` contains
those statistics per environment seed. Console output shows sweep statistics and
per-seed medians. Best checkpoint state contains the validation protocol, rank,
completion rate, q25/median/mean survival and mean official score, while retaining
existing state fields. Both best_validation.pt and the stage-specific best file
are saved. Partial sweeps never update either best file or the best-rank state.

Run the following from the survival-simulator directory. B, C and D are for the
user to execute; implementation verification uses only A with fake environments
and tiny tensor fixtures. Use fresh output paths for new experiments.

A. Lightweight tests (PowerShell):

```powershell
rl\.venv\Scripts\python.exe -m unittest discover -s rl/tests -v
```

B. Manual evaluation of the existing best, five environment seeds x five rollouts:

```powershell
rl\.venv\Scripts\python.exe -m rl.evaluate --checkpoint rl/runs/sanity/best_validation.pt --seeds 2147483648 2147483649 2147483650 2147483651 2147483652 --horizon 3000 --policy-mode stochastic --policy-seed 20260918 --policy-rollouts 5 --output rl/runs/manual_robust_25.csv
```

The manual evaluator preserves its existing seed mapping: base+j is reused across
environment seeds. Automatic training validation uses base+i*N+j. Sampling semantics
are identical, but these are different sampling plans; logged policy seeds make
that distinction explicit. The manual command reports q25 and the robust rank.

C. Short PC init-from test (six-minute wall budget, validation scheduled after
60 seconds, full horizon, six workers, three stochastic rollouts per seed):

```powershell
rl\.venv\Scripts\python.exe -m rl.train --config rl/config_hpc_finetune.json --init-from rl/runs/sanity/best_validation.pt --start-horizon 3000 --workers 6 --learning-rate 0.0001 --max-hours 0.1 --validation-seconds 60 --output-dir rl/runs/pc_robust_short
```

D. Future HPC template (Bash, after activating the HPC Python environment).
Set WORKERS to the worker count chosen for your allocation before running:

```bash
python -m rl.train --config rl/config_hpc_finetune.json --init-from rl/runs/sanity/best_validation.pt --start-horizon 3000 --workers "${WORKERS:?Set WORKERS for your allocation}" --learning-rate 0.0001 --max-hours 6 --output-dir rl/runs/hpc_robust_finetune
```

`--init-from` loads weights into a fresh experiment; `--resume` restores existing
training state and retains its existing worker/base-seed constraints. Neither
mode's behavior has been changed by the validation extension.
