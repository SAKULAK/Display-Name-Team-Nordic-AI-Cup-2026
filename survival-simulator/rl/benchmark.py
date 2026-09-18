"""User-run headless worker throughput benchmark; random actions, no learning."""
import argparse
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--output", type=Path, default=Path("rl/runs/throughput.csv"))
    args = parser.parse_args(argv)
    if min(args.workers) < 1 or args.seconds <= 0:
        parser.error("workers and seconds must be positive")
    import numpy as np
    from rl.config import Config, worker_seed
    from rl.workers import WorkerPool
    from rl.metrics import CSVLogger
    log = CSVLogger(args.output)
    for count in args.workers:
        config = Config(workers=count)
        rngs = [np.random.default_rng(worker_seed(config.base_seed, i, 0, count)) for i in range(count)]
        episodes = [0] * count
        def resets(worker_ids):
            commands = {}
            for i in worker_ids:
                commands[i] = ("reset", dict(seed=worker_seed(config.base_seed, i, episodes[i], count), horizon=3000.))
                episodes[i] += 1
            return commands
        def random_actions(i, packet, spawn_mask=1):
            actions = rngs[i].random((len(packet["ids"]), 4)).astype(np.float32)
            actions[:, 3] = (actions[:, 3] > .5) * spawn_mask
            return dict(zip(packet["ids"], actions))
        with WorkerPool(config) as pool:
            startup = time.monotonic()
            packets = pool.exchange(resets(range(count)))
            startup_seconds = time.monotonic() - startup
            started = time.monotonic()
            ticks = decisions = 0
            memory_peak = None
            while time.monotonic() - started < args.seconds:
                responses = pool.exchange({i: ("begin", random_actions(i, packets[i])) for i in range(count)})
                completed = {}
                while responses:
                    commands = {}
                    for i, r in responses.items():
                        if r["kind"] == "transition":
                            completed[i] = r
                        else:
                            commands[i] = ("continue", random_actions(i, r["packet"], r["spawn_mask"]))
                    responses = pool.exchange(commands) if commands else {}
                reset_ids = []
                for i, r in completed.items():
                    ticks += r["ticks"]
                    decisions += 1
                    packets[i] = r["packet"]
                    if r["done"]:
                        reset_ids.append(i)
                if reset_ids:
                    packets.update(pool.exchange(resets(reset_ids)))
                try:
                    import psutil
                    resident = psutil.Process().memory_info().rss
                    resident += sum(psutil.Process(p.pid).memory_info().rss for p in pool.processes)
                    memory_peak = max(memory_peak or 0, resident)
                except ImportError:
                    pass
            elapsed = time.monotonic() - started
            row = dict(workers=count, seconds=elapsed, startup_seconds=startup_seconds, environment_ticks=ticks,
                       decision_steps=decisions, ticks_per_second=ticks / elapsed,
                       process_rss_mib=memory_peak / 2**20 if memory_peak is not None else None)
            log.write(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
