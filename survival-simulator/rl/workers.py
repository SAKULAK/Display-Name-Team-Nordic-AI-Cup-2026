"""Windows-spawn simulator workers. Workers never import torch or own a model."""
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback
import sys


def _worker(connection, config):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    # Imports happen after thread limits; no display initialization or GPU context.
    from rl.env_wrapper import SurvivalEnv
    env = SurvivalEnv(config)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "ping":
                result = dict(pid=os.getpid(), torch_loaded="torch" in sys.modules,
                              threads=os.environ["OMP_NUM_THREADS"])
            elif command == "reset":
                result = env.reset(**payload)
            elif command == "begin":
                result = env.begin(payload)
            elif command == "continue":
                result = env.continue_with(payload)
            else:
                raise ValueError(f"Unknown worker command {command}")
            connection.send(("ok", result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class WorkerPool:
    def __init__(self, config):
        self.config = config
        self.connections = []
        self.processes = []
        self.closed = False
        self.deadline = float("inf")
        context = mp.get_context("spawn")
        try:
            for _ in range(config.workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, config), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def exchange(self, commands):
        """Send to several workers; receive in readiness order with a bounded wait."""
        pending = {}
        for worker_id, (command, payload) in commands.items():
            connection = self.connections[worker_id]
            connection.send((command, payload))
            pending[connection] = worker_id
        results = {}
        deadline = min(self.deadline, time.monotonic() + self.config.worker_timeout_seconds)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Simulator worker timed out; workers will be shut down")
            ready = wait(list(pending), timeout=min(remaining, 1.))
            for connection in ready:
                worker_id = pending.pop(connection)
                status, result = connection.recv()
                if status != "ok":
                    raise RuntimeError(f"Simulator worker {worker_id} failed:\n{result}")
                results[worker_id] = result
        return results

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
        deadline = time.monotonic() + 5.
        for process in self.processes:
            process.join(max(0., deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(2)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
