"""Atomic hierarchical checkpoints with a protected-source compatibility stamp."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import random
import numpy as np
import torch
from rl_hier.config import protect_output

FORMAT = "rl_hier_v2_public103_categorical6_beta2_bernoulli"


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = []
    for directory in ("rl", "src", "rl_hier"):
        for folder, directories, files in os.walk(root / directory):
            directories[:] = sorted(d for d in directories if not d.startswith(".")
                                     and d not in ("tests", "runs", "__pycache__"))
            paths.extend(Path(folder) / name for name in files if name.endswith(".py"))
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, policy, optimizer, config, state):
    path = Path(path)
    protect_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=FORMAT, source_digest=source_digest(),
                   actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
                   optimizer=optimizer.state_dict() if optimizer else None,
                   config=config.to_dict(), state=state,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, policy=None, optimizer=None, restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("Not a compatible hierarchical checkpoint; baseline initialization is forbidden")
    if payload.get("source_digest") != source_digest():
        raise ValueError("Experiment or baseline/simulator source changed since checkpoint creation")
    if policy is not None:
        policy.actor.load_state_dict(payload["actor"])
        policy.critic.load_state_dict(payload["critic"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count differs from resume checkpoint")
            torch.cuda.set_rng_state_all(rng["cuda"])
    return payload


def save_periodic(directory, policy, optimizer, config, state):
    directory = Path(directory)
    save_checkpoint(directory / "latest.pt", policy, optimizer, config, state)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    save_checkpoint(directory / f"checkpoint_step{state['training_step']}_{stamp}.pt", policy, optimizer, config, state)
