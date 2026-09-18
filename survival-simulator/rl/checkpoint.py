"""Atomic trusted-local PyTorch checkpoints and practical RNG restoration."""
from datetime import datetime, timezone
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, policy, optimizer, config, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
                   optimizer=optimizer.state_dict() if optimizer else None,
                   config=config.to_dict(), state=state,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, policy=None, optimizer=None, restore_rng=False):
    # Contains Python/NumPy RNG metadata; load only checkpoints you trust.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
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
        if torch.cuda.is_available() and rng["cuda"] is not None and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return payload


def save_periodic(directory, policy, optimizer, config, state):
    directory = Path(directory)
    save_checkpoint(directory / "latest.pt", policy, optimizer, config, state)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    save_checkpoint(directory / f"checkpoint_{stamp}.pt", policy, optimizer, config, state)


def initialize_weights(path, policy):
    """Load only weights; preflight BOTH modules before modifying either one."""
    payload = load_checkpoint(path)
    for name in ("actor", "critic"):
        expected = getattr(policy, name).state_dict()
        supplied = payload.get(name, {})
        if (set(expected) != set(supplied)
                or any(expected[k].shape != supplied[k].shape for k in expected)):
            raise ValueError(f"Incompatible {name} architecture in init-from checkpoint: {path}")
    policy.actor.load_state_dict(payload["actor"], strict=True)
    policy.critic.load_state_dict(payload["critic"], strict=True)
    old = payload.get("state", {})
    return dict(initialized_from=str(Path(path).resolve()),
                initialized_from_training_step=old.get("training_step"),
                initialized_from_environment_ticks=old.get("environment_ticks", old.get("environment_steps")))
