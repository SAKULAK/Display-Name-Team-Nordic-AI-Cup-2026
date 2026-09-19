import random

import torch

from src.training.gym_env import OBS_DIM, decode_action, encode_observation
from src.training.model import ActorCritic
from src.training.train import CHECKPOINT_PATH
from src.utils.DTOs import ActionRequest

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

_model = ActorCritic(obs_dim=OBS_DIM).to(device)
try:
    _checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"No checkpoint found at '{CHECKPOINT_PATH}'. Run src/training/train.py first."
    ) from exc
_model.load_state_dict(_checkpoint["model"])
_model.eval()


def action_decision(observation_response: dict, rng: random.Random):
    """
    Trained-policy action selection for the agent.

    Uses the deterministic (mean/argmax) output of the trained ActorCritic rather
    than sampling, since at evaluation time we want the policy's best guess, not
    exploration noise.

    Args:
        observation_response (dict): Observation response from the environment
        rng (random.Random): Random number generator ( unused since I just wanted to replace the import 
        for the dummy_agent that uses it to insert my own agents. spywork)

    Returns:
        ActionRequest: Action decision
    """
    agent_id = observation_response["agent_id"]
    sprint_speed = observation_response["sprint_speed"]

    obs = encode_observation(observation_response)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mean, _, spawn_logit, _ = _model.forward(obs_tensor)

    continuous_action = mean.squeeze(0).cpu().numpy()
    spawn_action = 1.0 if spawn_logit.item() > 0.0 else 0.0

    move_distance, move_direction, turn_angle, spawn_agent = decode_action(
        continuous_action, spawn_action, sprint_speed
    )

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
