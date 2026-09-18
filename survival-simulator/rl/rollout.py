"""Central batched inference and team-transition GAE, including newborn actions."""
import numpy as np
import torch
from rl.config import worker_seed


def gae(rewards, values, dones, discounts, bootstrap, lam):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.
    next_value = float(bootstrap)
    for t in reversed(range(len(rewards))):
        nonterminal = 1. - float(dones[t])
        delta = rewards[t] + discounts[t] * next_value * nonterminal - values[t]
        carry = delta + discounts[t] * lam * nonterminal * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def population_weights(n):
    if n < 1:
        raise ValueError("Each decision must contain at least one acting agent")
    return np.full(n, 1. / n, dtype=np.float32)


class Collector:
    def __init__(self, pool, policy, config, episode_indices=None):
        self.pool, self.policy, self.config = pool, policy, config
        self.episode_indices = list(episode_indices or [0] * config.workers)
        self.packets = {}

    def reset(self, worker_ids, horizon):
        commands = {}
        for wid in worker_ids:
            seed = worker_seed(self.config.base_seed, wid, self.episode_indices[wid], self.config.workers)
            self.episode_indices[wid] += 1
            commands[wid] = ("reset", dict(seed=seed, horizon=horizon))
        self.packets.update(self.pool.exchange(commands))

    def _infer(self, packets, masks):
        worker_ids = sorted(packets)
        counts = [len(packets[i]["ids"]) for i in worker_ids]
        observations = np.concatenate([packets[i]["observations"] for i in worker_ids])
        spawn_masks = np.concatenate([np.full(n, masks[i], np.float32) for i, n in zip(worker_ids, counts)])
        actions, log_probs = self.policy.act(observations, spawn_mask=spawn_masks)
        results = {}
        offset = 0
        for wid, count in zip(worker_ids, counts):
            sl = slice(offset, offset + count)
            results[wid] = dict(observations=observations[sl], actions=actions[sl],
                                log_probs=log_probs[sl], spawn_masks=spawn_masks[sl],
                                action_map=dict(zip(packets[wid]["ids"], actions[sl])))
            offset += count
        return results

    @torch.no_grad()
    def values(self):
        contexts = np.stack([self.packets[i]["context"] for i in range(self.config.workers)])
        return self.policy.value(torch.as_tensor(contexts, dtype=torch.float32, device=self.policy.device)).cpu().numpy()

    def step(self, horizon):
        worker_ids = list(range(self.config.workers))
        starting_values = self.values()
        contexts = {i: self.packets[i]["context"] for i in worker_ids}
        inferred = self._infer(self.packets, dict.fromkeys(worker_ids, 1.))
        records = {i: [inferred[i]] for i in worker_ids}
        responses = self.pool.exchange({i: ("begin", inferred[i]["action_map"]) for i in worker_ids})
        completed = {}
        while responses:
            waiting = {}
            masks = {}
            for i, response in responses.items():
                if response["kind"] == "transition":
                    completed[i] = response
                else:
                    waiting[i] = response["packet"]
                    masks[i] = response["spawn_mask"]
            if not waiting:
                break
            inferred = self._infer(waiting, masks)
            for i in inferred:
                records[i].append(inferred[i])
            responses = self.pool.exchange({i: ("continue", inferred[i]["action_map"]) for i in inferred})
        transitions, episodes, reset_ids = [], [], []
        for i in worker_ids:
            result = completed[i]
            self.packets[i] = result["packet"]
            actors = {key: np.concatenate([r[key] for r in records[i]])
                      for key in ("observations", "actions", "log_probs", "spawn_masks")}
            transitions.append(dict(worker=i, actors=actors, context=contexts[i], value=float(starting_values[i]),
                                    reward=result["reward"], official_reward=result["official_reward"],
                                    done=result["done"], discount=result["discount"], ticks=result["ticks"],
                                    births=result["births"], deaths=result["deaths"], peak_population=result["peak_population"]))
            if result["done"]:
                episodes.append(result["episode"])
                reset_ids.append(i)
        if reset_ids:
            self.reset(reset_ids, horizon)
        return transitions, episodes


def prepare_batch(transitions, bootstrap_values, config):
    """GAE is computed once per team decision, never across agent IDs/deaths."""
    targets = {}
    for wid in range(config.workers):
        indices = [i for i, t in enumerate(transitions) if t["worker"] == wid]
        group = [transitions[i] for i in indices]
        adv, returns = gae([t["reward"] for t in group], [t["value"] for t in group],
                           [t["done"] for t in group], [t["discount"] for t in group],
                           bootstrap_values[wid], config.gae_lambda)
        for index, advantage, target in zip(indices, adv, returns):
            targets[index] = (advantage, target)
    team_adv = np.asarray([targets[i][0] for i in range(len(transitions))], dtype=np.float32)
    team_adv = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)
    batch = {key: [] for key in ("observations", "actions", "log_probs", "spawn_masks",
                                "contexts", "old_values", "advantages", "returns", "weights")}
    for i, t in enumerate(transitions):
        actors = t["actors"]
        n = len(actors["actions"])
        for key in actors:
            batch[key].append(actors[key])
        batch["contexts"].append(np.repeat(t["context"][None, :], n, axis=0))
        batch["old_values"].append(np.full(n, t["value"], np.float32))
        batch["advantages"].append(np.full(n, team_adv[i], np.float32))
        batch["returns"].append(np.full(n, targets[i][1], np.float32))
        # Newborn decisions join the same macro-transition. Every team step still sums to 1.
        batch["weights"].append(population_weights(n))
    return {key: np.concatenate(values) for key, values in batch.items()}
