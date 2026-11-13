"""Tabular Q-learning with reverse replay and periodic MPI synchronization.

The study object supplies seeded resets, aggregation, durable metrics,
checkpoint publication, and the epoch budget.
"""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import random
import sys
import time
import traceback

from mpi4py import MPI
import torch


def epsilon_for_epoch(initial_epsilon, epoch, decay_epochs):
    """Linearly decay exploration to zero, or keep it constant when disabled."""
    if epoch < 0 or decay_epochs < 0:
        raise ValueError("epoch and decay_epochs must be nonnegative")
    if decay_epochs == 0:
        return initial_epsilon
    if epoch >= decay_epochs:
        return 0.0
    return initial_epsilon * (1.0 - epoch / decay_epochs)


def train(
    env_fn,
    steps_per_epoch=4000,
    epsilon=0.03,
    epsilon_decay_epochs=0,
    *,
    study,
):
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if not isinstance(epsilon_decay_epochs, int) or epsilon_decay_epochs < 0:
        raise ValueError("epsilon_decay_epochs must be a nonnegative integer")

    current_threads = torch.get_num_threads()
    if current_threads > 1:
        torch.set_num_threads(max(current_threads // MPI.COMM_WORLD.Get_size(), 1))

    study.prepare()
    q_table = defaultdict(lambda: 0.0)
    v_table = defaultdict(lambda: 1)

    alpha = 0.05
    make_sparse = True
    gamma = 0.99

    env = env_fn()
    act_dim = env.action_space.n
    scalers = torch.tensor([30.0, 1, 1, 1, 1, 1, 1, 1])
    resolutions = [1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    n_procs = MPI.COMM_WORLD.Get_size()
    local_steps_per_epoch = int(steps_per_epoch / n_procs)

    def process_obs(obs):
        with torch.no_grad():
            normalized = torch.tanh(torch.tensor(obs) * scalers).tolist()
            quantized_s = tuple([
                round(x / resolutions[idx])
                for idx, x in enumerate(normalized)
            ])
            return obs, quantized_s

    epoch = 0
    effective_epsilon = epsilon_for_epoch(epsilon, epoch, epsilon_decay_epochs)
    trajectory = deque(maxlen=100000)
    total_epoch_episodes = 0
    total_epoch_return = 0
    total_epoch_steps = 0

    game_state = "INIT"
    while True:
        if game_state == "INIT":
            o, info = study.reset_environment(env)
            o, s = process_obs(o)

            total_episode_return = 0
            total_episode_steps = 0
            game_state = "RUNNING"

        elif game_state == "RUNNING":
            if random.uniform(0, 1) < effective_epsilon:
                a = random.randrange(0, 4)
            else:
                curr_q_max = float("-inf")
                a = None
                for _action in range(act_dim):
                    _q_val = q_table[s, _action]
                    if _q_val > curr_q_max:
                        curr_q_max = _q_val
                        a = _action

            v_table[s, a] += 1

            next_o, r, terminated, truncated, _ = env.step(a)
            next_o, s_prime = process_obs(next_o)
            total_episode_steps += 1
            stats_reward = r
            bellman_reward = r
            total_episode_return += stats_reward
            if make_sparse:
                if not terminated:
                    bellman_reward = 0

            study.on_step(stats_reward, bellman_reward, terminated, truncated)

            q_vals_update = [
                q_table[(s_prime, _action)] for _action in range(act_dim)
            ]
            q_table[(s, a)] += alpha * (
                bellman_reward
                + gamma * (1 - terminated) * max(q_vals_update)
                - q_table[(s, a)]
            )

            _exp = (o, s, a, bellman_reward, next_o, s_prime, terminated)
            trajectory.append(_exp)

            o = next_o
            s = s_prime

            # Time-limit truncation does not end a training episode.
            if terminated:
                game_state = "TERMINATE"

        elif game_state == "TERMINATE":
            total_epoch_episodes += 1
            total_epoch_return += total_episode_return
            total_epoch_steps += total_episode_steps

            study.on_episode_end(total_episode_steps, total_episode_return, r)

            if total_epoch_steps > local_steps_per_epoch:
                game_state = "END_EPOCH"
            else:
                game_state = "INIT"

        elif game_state == "END_EPOCH":
            t_bi_0 = time.perf_counter()
            for experience in list(trajectory)[::-1]:
                _o, _s, _a, _bellman_reward, _o_prime, _s_prime, _d = experience
                q_vals_update = [
                    q_table[(_s_prime, _action)] for _action in range(act_dim)
                ]
                q_table[(_s, _a)] += alpha * (
                    _bellman_reward
                    + gamma * (1 - _d) * max(q_vals_update)
                    - q_table[(_s, _a)]
                )
            t_bi_1 = time.perf_counter()
            study.on_replay_complete(len(trajectory), t_bi_1 - t_bi_0)
            trajectory.clear()

            q_table, v_table = study.synchronize(q_table, v_table)
            if study.on_epoch_end(
                epoch, effective_epsilon, total_epoch_steps,
                total_epoch_episodes, total_epoch_return, q_table,
            ):
                return

            epoch += 1
            effective_epsilon = epsilon_for_epoch(epsilon, epoch, epsilon_decay_epochs)
            total_epoch_steps = 0
            total_epoch_return = 0
            total_epoch_episodes = 0
            game_state = "INIT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gymnasium as gym
    from sparse_parallel_q.comparison_support import ComparisonRun

    study = ComparisonRun(config, args.run_dir)
    environment = None

    def make_environment():
        nonlocal environment
        environment = gym.make("LunarLander-v3")
        return environment

    try:
        train(
            make_environment,
            steps_per_epoch=config["workers"] * config["local_steps"],
            epsilon=config["epsilon"],
            epsilon_decay_epochs=config["epsilon_decay_epochs"],
            study=study,
        )
        if study.completed_epochs != config["epochs"]:
            raise RuntimeError("Trainer returned before the planned epoch budget")
    except BaseException:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
        raise
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
