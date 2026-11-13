"""Seeded, frozen-policy evaluation for the aggregation comparison.

Evaluation runs separately from training. A guard censors an unfinished
episode without changing its reward or counting it as a crash.
"""

import argparse
from collections.abc import Mapping
import importlib.metadata
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import pickle
import platform
import time

from sparse_parallel_q.comparison_support import atomic_json, file_digest


def load_seed_bank(path):
    bank = json.loads(Path(path).read_text())
    if not isinstance(bank, dict) or set(bank) != {"schema_version", "bank", "seeds"}:
        raise ValueError("Seed bank requires schema_version, bank, and seeds")
    if type(bank["schema_version"]) is not int or bank["schema_version"] != 1:
        raise ValueError("Seed bank schema_version must be 1")
    if bank["bank"] not in ("monitor", "final"):
        raise ValueError("Seed bank must be monitor or final")
    seeds = bank["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Seed bank must contain a nonempty list of seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Environment seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Environment seeds must be distinct within a bank")
    return bank


def assigned_episodes(seeds, rank, workers):
    if workers < 1 or rank < 0 or rank >= workers:
        raise ValueError("Rank must lie within a positive worker count")
    return [(index, seeds[index]) for index in range(rank, len(seeds), workers)]


def load_q_table(path):
    # Checkpoints are trusted, locally produced research artifacts, not downloads.
    with open(path, "rb") as stream:
        values = pickle.load(stream)
    if not isinstance(values, Mapping):
        raise ValueError("Checkpoint must contain a plain state/action Q-table")
    for pair, value in values.items():
        if (not isinstance(pair, tuple) or len(pair) != 2
                or not isinstance(pair[0], tuple) or len(pair[0]) != 8
                or any(not isinstance(component, Integral) for component in pair[0])
                or not isinstance(pair[1], Integral) or pair[1] not in range(4)
                or not isinstance(value, Real) or not math.isfinite(value)):
            raise ValueError("Checkpoint contains an invalid LunarLander Q entry")
    return dict(values)


def make_encoder(torch):
    scalers = torch.tensor([30.0, 1, 1, 1, 1, 1, 1, 1])
    resolutions = [1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    def encode_observation(obs):
        with torch.no_grad():
            normalized = torch.tanh(torch.tensor(obs) * scalers).tolist()
            quantized_s = tuple([round(x / resolutions[idx]) for idx, x in enumerate(normalized)])
            return quantized_s

    return encode_observation


def greedy_action(q_table, state):
    curr_q_max = float("-inf")
    action = None
    unknown_reads = 0
    for candidate in range(4):
        pair = (state, candidate)
        unknown_reads += int(pair not in q_table)
        q_value = q_table.get(pair, 0.0)
        if q_value > curr_q_max:
            curr_q_max = q_value
            action = candidate
    return action, unknown_reads


def evaluate_episode(env, q_table, encode, episode_index, environment_seed,
                     guard_steps, *, rank=0):
    if type(guard_steps) is not int or guard_steps < 1:
        raise ValueError("guard_steps must be a positive integer")
    started = time.perf_counter()
    record = {
        "episode_index": episode_index, "environment_seed": environment_seed,
        "rank": rank, "steps": 0, "raw_environment_return": 0.0,
        "outcome": "running", "success": None, "terminal_reward": None,
        "terminated": False, "truncated": False, "truncated_observed": False,
        "first_truncation_step": None, "raw_return_at_first_truncation": None,
        "guard_steps": guard_steps, "guard_reached": False,
        "unknown_state_action_reads": 0, "steps_with_unknown_q": 0,
        "elapsed_seconds": 0.0,
    }
    observation, info = env.reset(seed=environment_seed)
    state = encode(observation)
    while True:
        action, unknown_reads = greedy_action(q_table, state)
        observation, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Environment returned a nonfinite reward")
        record["steps"] += 1
        record["raw_environment_return"] += reward
        record["unknown_state_action_reads"] += unknown_reads
        record["steps_with_unknown_q"] += int(unknown_reads > 0)
        record["terminated"] = bool(terminated)
        record["truncated"] = bool(truncated)
        if truncated and not record["truncated_observed"]:
            record["first_truncation_step"] = record["steps"]
            record["raw_return_at_first_truncation"] = record["raw_environment_return"]
        record["truncated_observed"] |= bool(truncated)
        # Genuine termination takes precedence if it occurs on the guard step.
        if terminated:
            if reward not in (-100.0, 100.0):
                raise ValueError("Unexpected native terminal reward: %r" % reward)
            record["terminal_reward"] = reward
            record["success"] = reward == 100.0
            record["outcome"] = "success" if record["success"] else "failure"
            record["elapsed_seconds"] = time.perf_counter() - started
            return record
        if record["steps"] >= guard_steps:
            record["guard_reached"] = True
            record["outcome"] = "censored"
            record["elapsed_seconds"] = time.perf_counter() - started
            return record
        state = encode(observation)


def wilson_interval(successes, count):
    if count < 1 or successes < 0 or successes > count:
        raise ValueError("Wilson interval requires 0 <= successes <= count and count > 0")
    fraction = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(fraction * (1 - fraction) / count
                               + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def summarize_episodes(episodes, planned_episodes):
    if planned_episodes < 1:
        raise ValueError("A positive episode budget is required")
    episodes = sorted(episodes, key=lambda item: item["episode_index"])
    indices = [item["episode_index"] for item in episodes]
    if (len(set(indices)) != len(indices)
            or any(index not in range(planned_episodes) for index in indices)):
        raise ValueError("Episode indices must be unique and within the planned bank")
    counts = {outcome: sum(item["outcome"] == outcome for item in episodes)
              for outcome in ("success", "failure", "censored")}
    if sum(counts.values()) != len(episodes):
        raise ValueError("Only completed or censored episode records may be summarized")
    completed = len(episodes)
    missing = planned_episodes - completed
    successes, failures, censored = (counts[name] for name in ("success", "failure", "censored"))
    return {
        "planned_episodes": planned_episodes, "episode_count": completed,
        "successes": successes, "failures": failures, "censored": censored,
        "unrecorded_episodes": missing,
        "unrecorded_episode_indices": sorted(set(range(planned_episodes)) - set(indices)),
        "observed_success_fraction": successes / planned_episodes,
        "observed_failure_fraction": failures / planned_episodes,
        "censored_fraction": censored / planned_episodes,
        "success_fraction_bounds": [successes / planned_episodes,
                                    (successes + censored + missing) / planned_episodes],
        "success_fraction_bounds_scope": "observed_bank_bounds_not_a_confidence_interval",
        "success_wilson_95": wilson_interval(successes, planned_episodes)
        if censored == 0 and missing == 0 else None,
        "interval_scope": "evaluation_episodes_for_this_policy_not_training_seed_variance",
        "total_environment_steps": sum(item["steps"] for item in episodes),
        "mean_raw_environment_return": (sum(item["raw_environment_return"] for item in episodes)
                                        / completed) if completed else None,
        "raw_return_scope": "includes_partial_returns_for_censored_episodes",
        "mean_episode_steps": (sum(item["steps"] for item in episodes) / completed)
        if completed else None,
        "truncation_observed_episodes": sum(item["truncated_observed"] for item in episodes),
        "episodes": episodes,
    }


def dependency_versions():
    versions = {"python": platform.python_version()}
    for package in ("gymnasium", "torch", "numpy", "mpi4py", "box2d-py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-steps", type=int, default=100000)
    args = parser.parse_args(argv)
    if args.guard_steps < 1:
        parser.error("--guard-steps must be positive")

    import gymnasium as gym
    import torch
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, workers = comm.Get_rank(), comm.Get_size()
    torch.set_num_threads(1)
    output = args.output.resolve()
    setup = None
    if rank == 0:
        try:
            bank = load_seed_bank(args.seed_bank)
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise ValueError("Evaluation output directory must be empty")
            manifest = {
                "schema_version": 1, "status": "running", "bank": bank["bank"],
                "seed_bank": bank, "seed_bank_path": str(args.seed_bank.resolve()),
                "seed_bank_sha256": file_digest(args.seed_bank),
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_sha256_before": file_digest(args.checkpoint),
                "source_path": str(Path(__file__).resolve()),
                "source_sha256": file_digest(__file__), "workers": workers,
                "guard_steps": args.guard_steps, "environment": "LunarLander-v3",
                "termination_mode": "native_terminated_only_with_guard_censoring",
                "truncation_mode": "record_and_continue_without_reward_modification",
                "policy": "frozen_greedy_lowest_action_index_ties",
                "observation_scales": [30.0, 1, 1, 1, 1, 1, 1, 1],
                "observation_resolutions": [1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                "dependencies": dependency_versions(), "started_unix_seconds": time.time(),
            }
            setup = {"manifest": manifest, "bank": bank, "error": None}
        except Exception as error:
            setup = {"error": "%s: %s" % (type(error).__name__, error)}
    setup = comm.bcast(setup, root=0)
    if setup["error"] is not None:
        if rank == 0:
            print(setup["error"], flush=True)
        return 1

    records, error_message, loaded_hash = [], None, None
    env = None
    try:
        loaded_hash = file_digest(args.checkpoint)
        if loaded_hash != setup["manifest"]["checkpoint_sha256_before"]:
            raise ValueError("Checkpoint changed before worker loaded it")
        q_table = load_q_table(args.checkpoint)
        if file_digest(args.checkpoint) != loaded_hash:
            raise ValueError("Checkpoint changed while worker loaded it")
        encode = make_encoder(torch)
        env = gym.make("LunarLander-v3")
        if env.action_space.n != 4 or env.observation_space.shape != (8,):
            raise ValueError("Expected LunarLander's eight observations and four actions")
        with open(output / ("rank-%05d.jsonl" % rank), "x", encoding="utf-8") as stream:
            for index, seed in assigned_episodes(setup["bank"]["seeds"], rank, workers):
                record = evaluate_episode(env, q_table, encode, index, seed, args.guard_steps,
                                          rank=rank)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
            os.fsync(stream.fileno())
    except Exception as error:
        error_message = "%s: %s" % (type(error).__name__, error)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                error_message = error_message or "Environment close failed: %s" % error

    gathered = comm.gather({"rank": rank, "episodes": records, "error": error_message,
                            "loaded_checkpoint_sha256": loaded_hash}, root=0)
    exit_code = None
    if rank == 0:
        try:
            result = dict(setup["manifest"])
            all_records = [episode for worker in gathered for episode in worker["episodes"]]
            result.update(summarize_episodes(all_records, len(setup["bank"]["seeds"])))
            result["worker_errors"] = [{"rank": worker["rank"], "error": worker["error"]}
                                       for worker in gathered if worker["error"] is not None]
            result["worker_checkpoint_sha256"] = [worker["loaded_checkpoint_sha256"] for worker in gathered]
            try:
                result["checkpoint_sha256_after"] = file_digest(args.checkpoint)
            except OSError:
                result["checkpoint_sha256_after"] = None
            result["checkpoint_file_unchanged"] = (
                result["checkpoint_sha256_after"] == result["checkpoint_sha256_before"])
            complete = (not result["worker_errors"] and not result["unrecorded_episodes"]
                        and result["checkpoint_file_unchanged"])
            result["status"] = "completed" if complete else "failed"
            result["finished_unix_seconds"] = time.time()
            result["elapsed_seconds"] = result["finished_unix_seconds"] - result["started_unix_seconds"]
            atomic_json(output / "results.json", result)
            print(json.dumps({key: result[key] for key in
                              ("status", "episode_count", "successes", "failures", "censored",
                               "success_fraction_bounds", "success_wilson_95")}), flush=True)
            exit_code = 0 if complete else 1
        except Exception as error:
            # Every rank still reaches the final broadcast if publication fails.
            print("Evaluation publication failed: %s: %s" % (type(error).__name__, error), flush=True)
            exit_code = 1
    return comm.bcast(exit_code, root=0)


if __name__ == "__main__":
    raise SystemExit(main())
