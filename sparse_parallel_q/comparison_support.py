"""Seeded experiment control, metrics, and checkpoints for the trainer.

This observer never changes rewards, action selection, episode boundaries, or
replay. Its seed streams, aggregation choice, and finite epoch budget are
explicit study controls. It is not an alternative training engine.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time


CONDITIONS = ("frequency_wta", "count_weighted_mean", "uniform_mean", "random_visited")
SEED_STREAMS = {"environment": 10, "exploration": 20, "aggregation": 30,
                "numpy": 40, "torch": 50}


def validate_run_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Comparison config requires schema_version 1")
    if config.get("condition") not in CONDITIONS:
        raise ValueError("Unknown aggregation condition")
    for key in ("workers", "epochs", "local_steps", "checkpoint_every_epochs"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("training_seed", "epsilon_decay_epochs"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    epsilon = config.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    return config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_checkpoint(path, q_table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen checkpoint: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(dict(q_table), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "sha256": file_digest(path), "entries": len(q_table)}


class ComparisonRun:
    def __init__(self, config, run_dir):
        self.config = validate_run_config(dict(config))
        self.run_dir = Path(run_dir).resolve()
        self.first_reset = True
        self.total_steps = 0
        self.total_replay_updates = 0
        self.completed_epochs = 0
        self.episode_steps = 0
        self.episode_raw_return = 0.0
        self.episode_truncated = False
        self.total_episodes = 0
        self.total_successes = 0
        self.total_failures = 0
        self.checkpoints = []
        self.reset_epoch_counters()

    def reset_epoch_counters(self):
        self.epoch_steps = 0
        self.epoch_replay_updates = 0
        self.epoch_replay_seconds = 0.0
        self.epoch_sync_seconds = 0.0
        self.epoch_successes = 0
        self.epoch_failures = 0
        self.epoch_truncated_episodes = 0
        self.epoch_max_episode_length = 0
        self.epoch_raw_return = 0.0
        self.epoch_learning_return = 0.0
        self.epoch_visited_keys = 0

    def prepare(self):
        import numpy as np
        import torch
        from mpi4py import MPI
        from sparse_parallel_q import core

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.workers = self.comm.Get_size()
        self.core = core
        if self.workers != self.config["workers"]:
            raise ValueError("MPI size differs from configured workers")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        error = None
        if self.rank == 0:
            for name in ("training_protocol.json", "run_training_summary.json",
                         "checkpoints", "rank_metrics"):
                if (self.run_dir / name).exists():
                    error = f"Fresh comparison refuses pre-existing training artifact: {name}"
                    break
        error = self.comm.bcast(error, root=0)
        if error:
            raise FileExistsError(error)
        self.seeds = {
            name: int(np.random.SeedSequence([self.config["training_seed"], self.rank, stream])
                      .generate_state(1, dtype=np.uint32)[0])
            for name, stream in SEED_STREAMS.items()
        }
        random.seed(self.seeds["exploration"])
        np.random.seed(self.seeds["numpy"])
        torch.manual_seed(self.seeds["torch"])
        self.aggregation_rng = (np.random.default_rng(self.seeds["aggregation"])
                                if self.rank == 0 else None)
        self.started = time.perf_counter()
        rank_seeds = self.comm.gather({"rank": self.rank, "seeds": self.seeds}, root=0)
        if self.rank == 0:
            atomic_json(self.run_dir / "training_protocol.json", {
                "schema_version": 1, "config": self.config, "rank_seeds": rank_seeds,
                "trainer": "train.py", "initialization": "empty_q_and_counts",
                "seed_recipe": "uint32(SeedSequence([training_seed, rank, stream_id]))",
                "environment_reset": "seed_first_reset_only_then_continuous_environment_rng",
                "learning_reward": "zero_nonterminal_native_terminal_plus_or_minus_100",
                "make_sparse": True, "manual_timeout_penalty": False,
                "truncation": "recorded_but_ignored_for_training",
                "epoch_boundary": "finish_episode_until_local_steps_strictly_greater_than_target",
                "replay": "reverse_last_100000_transitions_constant_alpha_then_clear",
                "alpha": 0.05, "gamma": 0.99,
                "scales": [30.0, 1, 1, 1, 1, 1, 1, 1],
                "resolutions": [1.0, .3, .3, .3, .3, .3, .3, .3],
                "visit_counts": "one_pseudocount_online_only_reset_after_sync",
                "frozen_evaluations": "separate_process_after_training_no_training_rng_consumed",
            })
        self.comm.Barrier()

    def reset_environment(self, env):
        if self.first_reset:
            self.first_reset = False
            return env.reset(seed=self.seeds["environment"])
        return env.reset()

    def on_step(self, stats_reward, bellman_reward, terminated, truncated):
        expected = float(stats_reward) if terminated else 0.0
        if float(bellman_reward) != expected:
            raise AssertionError("Sparse learning reward gate violated")
        if terminated and float(stats_reward) not in (-100.0, 100.0):
            raise AssertionError("Unexpected native terminal reward")
        self.total_steps += 1
        self.epoch_steps += 1
        self.episode_steps += 1
        self.episode_truncated = self.episode_truncated or bool(truncated)
        self.epoch_raw_return += float(stats_reward)
        self.episode_raw_return += float(stats_reward)
        self.epoch_learning_return += float(bellman_reward)

    def on_episode_end(self, length, raw_return, terminal_reward):
        if length != self.episode_steps:
            raise AssertionError("Trainer and instrumentation episode lengths differ")
        if not math.isclose(raw_return, self.episode_raw_return, rel_tol=1e-9, abs_tol=1e-5):
            raise AssertionError("Trainer and instrumentation episode returns differ")
        success = float(terminal_reward) == 100.0
        if float(terminal_reward) not in (-100.0, 100.0):
            raise AssertionError("Episode ended without native outcome")
        self.total_episodes += 1
        self.epoch_successes += int(success)
        self.epoch_failures += int(not success)
        self.total_successes += int(success)
        self.total_failures += int(not success)
        self.epoch_truncated_episodes += int(self.episode_truncated)
        self.epoch_max_episode_length = max(self.epoch_max_episode_length, length)
        self.episode_steps = 0
        self.episode_raw_return = 0.0
        self.episode_truncated = False

    def on_replay_complete(self, replay_count, replay_seconds):
        if replay_count != min(self.epoch_steps, 100000):
            raise AssertionError("Replay did not cover the retained per-epoch trajectory")
        self.epoch_replay_updates = int(replay_count)
        self.total_replay_updates += int(replay_count)
        self.epoch_replay_seconds = float(replay_seconds)

    def synchronize(self, q_table, counts):
        started = time.perf_counter()
        if sum(int(count) - 1 for count in counts.values()) != self.epoch_steps:
            raise AssertionError("Visit counts do not match online environment interactions")
        self.epoch_visited_keys = sum(count > 1 for count in counts.values())
        condition = self.config["condition"]
        if condition == "random_visited":
            result = self.core.sync_random_visited_q_tables(q_table, counts, rng=self.aggregation_rng)
        else:
            helpers = {
                "frequency_wta": self.core.sync_frequency_wta_q_tables,
                "count_weighted_mean": self.core.sync_count_weighted_mean_q_tables,
                "uniform_mean": self.core.sync_uniform_mean_q_tables,
            }
            result = helpers[condition](q_table, counts)
        if any(count != 1 for count in result[1].values()):
            raise AssertionError("Aggregation did not reset visit counts")
        self.epoch_sync_seconds = time.perf_counter() - started
        return result

    def on_epoch_end(self, epoch, epsilon, local_steps, local_episodes, raw_return, q_table):
        if epoch != self.completed_epochs or local_steps != self.epoch_steps:
            raise AssertionError("Trainer and observer epoch counters differ")
        if local_steps <= self.config["local_steps"] or self.episode_steps:
            raise AssertionError("Epoch ended before complete-episode local target")
        if local_episodes != self.epoch_successes + self.epoch_failures:
            raise AssertionError("Trainer and observer episode counts differ")
        if not math.isclose(raw_return, self.epoch_raw_return, rel_tol=1e-9, abs_tol=1e-5):
            raise AssertionError("Trainer and observer epoch returns differ")
        completed = epoch + 1
        record = {
            "rank": self.rank, "epoch": epoch, "completed_epochs": completed,
            "epsilon": float(epsilon), "training_steps": int(local_steps),
            "training_episodes": int(local_episodes), "raw_return_sum": float(raw_return),
            "mean_raw_return": float(raw_return) / local_episodes,
            "learning_reward_sum": self.epoch_learning_return,
            "visited_state_action_pairs": self.epoch_visited_keys,
            "native_successes": self.epoch_successes, "native_failures": self.epoch_failures,
            "truncated_episodes": self.epoch_truncated_episodes,
            "max_episode_length": self.epoch_max_episode_length,
            "replay_updates": self.epoch_replay_updates,
            "replay_seconds": self.epoch_replay_seconds, "sync_seconds": self.epoch_sync_seconds,
            "cumulative_training_steps": self.total_steps,
            "cumulative_replay_updates": self.total_replay_updates,
            "cumulative_training_episodes": self.total_episodes,
            "cumulative_native_successes": self.total_successes,
            "cumulative_native_failures": self.total_failures,
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        append_record(self.run_dir / "rank_metrics" / f"rank-{self.rank:05d}.jsonl", record)
        ranks = self.comm.gather(record, root=0)
        error = None
        if self.rank == 0:
            try:
                summary = {
                    "epoch": epoch, "completed_epochs": completed,
                    "condition": self.config["condition"], "training_seed": self.config["training_seed"],
                    "epsilon": float(epsilon), "q_entries": len(q_table), "ranks": ranks,
                    "training_steps": sum(item["training_steps"] for item in ranks),
                    "training_episodes": sum(item["training_episodes"] for item in ranks),
                    "native_successes": sum(item["native_successes"] for item in ranks),
                    "native_failures": sum(item["native_failures"] for item in ranks),
                    "cumulative_training_steps": sum(item["cumulative_training_steps"] for item in ranks),
                    "cumulative_replay_updates": sum(item["cumulative_replay_updates"] for item in ranks),
                    "elapsed_seconds": max(item["elapsed_seconds"] for item in ranks),
                    "minimum_worker_mean_raw_return": min(item["mean_raw_return"] for item in ranks),
                    "maximum_worker_mean_raw_return": max(item["mean_raw_return"] for item in ranks),
                }
                if completed % self.config["checkpoint_every_epochs"] == 0 or completed == self.config["epochs"]:
                    checkpoint = publish_checkpoint(
                        self.run_dir / "checkpoints" / f"epoch-{completed:06d}.pkl", q_table)
                    checkpoint.update({"epoch": epoch, "completed_epochs": completed,
                                       "cumulative_training_steps": summary["cumulative_training_steps"],
                                       "cumulative_replay_updates": summary["cumulative_replay_updates"]})
                    atomic_json(Path(checkpoint["path"]).with_suffix(".json"), checkpoint)
                    self.checkpoints.append(checkpoint)
                append_record(self.run_dir / "epochs.jsonl", summary)
                if completed == self.config["epochs"]:
                    summary.update({"status": "completed", "checkpoints": self.checkpoints,
                                    "initialization": "fresh", "config": self.config})
                    atomic_json(self.run_dir / "run_training_summary.json", summary)
                print(f"STUDY epoch={completed}/{self.config['epochs']} "
                      f"condition={self.config['condition']} seed={self.config['training_seed']} "
                      f"steps={summary['cumulative_training_steps']} "
                      f"worker_return_range={summary['minimum_worker_mean_raw_return']:.2f}.."
                      f"{summary['maximum_worker_mean_raw_return']:.2f}", flush=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self.comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        self.completed_epochs = completed
        self.reset_epoch_counters()
        return completed == self.config["epochs"]
