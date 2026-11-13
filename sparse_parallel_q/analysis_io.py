"""Read completed studies and portable outcome bundles without training imports."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


BUNDLE_KIND = "sparse_parallel_q_outcomes"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Input path escapes study directory")
    return path


def expected_epochs(config):
    return sorted(set(range(config["checkpoint_every_epochs"], config["epochs"] + 1,
                            config["checkpoint_every_epochs"])) | {config["epochs"]})


def selected_grid(config, selection):
    require(set(selection) <= {"condition", "seed"}, "Unknown study selection fields")
    condition, seed = selection.get("condition"), selection.get("seed")
    require(condition is None or condition in config["conditions"], "Unknown selected condition")
    require(seed is None or seed in config["training_seeds"], "Unknown selected training seed")
    return {(item, value) for item in config["conditions"] for value in config["training_seeds"]
            if (condition is None or item == condition) and (seed is None or value == seed)}


def outcome_metrics(assessment, config, banks):
    episodes = assessment["episodes"]
    bank = assessment["bank"]
    count = config[bank + "_episodes"]
    require(len(episodes) == count and sorted(item["episode_index"] for item in episodes) == list(range(count)),
            "Evaluation episode indices are missing or duplicated")
    for episode in episodes:
        index = episode["episode_index"]
        require(episode["environment_seed"] == banks[bank]["seeds"][index], "Evaluation seed mismatch")
        require(episode["rank"] == index % config["workers"], "Evaluation rank assignment mismatch")
        steps = episode["steps"]
        require(type(steps) is int and 1 <= steps <= config["evaluation_guard_steps"], "Invalid episode steps")
        require(episode["guard_steps"] == config["evaluation_guard_steps"], "Episode guard mismatch")
        outcome = episode["outcome"]
        if outcome in ("success", "failure"):
            success = outcome == "success"
            require(episode["terminated"] is True and episode["terminal_reward"] == (100 if success else -100)
                    and episode["success"] is success and episode["guard_reached"] is False,
                    "Success/failure must be its genuine terminal outcome")
        else:
            require(outcome == "censored" and episode["terminated"] is False
                    and episode["terminal_reward"] is None and episode["success"] is None
                    and episode["guard_reached"] is True and steps == config["evaluation_guard_steps"],
                    "Censoring must retain an unresolved episode at the guard")
        require(isinstance(episode["raw_environment_return"], (int, float))
                and math.isfinite(episode["raw_environment_return"]), "Nonfinite raw episode return")
    counts = Counter(item["outcome"] for item in episodes)
    successes, failures, censored = (counts[name] for name in ("success", "failure", "censored"))
    metrics = {"episodes": count, "successes": successes, "failures": failures, "censored": censored,
               "observed_success_fraction": successes / count,
               "success_fraction_bounds": [successes / count, (successes + censored) / count],
               "evaluation_steps": sum(item["steps"] for item in episodes),
               "mean_raw_environment_return": statistics.mean(item["raw_environment_return"] for item in episodes)}
    recorded = assessment["recorded_summary"]
    for name in ("successes", "failures", "censored", "observed_success_fraction", "success_fraction_bounds"):
        require(recorded[name] == metrics[name], "Raw outcomes disagree with recorded summary: " + name)
    require(recorded["episode_count"] == count and recorded["planned_episodes"] == count,
            "Recorded evaluation bank denominator changed")
    require(recorded["total_environment_steps"] == metrics["evaluation_steps"], "Evaluation step sum changed")
    require(math.isclose(recorded["mean_raw_environment_return"], metrics["mean_raw_environment_return"],
                         rel_tol=1e-12, abs_tol=1e-12), "Raw return mean disagrees with episode records")
    return metrics


def validate_dataset(dataset):
    config, runs, banks = dataset["config"], dataset["runs"], dataset["seed_banks"]
    require(config["conditions"] and len(set(config["conditions"])) == len(config["conditions"]),
            "Conditions must be nonempty and unique")
    require(config["training_seeds"] and len(set(config["training_seeds"])) == len(config["training_seeds"]),
            "Training seeds must be nonempty and unique")
    for key in ("epochs", "checkpoint_every_epochs", "workers", "local_steps", "evaluation_guard_steps",
                "monitor_episodes", "final_episodes"):
        require(type(config[key]) is int and config[key] > 0, "Expected positive integer: " + key)
    for bank in ("monitor", "final"):
        seeds = banks[bank]["seeds"]
        require(banks[bank]["bank"] == bank and len(seeds) == config[bank + "_episodes"]
                and len(set(seeds)) == len(seeds), "Invalid seed bank")
    require(not set(banks["monitor"]["seeds"]) & set(banks["final"]["seeds"]), "Seed banks overlap")
    grid = selected_grid(config, dataset.get("selection", {}))
    identities = [(run["condition"], run["training_seed"]) for run in runs]
    require(len(identities) == len(grid) and set(identities) == grid, "Completed run matrix is missing or duplicated")
    epochs = expected_epochs(config)
    checkpoints = {}
    for run in runs:
        require(run["completed_epochs"] == config["epochs"], "Training did not complete its fixed epoch budget")
        for field in ("actual_training_steps", "replay_updates", "training_episodes"):
            require(type(run[field]) is int and run[field] >= 0, "Invalid training counter")
        require(run["replay_updates"] <= run["actual_training_steps"], "Replay exceeds the one-pass collection budget")
        points = run["checkpoints"]
        require([point["completed_epochs"] for point in points] == epochs, "Training checkpoints are incomplete or duplicated")
        previous_steps = previous_replay = 0
        for point in points:
            steps, replay = point["actual_training_steps"], point["replay_updates"]
            require(type(steps) is int and type(replay) is int and steps > previous_steps
                    and previous_replay <= replay <= steps, "Invalid cumulative checkpoint counters")
            identity = run["condition"], run["training_seed"], point["completed_epochs"]
            checkpoints[identity] = point
            previous_steps, previous_replay = steps, replay
        require(previous_steps == run["actual_training_steps"] and previous_replay == run["replay_updates"],
                "Final counters disagree with final checkpoint")
    expected = {(condition, seed, epoch, "monitor") for condition, seed in grid for epoch in epochs}
    expected |= {(condition, seed, config["epochs"], "final") for condition, seed in grid}
    actual = []
    for assessment in dataset["evaluations"]:
        identity = assessment["condition"], assessment["training_seed"], assessment["completed_epochs"], assessment["bank"]
        require(identity in expected, "Unexpected evaluation identity")
        actual.append(identity)
        require(assessment["checkpoint_sha256"] == checkpoints[identity[:3]]["sha256"],
                "Evaluation checkpoint differs from training snapshot")
        outcome_metrics(assessment, config, banks)
    require(len(actual) == len(expected) and set(actual) == expected, "Evaluation matrix is incomplete or duplicated")
    return dataset


def load_bundle(directory):
    directory = Path(directory).resolve()
    metadata = read_json(directory / "study.json")
    require(metadata.get("kind") == BUNDLE_KIND and metadata.get("schema_version") == 1,
            "Unsupported portable results format")
    for record in metadata["files"]:
        require(file_sha256(safe_path(directory, record["path"])) == record["sha256"],
                "Portable input hash mismatch: " + record["path"])
    required = {"config.json", "runs.json", "evaluations.jsonl", "seed_banks/monitor.json", "seed_banks/final.json"}
    require(required <= {record["path"] for record in metadata["files"]}, "Missing required input hash records")
    snapshots = [record for record in metadata["files"] if record["path"].startswith("source_snapshot/")]
    snapshot_status = metadata["provenance"].get("source_snapshot_status")
    if snapshot_status is not None:
        require(snapshot_status == ("included" if snapshots else "not_included"),
                "Source snapshot availability disagrees with included files")
    dataset = {"config": read_json(directory / "config.json"), "runs": read_json(directory / "runs.json"),
               "seed_banks": {bank: read_json(directory / "seed_banks" / (bank + ".json")) for bank in ("monitor", "final")},
               "evaluations": [json.loads(line, parse_constant=reject_constant)
                               for line in (directory / "evaluations.jsonl").read_text().splitlines()],
               "selection": metadata.get("selection", {}), "provenance": metadata["provenance"],
               "input_hashes": [{"path": "study.json", "sha256": file_sha256(directory / "study.json")}, *metadata["files"]]}
    return validate_dataset(dataset)


def load_native_study(directory):
    """Read a completed comparison and verify its source and checkpoint files."""
    root = Path(directory).resolve()
    manifest, config, status = (
        read_json(root / name) for name in ("manifest.json", "config.json", "status.json")
    )
    require(status["status"] == "completed", "Analysis requires a completed study")
    hashes = {}

    def verify(relative, expected=None):
        path = safe_path(root, relative)
        key = path.relative_to(root).as_posix()
        if key not in hashes:
            hashes[key] = {"path": key, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        require(expected is None or hashes[key]["sha256"] == expected, "File hash mismatch: " + key)
        return hashes[key]["sha256"]

    for name in ("manifest.json", "status.json", "config.json"):
        verify(name)
    require(hashlib.sha256(canonical_json(config).encode()).hexdigest() == manifest["config_sha256"],
            "Configuration digest mismatch")
    for source in manifest["source_files"]:
        verify(Path("source_snapshot") / source["path"], source["sha256"])
    evaluator_hash = next(source["sha256"] for source in manifest["source_files"]
                          if source["path"].endswith("/comparison_evaluate.py"))
    banks = {}
    for bank in ("monitor", "final"):
        path = Path("seed_banks") / (bank + ".json")
        verify(path, manifest["seed_bank_sha256"][bank])
        banks[bank] = read_json(root / path)

    runs, evaluations = [], []
    for planned in manifest["runs"]:
        condition, seed = planned["condition"], planned["training_seed"]
        job = status["jobs"][f"{condition}/seed-{seed}"]
        require(job["status"] == "completed", "Planned run is not completed")
        run_path = Path(job["directory"])
        verify(run_path / "config.json")
        require(read_json(safe_path(root, run_path / "config.json")) == planned,
                "Run configuration differs from plan")
        summary_path = run_path / "run_training_summary.json"
        trained = read_json(safe_path(root, summary_path))
        require(trained["status"] == "completed" and trained["condition"] == condition
                and trained["training_seed"] == seed and trained["completed_epochs"] == planned["epochs"],
                "Training identity or epoch budget mismatch")
        ranks = trained["ranks"]
        require(sorted(row["rank"] for row in ranks) == list(range(planned["workers"])),
                "Missing training ranks")
        for field in ("cumulative_training_steps", "cumulative_replay_updates"):
            require(sum(row[field] for row in ranks) == trained[field], "Training rank totals disagree")
        checkpoints = []
        for point in trained["checkpoints"]:
            verify(run_path / f"checkpoints/epoch-{point['completed_epochs']:06d}.pkl", point["sha256"])
            checkpoints.append({"completed_epochs": point["completed_epochs"], "sha256": point["sha256"],
                                "actual_training_steps": point["cumulative_training_steps"],
                                "replay_updates": point["cumulative_replay_updates"], "entries": point["entries"]})
        runs.append({"condition": condition, "training_seed": seed, "completed_epochs": trained["completed_epochs"],
                     "actual_training_steps": trained["cumulative_training_steps"],
                     "replay_updates": trained["cumulative_replay_updates"],
                     "training_episodes": sum(row["cumulative_training_episodes"] for row in ranks),
                     "trainer_elapsed_seconds": trained["elapsed_seconds"], "checkpoints": checkpoints,
                     "original_training_summary_sha256": verify(summary_path)})
        assessments = [(epoch, "monitor") for epoch in expected_epochs(planned)] + [(planned["epochs"], "final")]
        for epoch, bank in assessments:
            result_path = run_path / "evaluations" / f"epoch-{epoch:06d}-{bank}" / "results.json"
            result = read_json(safe_path(root, result_path))
            checkpoint_hash = next(point["sha256"] for point in checkpoints if point["completed_epochs"] == epoch)
            require(result["status"] == "completed" and result["bank"] == bank
                    and result["source_sha256"] == evaluator_hash, "Evaluation source or identity mismatch")
            require(result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"] == checkpoint_hash
                    and result["checkpoint_file_unchanged"] is True
                    and result["worker_checkpoint_sha256"] == [checkpoint_hash] * planned["workers"],
                    "Evaluation checkpoint changed")
            require(result["seed_bank_sha256"] == manifest["seed_bank_sha256"][bank]
                    and result["seed_bank"] == banks[bank], "Evaluation seed bank changed")
            require(result["guard_steps"] == planned["evaluation_guard_steps"]
                    and result["workers"] == planned["workers"] and not result["worker_errors"]
                    and result["unrecorded_episodes"] == 0, "Evaluation protocol or completion mismatch")
            require(result["environment"] == "LunarLander-v3"
                    and result["termination_mode"] == "native_terminated_only_with_guard_censoring"
                    and result["truncation_mode"] == "record_and_continue_without_reward_modification"
                    and result["policy"] == "frozen_greedy_lowest_action_index_ties",
                    "Evaluation policy or termination protocol changed")
            fields = ("successes", "failures", "censored", "observed_success_fraction", "success_fraction_bounds",
                      "episode_count", "planned_episodes", "total_environment_steps", "mean_raw_environment_return")
            evaluations.append({"condition": condition, "training_seed": seed, "completed_epochs": epoch,
                                "bank": bank, "checkpoint_sha256": checkpoint_hash,
                                "original_results_sha256": verify(result_path),
                                "source_sha256": result["source_sha256"], "episodes": result["episodes"],
                                "recorded_summary": {field: result[field] for field in fields}})
    return validate_dataset({"config": config, "runs": runs, "evaluations": evaluations, "seed_banks": banks,
                             "selection": manifest.get("selection", {}), "input_hashes": list(hashes.values())})


def load_dataset(directory):
    return load_bundle(directory) if (Path(directory) / "study.json").exists() else load_native_study(directory)
