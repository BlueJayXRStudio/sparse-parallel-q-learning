"""Run the fixed aggregation comparison as sequential MPI jobs."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import subprocess
import sys

from sparse_parallel_q.comparison_support import atomic_json, file_digest


PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parent
DEFAULT_CONFIG = PROJECT / "configs/comparison.json"
CONDITIONS = ("frequency_wta", "count_weighted_mean", "uniform_mean", "random_visited")
RUN_FIELDS = {
    "schema_version", "workers", "epochs", "local_steps", "epsilon",
    "epsilon_decay_epochs", "checkpoint_every_epochs", "monitor_episodes",
    "final_episodes", "evaluation_guard_steps",
}
NATIVE_THREADS = {name: "1" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}
SEED_BANK_GENERATOR_SEED = 202611120264


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def value_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_config(config):
    fields = RUN_FIELDS | {"study_id", "conditions", "training_seeds"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("Configuration fields must match the reference study")
    if not isinstance(config["study_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", config["study_id"]
    ):
        raise ValueError("study_id must be a safe directory name")
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("The reference study uses the four declared aggregation rules")
    if config["training_seeds"] != [101, 102, 103] or any(
        type(seed) is not int for seed in config["training_seeds"]
    ):
        raise ValueError("The reference study uses paired seeds 101, 102, and 103")
    fixed = {
        "schema_version": 1, "workers": 16, "epochs": 264, "local_steps": 64000,
        "epsilon_decay_epochs": 100, "checkpoint_every_epochs": 32,
        "monitor_episodes": 100, "final_episodes": 200, "evaluation_guard_steps": 100000,
    }
    for key, expected in fixed.items():
        if type(config[key]) is not int or config[key] != expected:
            raise ValueError(f"{key} must be {expected} for the reference study")
    epsilon = config["epsilon"]
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon) or epsilon != 0.03):
        raise ValueError("epsilon must be 0.03")
    return config


def resolve_runs(config, condition=None, seed=None):
    if condition is not None and condition not in config["conditions"]:
        raise ValueError("Selected condition is not in the study")
    if seed is not None and (type(seed) is not int or seed not in config["training_seeds"]):
        raise ValueError("Selected seed is not in the study")
    runs = []
    for index, training_seed in enumerate(config["training_seeds"]):
        order = config["conditions"][index:] + config["conditions"][:index]
        for aggregation in order:
            if ((condition is not None and aggregation != condition)
                    or (seed is not None and training_seed != seed)):
                continue
            runs.append({**{key: config[key] for key in sorted(RUN_FIELDS)},
                         "condition": aggregation, "training_seed": training_seed})
    return runs


def checkpoint_epochs(config):
    return sorted(set(range(config["checkpoint_every_epochs"], config["epochs"] + 1,
                            config["checkpoint_every_epochs"])) | {config["epochs"]})


def seed_banks(config):
    count = config["monitor_episodes"] + config["final_episodes"]
    seeds = random.Random(SEED_BANK_GENERATOR_SEED).sample(range(2 ** 32), count)
    split = config["monitor_episodes"]
    return {"monitor": {"schema_version": 1, "bank": "monitor", "seeds": seeds[:split]},
            "final": {"schema_version": 1, "bank": "final", "seeds": seeds[split:]}}


def job_id(run):
    return f"{run['condition']}/seed-{run['training_seed']}"


def source_paths():
    return sorted([*PROJECT.glob("*.py"), *(PROJECT / "configs").glob("*.json"),
                   REPOSITORY / "pyproject.toml"])


def prepare_study(config, runs, directory, mpirun, selection):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "config.json", config)
    sources = []
    for path in source_paths():
        relative = path.relative_to(REPOSITORY)
        target = directory / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources.append({"path": relative.as_posix(), "sha256": file_digest(target)})
    banks = seed_banks(config)
    for bank, value in banks.items():
        atomic_json(directory / "seed_banks" / f"{bank}.json", value)
    manifest = {
        "schema_version": 1, "created_utc": utc_now(), "config_sha256": value_hash(config),
        "runs": runs, "selection": selection, "source_files": sources,
        "seed_bank_sha256": {bank: file_digest(directory / "seed_banks" / f"{bank}.json")
                             for bank in banks},
        "seed_bank_generator_seed": SEED_BANK_GENERATOR_SEED,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "dependencies": {name: version(name) for name in
                                     ("gymnasium", "box2d-py", "pygame", "numpy", "torch", "mpi4py")}},
        "python": sys.executable, "mpirun": str(mpirun), "native_thread_limits": NATIVE_THREADS,
    }
    atomic_json(directory / "manifest.json", manifest)
    status = {"status": "prepared", "jobs": {
        job_id(run): {"status": "pending", "directory": "runs/" + job_id(run)} for run in runs
    }}
    atomic_json(directory / "status.json", status)
    return manifest, status


def stop_job(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_mpi(mpirun, workers, script, arguments, directory, log_path):
    command = [str(mpirun), "--tag-output", "--oversubscribe", "-np", str(workers),
               sys.executable, "-u", "-B", str(PROJECT / script), *map(str, arguments)]
    environment = dict(os.environ, **NATIVE_THREADS, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(REPOSITORY))
    print(f"Running {script}; log: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f"MPI job exited {code}; see {log_path}")
        except BaseException:
            stop_job(process)
            raise


def training_checkpoints(directory, run):
    summary = read_json(directory / "run_training_summary.json")
    if (summary.get("status") != "completed" or summary.get("completed_epochs") != run["epochs"]
            or summary.get("condition") != run["condition"]
            or summary.get("training_seed") != run["training_seed"]):
        raise ValueError("Training summary does not match the planned run")
    points = summary["checkpoints"]
    if [point["completed_epochs"] for point in points] != checkpoint_epochs(run):
        raise ValueError("Training checkpoint epochs do not match the study")
    hashes = {}
    for point in points:
        epoch = point["completed_epochs"]
        path = directory / "checkpoints" / f"epoch-{epoch:06d}.pkl"
        if file_digest(path) != point["sha256"]:
            raise ValueError(f"Training checkpoint changed: {path}")
        hashes[epoch] = point["sha256"]
    return hashes


def verify_evaluation(output, checkpoint, digest, bank_path, run, bank):
    result = read_json(output / "results.json")
    expected = run[bank + "_episodes"]
    if (result.get("status") != "completed" or result.get("bank") != bank
            or result.get("planned_episodes") != expected or result.get("episode_count") != expected
            or len(result.get("episodes", [])) != expected):
        raise ValueError("Evaluation did not complete the requested episode bank")
    counts = [result.get(key) for key in ("successes", "failures", "censored")]
    if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != expected:
        raise ValueError("Evaluation outcome counts are invalid")
    if (file_digest(checkpoint) != digest or result.get("checkpoint_sha256_before") != digest
            or result.get("checkpoint_sha256_after") != digest):
        raise ValueError("Evaluation checkpoint changed")
    if result.get("seed_bank_sha256") != file_digest(bank_path):
        raise ValueError("Evaluation seed bank changed")


def run_study(directory, manifest, status):
    status["status"] = "running"
    job = None
    try:
        for run in manifest["runs"]:
            job = status["jobs"][job_id(run)]
            run_dir = directory / job["directory"]
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(run_dir / "config.json", run)
            job.update(status="running", phase="training")
            atomic_json(directory / "status.json", status)
            run_mpi(manifest["mpirun"], run["workers"], "train.py",
                    ["--config", run_dir / "config.json", "--run-dir", run_dir],
                    run_dir, run_dir / "training.log")
            checkpoints = training_checkpoints(run_dir, run)
            assessments = [(epoch, "monitor") for epoch in checkpoint_epochs(run)]
            assessments.append((run["epochs"], "final"))
            for epoch, bank in assessments:
                label = f"epoch-{epoch:06d}-{bank}"
                output = run_dir / "evaluations" / label
                output.mkdir(parents=True, exist_ok=False)
                checkpoint = run_dir / "checkpoints" / f"epoch-{epoch:06d}.pkl"
                if file_digest(checkpoint) != checkpoints[epoch]:
                    raise ValueError("Checkpoint changed before evaluation")
                bank_path = directory / "seed_banks" / f"{bank}.json"
                if file_digest(bank_path) != manifest["seed_bank_sha256"][bank]:
                    raise ValueError("Seed bank changed before evaluation")
                job["phase"] = label
                atomic_json(directory / "status.json", status)
                run_mpi(manifest["mpirun"], run["workers"], "comparison_evaluate.py",
                        ["--checkpoint", checkpoint, "--seed-bank", bank_path,
                         "--output", output, "--guard-steps", run["evaluation_guard_steps"]],
                        output, output.with_suffix(".log"))
                verify_evaluation(output, checkpoint, checkpoints[epoch], bank_path, run, bank)
            job.update(status="completed", phase="completed")
            atomic_json(directory / "status.json", status)
        status.update(status="completed", finished_utc=utc_now())
    except BaseException as error:
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status.update(status=state, error=str(error), finished_utc=utc_now())
        if job is not None:
            job["status"] = state
        raise
    finally:
        atomic_json(directory / "status.json", status)


def interrupt_job(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=(101, 102, 103))
    parser.add_argument("--mpirun", default="mpirun")
    args = parser.parse_args(argv)
    config = validate_config(read_json(args.config))
    runs = resolve_runs(config, args.condition, args.seed)
    if args.dry_run:
        print(json.dumps({"runs": runs, "checkpoint_epochs": checkpoint_epochs(config),
                          "evaluation_episodes": {"monitor": config["monitor_episodes"],
                                                  "final": config["final_episodes"]},
                          "seed_bank_generator_seed": SEED_BANK_GENERATOR_SEED}, indent=2))
        return 0
    mpirun = shutil.which(args.mpirun)
    if mpirun is None:
        parser.error("MPI launcher not found: " + args.mpirun)
    directory = (args.study_dir or Path("data") / (
        config["study_id"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )).expanduser().resolve()
    previous_handler = signal.signal(signal.SIGTERM, interrupt_job)
    try:
        manifest, status = prepare_study(config, runs, directory, mpirun,
                                         {"condition": args.condition, "seed": args.seed})
        run_study(directory, manifest, status)
        print(f"Completed study: {directory}", flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"Interrupted study: {directory}", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
