# Reproducing the experiments

Install the dependencies using the [README](../README.md#installation).
Run the following commands from the repository root with the environment activated.

## Configuration and launch

The bundled [comparison configuration](../sparse_parallel_q/configs/comparison.json)
records the reference protocol: sixteen workers, 264 epochs, 64,000 local steps,
four aggregation rules, and paired training seeds 101, 102, and 103.
The launcher keeps these scientific settings fixed. A condition or seed filter
selects a subset without changing its budget.

```sh
spq-compare --dry-run
spq-compare --execute --condition frequency_wta --seed 101 --study-dir data/wta-101
spq-compare --execute --study-dir data/comparison
```

Conditions are `frequency_wta`, `random_visited`, `count_weighted_mean`, and
`uniform_mean`. The launcher runs their MPI jobs sequentially, using the active
Python interpreter. It starts MPI itself; do not wrap it in another `mpirun`.
Use `--mpirun PATH` if Open MPI is not on the executable search path.

Each launch requires a new output directory and starts from empty Q-tables.
Ctrl-C stops the active job. Partial outputs remain available for inspection;
rerun an interrupted condition in a new directory. Training has no artificial
wall-clock cutoff or episode timeout.

## Recorded outputs

```text
data/comparison/
  config.json
  manifest.json
  status.json
  seed_banks/
    monitor.json
    final.json
  source_snapshot/
  runs/
    frequency_wta/
      seed-101/
        config.json
        training.log
        epochs.jsonl
        rank_metrics/
        run_training_summary.json
        checkpoints/
          epoch-000032.pkl
          ...
          epoch-000264.pkl
        evaluations/
          epoch-000032-monitor/results.json
          ...
          epoch-000264-final/results.json
```

The manifest records the selected runs, source hashes, dependency versions,
and seed banks. Training records interaction and replay counts, native outcomes,
epsilon, and epoch timing. Frozen checkpoints identify the policies evaluated
at each epoch. Evaluation verifies that their contents remain unchanged.

## Evaluate a checkpoint

The evaluator accepts a checkpoint, a JSON seed bank, and a new output directory.
It can run in one process or under MPI.

```sh
python -m sparse_parallel_q.comparison_evaluate \
  --checkpoint data/comparison/runs/frequency_wta/seed-101/checkpoints/epoch-000264.pkl \
  --seed-bank data/comparison/seed_banks/final.json \
  --output data/independent-evaluation \
  --guard-steps 100000
```

Greedy evaluation does not update Q-values. Only load pickle checkpoints whose
source you trust. Changing the seed bank or guard changes the evaluation protocol.
Repeated inspection of the final bank does not create a fresh holdout evaluation.

## Recompute the results

```sh
spq-analyze results/2026-11-12 --output-dir analysis-output --plots
spq-analyze data/comparison --output-dir analysis-output/new-study --plots
```

The first command uses the included records; the second uses a newly completed
study. Both derive the reported metrics from episode outcomes. See
[the results documentation](results.md) for denominators, learning-curve area,
censoring, and interpretation.
