"""Recompute descriptive aggregation results from saved episode outcomes.

No environments, training libraries, MPI workers, or checkpoints are executed.
Matplotlib is imported only when --plots is requested.
"""

import argparse
import math
from pathlib import Path
import statistics

from .analysis_io import file_sha256, load_dataset, outcome_metrics, write_json


LABELS = {"frequency_wta": "Frequency WTA", "count_weighted_mean": "Count-weighted mean",
          "uniform_mean": "Uniform mean", "random_visited": "Random visited"}
COLORS = {"frequency_wta": "#0072B2", "count_weighted_mean": "#E69F00",
          "uniform_mean": "#009E73", "random_visited": "#CC79A7"}
STYLES = {"frequency_wta": "-", "count_weighted_mean": "--", "uniform_mean": "-.", "random_visited": ":"}


def descriptive(values):
    return {"n_training_seeds": len(values), "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None}


def normalized_auc(epochs, values):
    """Trapezoidal area divided by the measured epoch window, without extrapolation."""
    if len(epochs) != len(values) or len(epochs) < 2:
        raise ValueError("AUC needs at least two matching measured epochs and values")
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Epochs must increase strictly")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("Landing fractions must be finite and in [0, 1]")
    return sum((b - a) * (u + v) / 2 for a, b, u, v in
               zip(epochs, epochs[1:], values, values[1:])) / (epochs[-1] - epochs[0])


def crossing_summary(epochs, values, threshold):
    reached = [value >= threshold - 1e-12 for value in values]
    return {"first_observed_epoch": next((epoch for epoch, hit in zip(epochs, reached) if hit), None),
            "first_two_consecutive_checks_start": next(
                (epochs[i] for i in range(len(epochs) - 1) if all(reached[i:i + 2])), None),
            "first_epoch_above_at_all_remaining_checks": next(
                (epochs[i] for i in range(len(epochs)) if all(reached[i:])), None)}


def summarize(dataset):
    config = dataset["config"]
    runs = {(run["condition"], run["training_seed"]): run for run in dataset["runs"]}
    final_rows, monitoring = [], []
    for assessment in dataset["evaluations"]:
        run = runs[assessment["condition"], assessment["training_seed"]]
        point = next(item for item in run["checkpoints"] if item["completed_epochs"] == assessment["completed_epochs"])
        metrics = outcome_metrics(assessment, config, dataset["seed_banks"])
        row = {"condition": assessment["condition"], "training_seed": assessment["training_seed"],
               "completed_epochs": assessment["completed_epochs"], "evaluation": metrics,
               "actual_training_steps": point["actual_training_steps"], "replay_updates": point["replay_updates"],
               "checkpoint_sha256": assessment["checkpoint_sha256"],
               "original_results_sha256": assessment["original_results_sha256"]}
        if assessment["bank"] == "final":
            row["training_episodes"] = run["training_episodes"]
            final_rows.append(row)
        else:
            monitoring.append(row)
    condition_rows = []
    for condition in config["conditions"]:
        rows = [row for row in final_rows if row["condition"] == condition]
        if not rows:
            continue
        condition_rows.append({"condition": condition, "training_seeds": [row["training_seed"] for row in rows],
                               "observed_success_fraction": descriptive([row["evaluation"]["observed_success_fraction"] for row in rows]),
                               "mean_censoring_bounds": [statistics.mean(row["evaluation"]["success_fraction_bounds"][i]
                                                                          for row in rows) for i in (0, 1)]})
    lookup = {(row["condition"], row["training_seed"]): row["evaluation"] for row in final_rows}
    paired = {}
    for condition in config["conditions"]:
        if condition == "frequency_wta":
            continue
        pairs = []
        for seed in config["training_seeds"]:
            if ("frequency_wta", seed) not in lookup or (condition, seed) not in lookup:
                continue
            wta, other = lookup["frequency_wta", seed], lookup[condition, seed]
            wl, wu = wta["success_fraction_bounds"]
            ol, ou = other["success_fraction_bounds"]
            pairs.append({"training_seed": seed, "observed_success_fraction_difference": wl - ol,
                          "difference_censoring_bounds": [wl - ou, wu - ol]})
        if pairs:
            paired[condition] = {"pairs": pairs, "difference": descriptive(
                [pair["observed_success_fraction_difference"] for pair in pairs])}
    return {"schema_version": 1, "study_id": config["study_id"], "selection": dataset.get("selection", {}),
            "full_declared_matrix": len(runs) == len(config["conditions"]) * len(config["training_seeds"]),
            "completed_runs": len(runs), "planned_epochs": config["epochs"],
            "evaluation_assessments": len(dataset["evaluations"]),
            "raw_evaluation_episodes": sum(len(item["episodes"]) for item in dataset["evaluations"]),
            "final_rows": final_rows, "monitoring_curves": monitoring,
            "condition_descriptives": condition_rows, "paired_seed_differences": paired,
            "interpretation": {"replication_unit": "training seed, not worker or evaluation episode",
                               "censoring": "included in assigned denominator; bounds are not confidence intervals",
                               "raw_returns": "reporting only; include partial returns for censored episodes",
                               "scope": "fixed-budget observations; no inferred convergence epoch or significance test"}}


def learning_speed(summary):
    rows = summary["monitoring_curves"]
    epochs = sorted({row["completed_epochs"] for row in rows})
    thresholds = (0.50, 0.60, 0.70, 0.75, 0.80)
    groups = {}
    for row in rows:
        groups.setdefault((row["condition"], row["training_seed"]), []).append(row)
    conditions = {}
    for (condition, seed), points in groups.items():
        points.sort(key=lambda row: row["completed_epochs"])
        curve = [row["evaluation"]["observed_success_fraction"] for row in points]
        upper = [row["evaluation"]["success_fraction_bounds"][1] for row in points]
        conditions.setdefault(condition, {"seeds": []})["seeds"].append({
            "training_seed": seed, "observed_landing_fractions": curve,
            "normalized_auc": normalized_auc(epochs, curve) if len(epochs) >= 2 else None,
            "normalized_auc_censoring_bounds": [normalized_auc(epochs, values) for values in (curve, upper)]
            if len(epochs) >= 2 else None,
            "thresholds": {str(t): crossing_summary(epochs, curve, t) for t in thresholds}})
    for entry in conditions.values():
        seeds = entry["seeds"]
        curve = [statistics.mean(row["observed_landing_fractions"][i] for row in seeds) for i in range(len(epochs))]
        entry["mean_observed_landing_fractions"] = curve
        entry["mean_curve_thresholds"] = {str(t): crossing_summary(epochs, curve, t) for t in thresholds}
        entry["mean_normalized_auc"] = statistics.mean(row["normalized_auc"] for row in seeds) if len(epochs) >= 2 else None
        entry["sample_sd_normalized_auc"] = statistics.stdev(row["normalized_auc"] for row in seeds) if len(seeds) > 1 and len(epochs) >= 2 else None
        entry["mean_normalized_auc_censoring_bounds"] = [
            statistics.mean(row["normalized_auc_censoring_bounds"][i] for row in seeds) for i in (0, 1)
        ] if len(epochs) >= 2 else None
    contrasts = {}
    if "frequency_wta" in conditions and len(epochs) >= 2:
        wta = {row["training_seed"]: row for row in conditions["frequency_wta"]["seeds"]}
        for condition, entry in conditions.items():
            if condition == "frequency_wta":
                continue
            pairs = []
            for other in entry["seeds"]:
                seed = other["training_seed"]
                if seed not in wta:
                    continue
                wl, wu = wta[seed]["normalized_auc_censoring_bounds"]
                ol, ou = other["normalized_auc_censoring_bounds"]
                pairs.append({"training_seed": seed, "normalized_auc_difference": wl - ol,
                              "difference_censoring_bounds": [wl - ou, wu - ol]})
            if pairs:
                mean_other = entry["mean_normalized_auc"]
                contrasts[condition] = {"paired_wta_minus_other": pairs,
                    "mean_normalized_auc_difference": statistics.mean(row["normalized_auc_difference"] for row in pairs),
                    "relative_auc_gain_not_speed_multiplier": conditions["frequency_wta"]["mean_normalized_auc"] / mean_other - 1
                    if mean_other else None}
    return {"schema_version": 1, "conditions": conditions, "wta_contrasts": contrasts,
            "method": {"analysis_status": "post-study descriptive, not confirmatory",
                       "fixed_training_budget": summary["planned_epochs"], "observed_epochs": epochs,
                       "auc_window": [epochs[0], epochs[-1]],
                       "auc_rule": "trapezoidal area divided by observed epoch span; no epoch-zero extrapolation",
                       "evaluation_bank": "monitoring bank only; final evaluation excluded",
                       "thresholds": thresholds, "threshold_status": "post-study descriptive benchmarks",
                       "crossings": "first observed checks, not exact or permanent convergence times",
                       "censoring": "retained in denominator; lower and upper AUC bounds included",
                       "speed": "relative AUC gain is not an epoch or wall-clock speed multiplier"}}


def plot_curves(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = summary["monitoring_curves"]
    seeds = sorted({row["training_seed"] for row in rows})
    epochs = sorted({row["completed_epochs"] for row in rows})
    conditions = [row["condition"] for row in summary["condition_descriptives"]]
    values = {(row["condition"], row["training_seed"], row["completed_epochs"]):
              row["evaluation"]["observed_success_fraction"] for row in rows}
    noun = "seed" if len(seeds) == 1 else "seeds"
    panels = [(None, f"Mean across {len(seeds)} training {noun}")] + [(seed, f"Training seed {seed}") for seed in seeds]
    nrows = math.ceil(len(panels) / 2)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "sparse-parallel-q"}):
        figure, axes = plt.subplots(nrows, 2, figsize=(13, 4 * nrows), squeeze=False, sharex=True, sharey=True)
        for axis, (seed, title) in zip(axes.flat, panels):
            for condition in conditions:
                selected = [value for value in seeds if (condition, value, epochs[0]) in values] if seed is None else [seed]
                if not selected or (condition, selected[0], epochs[0]) not in values:
                    continue
                fractions = [statistics.mean(values[condition, value, epoch] for value in selected) for epoch in epochs]
                axis.plot(epochs, fractions, label=LABELS.get(condition, condition), color=COLORS.get(condition),
                          linestyle=STYLES.get(condition, "-"), linewidth=2, marker="o", markersize=3.8)
            axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=9)
            axis.set_xlim(0, summary["planned_epochs"])
            axis.set_ylim(0, 1)
            ticks = sorted(set([0, *epochs[1::2], epochs[-1]]))
            ticks = [tick for tick in ticks if tick == epochs[-1] or epochs[-1] - tick >= 0.10 * epochs[-1]]
            axis.set_xticks(ticks)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
            axis.set_axisbelow(True)
        for axis in list(axes.flat)[len(panels):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.suptitle("Q-table aggregation under sparse terminal rewards", x=0.08, ha="left", y=0.985,
                       fontsize=17, fontweight="semibold")
        count = rows[0]["evaluation"]["episodes"]
        figure.text(0.08, 0.933, f"LunarLander-v3 | Common {count}-episode monitoring bank", fontsize=11, color="#4A5560")
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.919),
                      ncol=len(conditions), frameon=False, fontsize=10, handlelength=2.6)
        figure.supxlabel("Completed training epochs", y=0.085, fontsize=11)
        figure.supylabel("Observed landing fraction", x=0.018, fontsize=11)
        figure.text(0.08, 0.036, "Censored episodes remain in the denominator. The separate final evaluation bank is not plotted.",
                    fontsize=9, color="#4A5560")
        figure.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.825, wspace=0.17, hspace=0.30)
        try:
            for extension in ("png", "svg"):
                metadata = {"Title": "Q-table aggregation under sparse terminal rewards"}
                if extension == "svg":
                    metadata["Date"] = None
                figure.savefig(Path(output) / ("monitoring_landing_fraction." + extension),
                               dpi=220, facecolor="white", metadata=metadata)
        finally:
            plt.close(figure)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Portable outcome bundle or completed native study directory")
    parser.add_argument("--output-dir", type=Path, help="Derived files; defaults to STUDY/analysis")
    parser.add_argument("--plots", action="store_true", help="Also produce PNG/SVG curves; needs the analysis extra")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.study)
    summary = summarize(dataset)
    speed = learning_speed(summary)
    output = args.output_dir or args.study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "learning_speed.json", speed)
    plot_version = plot_curves(summary, output) if args.plots else None
    names = ["summary.json", "learning_speed.json"]
    if args.plots:
        names += ["monitoring_landing_fraction.png", "monitoring_landing_fraction.svg"]
    write_json(output / "provenance.json", {
        "schema_version": 1, "input_files": dataset["input_hashes"],
        "analysis_sources": [{"path": "sparse_parallel_q/" + name,
                              "sha256": file_sha256(Path(__file__).with_name(name))}
                             for name in ("analyze.py", "analysis_io.py")],
        "matplotlib_version": plot_version,
        "validation": "Input hashes, complete selected matrix, raw outcomes, seeds and censoring denominators verified",
        "checkpoint_scope": "Native studies verify checkpoint bytes; portable bundles retain historical checkpoint hashes only",
        "outputs": [{"path": name, "sha256": file_sha256(output / name)} for name in names]})
    print(f"Validated {summary['completed_runs']} runs and {summary['raw_evaluation_episodes']} episode records.")
    print(f"Wrote summary and descriptive AUC to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
