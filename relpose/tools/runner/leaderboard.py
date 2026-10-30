"""Aggregate evaluation reports into the accuracy-vs-compute leaderboard (table + optional plot)."""
import json
from pathlib import Path

import numpy as np


def load_reports(paths):
    """report.json files or directories containing them -> list of (label, report)."""
    reports = []
    for path in paths:
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.rglob("report.json"))
        for file in files:
            report = json.loads(file.read_text())
            label = Path(report["solution"]).stem
            reports.append((label, report))
    return reports


def rows(reports):
    """Per-dataset accuracy paired with the submission's worst-frame fuel across all evaluated datasets."""
    out = []
    for label, report in reports:
        overall = report["summary"]["overall"]
        if "fuel_per_frame_max" not in overall:
            raise ValueError(f"{label}: report has no maximum fuel per frame; rerun `relpose evaluate` with the current evaluator")
        for dataset, summary in sorted(report["summary"]["datasets"].items()):
            if "fuel_per_frame_mean" not in summary or "fuel_per_frame_p99" not in summary:
                raise ValueError(f"{label}: report lacks frame-fuel diagnostics; rerun `relpose evaluate` with the current evaluator")
            out.append({
                "submission": label,
                "dataset": dataset,
                "episodes": summary["episodes"],
                "completed": summary["completed"],
                "seconds": summary["duration_s_median"],
                "rpe_m": summary["rpe_translation_rmse_median"],  # the headline: m over 1 s windows
                "ate_m": summary["ate_rmse_median"],
                "rot_deg": float(np.degrees(summary["rotation_rmse_median"])),
                "fuel_per_frame": overall["fuel_per_frame_max"],
                "fuel_mean": summary["fuel_per_frame_mean"],
                "fuel_median": summary["fuel_per_frame_median"],
                "fuel_p95": summary["fuel_per_frame_p95"],
                "fuel_p99": summary["fuel_per_frame_p99"],
                "ms_per_frame": summary["wall_per_frame_median_ms"],
            })
    return out


def pareto_front(points):
    """Indices of the non-dominated (lower fuel, lower RPE) points."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front, best = [], float("inf")
    for i in order:
        if points[i][1] < best:
            front.append(i)
            best = points[i][1]
    return front


def markdown_table(table):
    header = "| submission | dataset | episodes | s/episode | RPE [m/1 s] | ATE [m] | rot [deg] | max fuel/frame (all datasets) | fuel mean | fuel median | fuel p95 | fuel p99 | ms/frame |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for row in sorted(table, key=lambda r: (r["dataset"], r["rpe_m"])):
        lines.append(f"| {row['submission']} | {row['dataset']} | {row['completed']}/{row['episodes']} | {row['seconds']:.0f} | {row['rpe_m']:.3f} | {row['ate_m']:.3f} | "
                     f"{row['rot_deg']:.1f} | {row['fuel_per_frame']:,.0f} | {row['fuel_mean']:,.0f} | {row['fuel_median']:,.0f} | "
                     f"{row['fuel_p95']:,.0f} | {row['fuel_p99']:,.0f} | {row['ms_per_frame']:.1f} |")
    return header + "\n".join(lines) + "\n\nMaximum fuel spans all datasets; mean, median, p95 and p99 are per-frame diagnostics for the row's dataset.\n"


def plot(table, path):
    """Per-dataset RPE vs maximum fuel per frame across all datasets (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in table})
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        points = [(row["fuel_per_frame"], row["rpe_m"], row["submission"]) for row in table if row["dataset"] == dataset]
        front = set(pareto_front([(p[0], p[1]) for p in points]))
        for index, (fuel, rpe, label) in enumerate(points):
            axis.scatter(fuel, rpe, marker="o" if index in front else "x", color="C0" if index in front else "C7")
            axis.annotate(label, (fuel, rpe), textcoords="offset points", xytext=(4, 4), fontsize=8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("max fuel per frame (all datasets)")
        axis.set_ylabel("RPE over 1 s [m] (median over episodes)")
        axis.set_title(dataset)
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path
