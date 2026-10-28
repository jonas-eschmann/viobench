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
    """One row per (submission, dataset) with the headline numbers."""
    out = []
    for label, report in reports:
        for dataset, summary in sorted(report["summary"]["datasets"].items()):
            out.append({
                "submission": label,
                "dataset": dataset,
                "episodes": summary["episodes"],
                "completed": summary["completed"],
                "seconds": summary["duration_s_median"],
                "ate_m": summary["ate_rmse_median"],
                "rpe_m_per_s": summary["rpe_translation_rmse_median"],
                "rot_deg": float(np.degrees(summary["rotation_rmse_median"])) if summary["rotation_rmse_median"] == summary["rotation_rmse_median"] else float("nan"),
                "fuel_per_frame": summary["fuel_per_frame_median"],
                "fuel_p95": summary["fuel_per_frame_p95"],
                "ms_per_frame": summary["wall_per_frame_median_ms"],
            })
    return out


def pareto_front(points):
    """Indices of the non-dominated (lower fuel, lower ATE) points."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front, best = [], float("inf")
    for i in order:
        if points[i][1] < best:
            front.append(i)
            best = points[i][1]
    return front


def markdown_table(table):
    header = "| submission | dataset | episodes | s/episode | ATE [m] | RPE [m/s] | rot [deg] | fuel/frame | fuel p95 | ms/frame |\n|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for row in sorted(table, key=lambda r: (r["dataset"], r["ate_m"])):
        lines.append(f"| {row['submission']} | {row['dataset']} | {row['completed']}/{row['episodes']} | {row['seconds']:.0f} | {row['ate_m']:.3f} | {row['rpe_m_per_s']:.3f} | "
                     f"{row['rot_deg']:.1f} | {row['fuel_per_frame']:,.0f} | {row['fuel_p95']:,.0f} | {row['ms_per_frame']:.1f} |")
    return header + "\n".join(lines) + "\n"


def plot(table, path):
    """ATE vs fuel per frame, one panel per dataset (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in table})
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        points = [(row["fuel_per_frame"], row["ate_m"], row["submission"]) for row in table if row["dataset"] == dataset]
        front = set(pareto_front([(p[0], p[1]) for p in points]))
        for index, (fuel, ate, label) in enumerate(points):
            axis.scatter(fuel, ate, marker="o" if index in front else "x", color="C0" if index in front else "C7")
            axis.annotate(label, (fuel, ate), textcoords="offset points", xytext=(4, 4), fontsize=8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("fuel per frame (median)")
        axis.set_ylabel("ATE [m] (median over episodes)")
        axis.set_title(dataset)
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path
