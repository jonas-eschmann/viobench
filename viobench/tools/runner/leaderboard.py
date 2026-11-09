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
    """One aggregate accuracy/compute point per submission report."""
    out = []
    for label, report in reports:
        summary = report["summary"]
        out.append({
            "submission": label,
            "episodes": summary["episodes"],
            "completed": summary["completed"],
            "seconds": summary["duration_s_median"],
            "sim3_ate_m": summary["sim3_ate_rmse_mean"],
            "rpe_m": summary["rpe_translation_rmse_median"],
            "ate_m": summary["ate_rmse_median"],
            "rot_deg": float(np.degrees(summary["rotation_rmse_median"])),
            "fuel_per_frame": summary["fuel_per_frame_max"],
            "fuel_mean": summary["fuel_per_frame_mean"],
            "fuel_median": summary["fuel_per_frame_median"],
            "fuel_p95": summary["fuel_per_frame_p95"],
            "fuel_p99": summary["fuel_per_frame_p99"],
            "ms_per_frame": summary["wall_per_frame_median_ms"],
        })
    return out


def pareto_front(points):
    """Indices of finite, non-dominated (lower fuel, lower error) points; equal points share the frontier."""
    order = sorted((i for i, (fuel, error) in enumerate(points) if np.isfinite(fuel) and np.isfinite(error)),
                   key=lambda i: (points[i][0], points[i][1]))
    front, best, best_fuel = [], float("inf"), None
    for i in order:
        fuel, error = points[i]
        if error < best or (error == best and fuel == best_fuel):
            front.append(i)
            best, best_fuel = error, fuel
    return front


def markdown_table(table):
    header = "| submission | episodes | s/episode | mean Sim(3) ATE [m] | RPE [m/1 s] | unaligned ATE [m] | rot [deg] | max fuel/frame | fuel mean | fuel median | fuel p95 | fuel p99 | ms/frame |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for row in sorted(table, key=lambda r: r["sim3_ate_m"]):
        lines.append(f"| {row['submission']} | {row['completed']}/{row['episodes']} | {row['seconds']:.0f} | {row['sim3_ate_m']:.3f} | {row['rpe_m']:.3f} | {row['ate_m']:.3f} | "
                     f"{row['rot_deg']:.1f} | {row['fuel_per_frame']:,.0f} | {row['fuel_mean']:,.0f} | {row['fuel_median']:,.0f} | "
                     f"{row['fuel_p95']:,.0f} | {row['fuel_p99']:,.0f} | {row['ms_per_frame']:.1f} |")
    return header + "\n".join(lines) + "\n"


def plot(table, path):
    """Sim(3) ATE vs maximum fuel per frame across all episodes (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.5, 4))
    points = [(row["fuel_per_frame"], row["sim3_ate_m"], row["submission"]) for row in table
              if np.isfinite(row["fuel_per_frame"]) and np.isfinite(row["sim3_ate_m"])]
    front = set(pareto_front([(p[0], p[1]) for p in points]))
    for index, (fuel, error, label) in enumerate(points):
        axis.scatter(fuel, error, marker="o" if index in front else "x", color="C0" if index in front else "C7")
        axis.annotate(label, (fuel, error), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("max fuel per frame")
    axis.set_ylabel("Sim(3) ATE [m] (mean over episodes)")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
