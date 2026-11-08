"""Server presets and aggregation of immutable per-episode results."""
from math import isfinite
from types import SimpleNamespace

import numpy as np

from ...episodes import SETS, public
from ..datasets.sources import reference_hash
from ..runner.leaderboard import pareto_front
from ..runner.run import summarize


def normalize_hashes(values):
    if not isinstance(values, list) or not values:
        raise ValueError("select at least one episode hash")
    hashes = [reference_hash(value) if isinstance(value, str) else None for value in values]
    if None in hashes:
        raise ValueError("episode hashes must be SHA-1 hashes (40 hexadecimal digits)")
    if len(hashes) != len(set(hashes)):
        raise ValueError("episode hashes must be distinct")
    return hashes


def presets():
    return [{"name": name, "hashes": [entry["hash"] for entry in entries]} for name, entries in SETS.items()]


def default_hashes():
    return [entry["hash"] for entry in public]


def results_for(store, submission_id, hashes, jobs):
    """Aggregate only selected episodes, pooling the original attempted-frame fuel samples."""
    episodes = [jobs[digest]["result"] for digest in hashes if digest in jobs and jobs[digest]["result"] is not None]
    fuels = [np.load(store.report_dir(submission_id, episode["hash"]) / "logs" / episode["hash"] / "fuel.npy",
                     allow_pickle=False) for episode in episodes]
    summary = summarize([SimpleNamespace(**episode) for episode in episodes], fuels) if episodes else None
    return {"summary": summary, "episodes": episodes, "partial": len(episodes) != len(hashes)}


def comparison_for(store, hashes, submissions):
    """Compare complete selected reports; only baseline accounts determine the frontier."""
    def valid(value):
        return isinstance(value, (int, float)) and isfinite(value) and value >= 0

    datasets = {}
    for row, jobs in submissions:
        episodes = [jobs[digest]["result"] for digest in hashes if digest in jobs and jobs[digest]["result"] is not None]
        if len(episodes) != len(hashes):
            continue
        scores = [episode["metrics"].get("sim3_ate_rmse") for episode in episodes]
        if not all(valid(score) for score in scores):
            continue  # never compare an aggregate that silently omitted an unscored episode
        summary = results_for(store, row["id"], hashes, jobs)["summary"]
        fuel = summary["overall"]["fuel_per_frame_max"]
        if not valid(fuel):
            continue
        for name, values in summary["datasets"].items():
            score = values["sim3_ate_rmse_median"]
            if valid(score):
                datasets.setdefault(name, []).append({
                    "id": row["id"], "label": row["label"] or row["id"], "team_name": row["team_name"],
                    "team_type": row["team_type"], "own": bool(row["own"]),
                    "fuel_per_frame_max": fuel, "sim3_ate_rmse_median": score,
                })
    result = []
    for name, points in sorted(datasets.items()):
        baselines = [point for point in points if point["team_type"] == "baseline"]
        front = pareto_front([(point["fuel_per_frame_max"], point["sim3_ate_rmse_median"]) for point in baselines])
        result.append({"name": name, "points": points, "frontier": [baselines[index]["id"] for index in front]})
    return {"evaluation_hashes": hashes, "datasets": result}
