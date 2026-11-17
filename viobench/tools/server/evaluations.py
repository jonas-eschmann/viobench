"""Server presets and aggregation of per-episode results for the current rules."""
from math import isfinite, log
from types import SimpleNamespace

import numpy as np

from ...episodes import SETS, public
from ..datasets.sources import reference_hash
from ..runner.leaderboard import pareto_front
from ..runner.run import summarize

FUEL_FLOOR = 1
ERROR_FLOOR = 0.001
EPISODE_FIELDS = ("hash", "frames", "frames_attempted", "duration_s", "completed", "metrics", "fuel_init", "fuel_per_frame_max", "fuel_per_frame_median",
                  "fuel_per_frame_p95", "fuel_per_frame_p99", "fuel_per_frame_mean", "fuel_total", "wall_init_s", "wall_per_frame_median_ms",
                  "wall_per_frame_p95_ms")


def public_episode(episode):
    """The episode metrics exposed to teams; no source labels or guest-written text."""
    return {key: episode[key] for key in EPISODE_FIELDS}


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
    outdated = any(jobs[digest]["outdated"] for digest in hashes if digest in jobs and jobs[digest]["status"] in ("done", "failed"))
    return {"summary": summary, "episodes": episodes, "partial": len(episodes) != len(hashes), "outdated": outdated}


def dominated_area(points, max_fuel, max_error, *, min_fuel=FUEL_FLOOR, min_error=ERROR_FLOOR):
    """Dominated area in natural-log coordinates, clamping points to the positive lower reference bounds."""
    if max_fuel <= min_fuel or max_error <= min_error:
        return 0.0
    max_x, max_y = log(max_fuel / min_fuel), log(max_error / min_error)
    ordered = sorted((log(max(fuel, min_fuel) / min_fuel), log(max(error, min_error) / min_error))
                     for fuel, error in points
                     if isfinite(fuel) and isfinite(error) and 0 <= fuel < max_fuel and 0 <= error < max_error)
    area, previous_x, best_y = 0.0, 0.0, max_y
    for x, y in ordered:
        if y >= best_y:
            continue
        area += (x - previous_x) * (max_y - best_y)
        previous_x, best_y = x, y
    return area + (max_x - previous_x) * (max_y - best_y)


def comparison_for(store, hashes, submissions, evaluation_config):
    """Compare complete reports and credit the best single submission's gain beyond the baselines."""
    def valid(value):
        return isinstance(value, (int, float)) and isfinite(value) and value >= 0

    points = []
    for row, jobs in submissions:
        episodes = [jobs[digest]["result"] for digest in hashes if digest in jobs and jobs[digest]["result"] is not None]
        if len(episodes) != len(hashes) or any(jobs[digest]["outdated"] for digest in hashes):
            continue
        scores = [episode["metrics"].get("sim3_ate_rmse") for episode in episodes]
        if not all(valid(score) for score in scores):
            continue  # never compare an aggregate that silently omitted an unscored episode
        summary = results_for(store, row["id"], hashes, jobs)["summary"]
        fuel = summary["fuel_per_frame_max"]
        score = summary["sim3_ate_rmse_mean"]
        if not valid(fuel) or not valid(score):
            continue
        points.append({
            "id": row["id"], "label": row["label"] or row["id"], "team_name": row["team_name"],
            "team_type": row["team_type"], "own": bool(row["own"]), "is_baseline": bool(row["is_baseline"]),
            "fuel_per_frame_max": fuel, "sim3_ate_rmse_mean": score,
        })
    fuel_reference = 1.2 * evaluation_config["limits"]["step_fuel"]
    baselines = [point for point in points if point["is_baseline"]]
    coordinates = [(max(point["fuel_per_frame_max"], FUEL_FLOOR), max(point["sim3_ate_rmse_mean"], ERROR_FLOOR))
                   for point in points]
    baseline_coordinates = [(max(point["fuel_per_frame_max"], FUEL_FLOOR), max(point["sim3_ate_rmse_mean"], ERROR_FLOOR))
                            for point in baselines]
    error_reference = 1.2 * max(error for _, error in baseline_coordinates) if baselines else None
    claimed_area, best_submission_id = None, None
    if error_reference is not None:
        baseline_area = dominated_area(baseline_coordinates, fuel_reference, error_reference)
        claimed_area = 0.0
        for point, coordinate in zip(points, coordinates):
            if not point["own"]:
                continue
            gain = max(0.0, dominated_area([*baseline_coordinates, coordinate], fuel_reference, error_reference) - baseline_area)
            # Store supplies creation/id order, so exact ties consistently select the earliest submission.
            if best_submission_id is None or gain > claimed_area:
                claimed_area, best_submission_id = gain, point["id"]
    best_points = [point for point in points if point["is_baseline"] or point["id"] == best_submission_id]
    best_coordinates = [(max(point["fuel_per_frame_max"], FUEL_FLOOR), max(point["sim3_ate_rmse_mean"], ERROR_FLOOR))
                        for point in best_points]
    return {"evaluation_hashes": hashes, "fuel_floor": FUEL_FLOOR, "error_floor": ERROR_FLOOR,
            "fuel_reference": fuel_reference, "points": points,
            "baseline_frontier": [baselines[index]["id"] for index in pareto_front(baseline_coordinates)],
            "frontier": [points[index]["id"] for index in pareto_front(coordinates)],
            "best_submission_id": best_submission_id,
            "best_frontier": [best_points[index]["id"] for index in pareto_front(best_coordinates)],
            "error_reference": error_reference, "claimed_area": claimed_area}
