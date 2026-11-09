"""Real WASM workloads exercise worst-frame scoring, including sparse spikes and failed calls."""
import json

import numpy as np
import pytest
import wasmtime

from viobench import Limits, abi
from viobench.tools.runner.leaderboard import markdown_table, pareto_front, rows
from viobench.tools.runner.run import evaluate, summarize
from viobench.tools.server.worker import public_episode
from tests.test_datasets import synthetic_episode


def metered_module(tmp_path, failure="", init_status=0):
    """The first image byte controls work; 255 also triggers the requested failure after doing work."""
    stores = "\n".join(f"(f64.store offset={8 * i} (local.get $out) (f64.const {int(i == 6)}))"
                       for i in range(abi.STATE_VALUES))
    wasm = tmp_path / "metered.wasm"
    wasm.write_bytes(wasmtime.wat2wasm(f"""
(module
  (memory (export "memory") 16)
  (global $next (mut i32) (i32.const 1024))
  (func (export "abi_version") (result i32) (i32.const {abi.ABI_VERSION}))
  (func (export "alloc") (param $size i32) (result i32) (local $pointer i32)
    (local.set $pointer (global.get $next))
    (global.set $next (i32.and (i32.add (i32.add (global.get $next) (local.get $size)) (i32.const 7)) (i32.const -8)))
    (local.get $pointer))
  (func $burn (param $n i32)
    (loop $again
      (local.set $n (i32.sub (local.get $n) (i32.const 1)))
      (br_if $again (local.get $n))))
  (func (export "init") (param i32 i32) (result i32)
    (call $burn (i32.const 10000))
    (i32.const {init_status}))
  (func (export "step") (param f64) (param $frame i32) (param i32 i32) (param $out i32) (result i32)
    (local $work i32)
    (local.set $work (i32.load8_u (local.get $frame)))
    (call $burn (i32.mul (local.get $work) (i32.const 10)))
    {stores}
    (if (i32.eq (local.get $work) (i32.const 255)) (then {failure}))
    (i32.const 0)))
"""))
    return wasm


def workload(name, counts):
    episode = synthetic_episode(n_frames=len(counts), width=640, height=480, channels=3)
    episode.name = name
    episode.frames[:, :2, :2, 0] = np.asarray(counts)[:, None, None]
    return episode


def test_peak_fuel_across_frames_and_episodes(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default output directory writes a report without rendering videos
    wasm = metered_module(tmp_path)
    counts = [1] * 24
    counts[11] = 200  # one expensive frame: even p95 misses it
    episodes = [workload("peak", counts), workload("cheap", [1] * 24),
                workload("medium", [5] * 24)]
    messages = []
    report = evaluate(wasm, episodes, engine=engine, log=messages.append)
    peak, cheap, medium = report["episodes"]
    maximum = peak["fuel_per_frame_max"]
    assert maximum > 100 * peak["fuel_per_frame_p95"]
    assert maximum > medium["fuel_per_frame_max"] > cheap["fuel_per_frame_max"] > 0
    assert peak["fuel_init"] > maximum  # setup is reported separately from per-frame work
    summary = report["summary"]
    assert summary["fuel_per_frame_max"] == maximum
    assert "datasets" not in summary and "overall" not in summary
    saved = json.loads((tmp_path / "report-metered" / "report.json").read_text())
    table = rows([("metered", saved)])
    assert len(table) == 1 and table[0]["fuel_per_frame"] == maximum
    assert table[0]["sim3_ate_m"] == summary["sim3_ate_rmse_mean"]
    assert "max fuel/frame" in markdown_table(table) and "dataset" not in markdown_table(table)
    assert "Sim(3) ATE [m]" in markdown_table(table)
    assert all("fuel/frame max" in message for message in messages[:-1])
    public = public_episode(saved["episodes"][0])
    assert public["fuel_per_frame_max"] == maximum
    for field in ("frames_attempted", "fuel_per_frame_mean", "fuel_per_frame_median", "fuel_per_frame_p95",
                  "fuel_per_frame_p99", "fuel_total"):
        assert public[field] == peak[field]


def test_diagnostics_pool_frames_from_unequal_episodes(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wasm = metered_module(tmp_path)
    report = evaluate(wasm, [workload("cheap", [1] * 24), workload("longer", [5] * 48),
                             workload("high", [20] * 24)], engine=engine, log=lambda *a: None)
    costs = [episode["fuel_per_frame_max"] for episode in report["episodes"]]
    pooled = np.repeat(costs, [24, 48, 24])
    summary = report["summary"]
    scores = [episode["metrics"]["sim3_ate_rmse"] for episode in report["episodes"]]
    assert summary["sim3_ate_rmse_mean"] == pytest.approx(np.mean(scores))
    assert summary["frames_attempted"] == len(pooled)
    assert summary["fuel_total"] == sum(pooled)
    assert summary["fuel_per_frame_mean"] == pytest.approx(np.mean(pooled))
    for name, percentile in (("median", 50), ("p95", 95), ("p99", 99)):
        assert summary[f"fuel_per_frame_{name}"] == pytest.approx(np.percentile(pooled, percentile))
    assert summary["fuel_per_frame_mean"] != pytest.approx(np.mean(costs))  # episode means are not equally weighted
    [row] = rows([("metered", report)])
    assert row["fuel_per_frame"] == costs[2]
    for statistic in ("mean", "median", "p95", "p99"):
        assert row[f"fuel_{statistic}"] == summary[f"fuel_per_frame_{statistic}"]
        assert f"fuel {statistic}" in markdown_table([row])


def test_no_attempted_frames_has_zero_fuel_statistics(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = evaluate(metered_module(tmp_path, init_status=-1), [workload("init", [1] * 24)],
                      engine=engine, log=lambda *a: None)
    assert report["episodes"][0]["failure"].startswith("init failed")
    assert report["episodes"][0]["metrics"]["filled_frames"] == 24
    assert np.isfinite(report["summary"]["sim3_ate_rmse_mean"])
    assert rows([("init failure", report)])[0]["fuel_per_frame"] == 0
    for summary in (report["episodes"][0], report["summary"]):
        assert summary["frames_attempted"] == summary["fuel_total"] == 0
        for statistic in ("max", "mean", "median", "p95", "p99"):
            assert summary[f"fuel_per_frame_{statistic}"] == 0


@pytest.mark.parametrize("failure, limit, reason", [
    ("unreachable", 100_000, "trap"),
    ("(return (i32.const -1))", 100_000, "step failed"),
    ("(f64.store (local.get $out) (f64.const nan))", 100_000, "step returned an unusable state"),
    ("(loop $forever (br $forever))", 1_000, "trap"),
])
@pytest.mark.parametrize("failed_frame", [0, 2])
def test_failed_frame_contributes_to_peak_fuel(failure, limit, reason, failed_frame, engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wasm = metered_module(tmp_path, failure)
    counts = [1] * 24
    counts[failed_frame] = 255
    report = evaluate(wasm, [workload("ok", [1] * 24), workload("failure", counts)],
                      engine=engine, limits=Limits(step_fuel=limit), log=lambda *a: None)
    cheap, failed = report["episodes"]
    assert cheap["completed"] and not failed["completed"] and failed["failure"].startswith(reason)
    maximum = failed["fuel_per_frame_max"]
    assert maximum > cheap["fuel_per_frame_max"]
    assert failed["fuel_total"] == failed_frame * cheap["fuel_per_frame_max"] + maximum
    assert failed["fuel_per_frame_mean"] == pytest.approx(failed["fuel_total"] / (failed_frame + 1))
    assert failed["frames_attempted"] == failed_frame + 1
    assert failed["metrics"]["filled_frames"] == 24 - failed_frame
    overall = report["summary"]
    assert overall["fuel_per_frame_max"] == maximum
    assert overall["frames_attempted"] == 24 + failed_frame + 1
    assert overall["fuel_total"] == cheap["fuel_total"] + failed["fuel_total"]
    assert overall["fuel_per_frame_mean"] == pytest.approx(overall["fuel_total"] / overall["frames_attempted"])
    attempted_costs = [cheap["fuel_per_frame_max"]] * (24 + failed_frame) + [maximum]
    assert overall["fuel_per_frame_p99"] == pytest.approx(np.percentile(attempted_costs, 99))
    if limit == 1_000:
        assert maximum == limit


def test_pareto_front_keeps_equal_points_and_ignores_non_finite_metrics():
    points = [(10, 3), (10, 3), (10, 4), (20, 3), (20, 2), (0, 5), (float("nan"), 0), (30, float("inf"))]
    assert pareto_front(points) == [5, 0, 1, 4]
    assert pareto_front([(0, 0), (1, 0), (0, 1)]) == [0]



def test_summary_aggregates_every_hash_with_mean_ate_and_pooled_fuel():
    from types import SimpleNamespace

    metric_names = ("sim3_ate_rmse", "rpe_translation_rmse", "rpe_rotation_rmse", "ate_rmse",
                    "rotation_rmse", "velocity_rmse", "angular_velocity_rmse")
    reports = [SimpleNamespace(hash=digest * 40, completed=True, duration_s=length,
                               wall_per_frame_median_ms=1, metrics=dict.fromkeys(metric_names, score))
               for digest, length, score in (("a", 1, 1), ("b", 7, 100), ("c", 2, 3))]
    fuels = [np.array([1]), np.repeat(2, 7), np.repeat(100, 2)]
    summary = summarize(reports, fuels)
    assert summary["episodes"] == summary["completed"] == 3
    assert summary["sim3_ate_rmse_mean"] == pytest.approx(104 / 3)  # each episode has equal weight, irrespective of frame count
    assert summary["ate_rmse_median"] == 3
    assert summary["frames_attempted"] == 10
    assert summary["fuel_per_frame_mean"] == 21.5
    assert summary["fuel_per_frame_median"] == 2
    assert summary["fuel_per_frame_max"] == 100
    assert "overall" not in summary and "datasets" not in summary
    with pytest.raises(ValueError):
        summarize(reports, fuels[:-1])
