"""Real WASM workloads exercise worst-frame scoring, including sparse spikes and failed calls."""
import json

import numpy as np
import pytest
import wasmtime

from viobench import Limits, abi
from viobench.tools.cli import main
from viobench.tools.runner.leaderboard import markdown_table, rows
from viobench.tools.runner.run import evaluate
from viobench.tools.server.worker import public_results
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


def workload(dataset, name, counts):
    episode = synthetic_episode(n_frames=len(counts), width=320, height=240, channels=3)
    episode.meta["dataset"] = dataset
    episode.name = name
    episode.frames[:, 0, 0, 0] = counts
    return episode


def test_peak_fuel_across_frames_episodes_and_datasets(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default output directory writes a report without rendering videos
    wasm = metered_module(tmp_path)
    counts = [1] * 24
    counts[11] = 200  # one expensive frame: even p95 misses it
    episodes = [workload("spiky", "peak", counts), workload("spiky", "cheap", [1] * 24),
                workload("steady", "medium", [5] * 24)]
    messages = []
    report = evaluate(wasm, episodes, engine=engine, log=messages.append)
    peak, cheap, medium = report["episodes"]
    maximum = peak["fuel_per_frame_max"]
    assert maximum > 100 * peak["fuel_per_frame_p95"]
    assert maximum > medium["fuel_per_frame_max"] > cheap["fuel_per_frame_max"] > 0
    assert peak["fuel_init"] > maximum  # setup is reported separately from per-frame work
    summary = report["summary"]
    assert summary["overall"]["fuel_per_frame_max"] == maximum
    assert summary["datasets"]["spiky"]["fuel_per_frame_max"] == maximum
    assert summary["datasets"]["steady"]["fuel_per_frame_max"] == medium["fuel_per_frame_max"]
    saved = json.loads((tmp_path / "report-metered" / "report.json").read_text())
    table = rows([("metered", saved)])
    assert len(table) == 2 and all(row["fuel_per_frame"] == maximum for row in table)
    assert "max fuel/frame (all datasets)" in markdown_table(table)
    assert all("fuel/frame max" in message for message in messages[:-1])
    public = public_results(saved)
    assert public["summary"]["overall"]["fuel_per_frame_max"] == maximum
    assert public["episodes"][0]["fuel_per_frame_max"] == maximum
    for field in ("frames_attempted", "fuel_per_frame_mean", "fuel_per_frame_median", "fuel_per_frame_p95",
                  "fuel_per_frame_p99", "fuel_total"):
        assert public["episodes"][0][field] == peak[field]
        assert public["summary"]["overall"][field] == summary["overall"][field]


def test_diagnostics_pool_frames_from_unequal_episodes(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wasm = metered_module(tmp_path)
    report = evaluate(wasm, [workload("mixed", "cheap", [1] * 24), workload("mixed", "longer", [5] * 48),
                             workload("expensive", "high", [20] * 24)], engine=engine, log=lambda *a: None)
    costs = [episode["fuel_per_frame_max"] for episode in report["episodes"]]
    pooled = np.repeat(costs, [24, 48, 24])
    overall, mixed = report["summary"]["overall"], report["summary"]["datasets"]["mixed"]
    for summary, samples in ((overall, pooled), (mixed, pooled[:72])):
        assert summary["frames_attempted"] == len(samples)
        assert summary["fuel_total"] == sum(samples)
        assert summary["fuel_per_frame_mean"] == pytest.approx(np.mean(samples))
        for name, percentile in (("median", 50), ("p95", 95), ("p99", 99)):
            assert summary[f"fuel_per_frame_{name}"] == pytest.approx(np.percentile(samples, percentile))
    assert overall["fuel_per_frame_mean"] != pytest.approx(np.mean(costs))  # episode means are not equally weighted
    assert mixed["fuel_per_frame_median"] == costs[1]  # median of episode medians would fall between the two costs
    table = rows([("metered", report)])
    mixed_row = next(row for row in table if row["dataset"] == "mixed")
    assert mixed_row["fuel_per_frame"] == costs[2]  # judging still uses the global maximum
    for statistic in ("mean", "median", "p95", "p99"):
        assert mixed_row[f"fuel_{statistic}"] == mixed[f"fuel_per_frame_{statistic}"]
        assert f"fuel {statistic}" in markdown_table(table)


def test_no_attempted_frames_has_zero_fuel_statistics(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = evaluate(metered_module(tmp_path, init_status=-1), [workload("broken", "init", [1] * 24)],
                      engine=engine, log=lambda *a: None)
    assert report["episodes"][0]["failure"].startswith("init failed")
    for summary in (report["episodes"][0], report["summary"]["overall"], report["summary"]["datasets"]["broken"]):
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
    report = evaluate(wasm, [workload("steady", "ok", [1] * 24), workload("spiky", "failure", counts)],
                      engine=engine, limits=Limits(step_fuel=limit), log=lambda *a: None)
    cheap, failed = report["episodes"]
    assert cheap["completed"] and not failed["completed"] and failed["failure"].startswith(reason)
    maximum = failed["fuel_per_frame_max"]
    assert maximum > cheap["fuel_per_frame_max"]
    assert failed["fuel_total"] == failed_frame * cheap["fuel_per_frame_max"] + maximum
    assert failed["fuel_per_frame_mean"] == pytest.approx(failed["fuel_total"] / (failed_frame + 1))
    assert failed["frames_attempted"] == failed_frame + 1
    assert failed["metrics"]["held"] == 24 - failed_frame
    overall = report["summary"]["overall"]
    assert overall["fuel_per_frame_max"] == maximum
    assert overall["frames_attempted"] == 24 + failed_frame + 1
    assert overall["fuel_total"] == cheap["fuel_total"] + failed["fuel_total"]
    assert overall["fuel_per_frame_mean"] == pytest.approx(overall["fuel_total"] / overall["frames_attempted"])
    attempted_costs = [cheap["fuel_per_frame_max"]] * (24 + failed_frame) + [maximum]
    assert overall["fuel_per_frame_p99"] == pytest.approx(np.percentile(attempted_costs, 99))
    if limit == 1_000:
        assert maximum == limit


@pytest.mark.parametrize("overall", [{"fuel_per_frame_median": 10}, {"fuel_per_frame_max": 20}])
def test_old_report_requires_reevaluation(overall, tmp_path, capsys):
    report = {"solution": "old.wasm", "summary": {"overall": overall, "datasets": {"set": {"fuel_per_frame_median": 10}}}}
    with pytest.raises(ValueError, match="rerun `viobench evaluate`"):
        rows([("old", report)])
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert main(["leaderboard", str(path)]) == 1
    assert "rerun `viobench evaluate`" in capsys.readouterr().err
