"""Complete server settings are validated independently of local partial limits overrides."""
import json
from dataclasses import asdict

import pytest

from viobench.tools.cli import main
from viobench.tools.runner.sandbox import Limits
from viobench.tools.server.config import Config, EvaluationConfig, load


def test_config_normalizes_and_roundtrips_without_sharing_input(tmp_path):
    values = load().as_dict()
    values["uploads"]["daily_quota"] = 0
    values["evaluation"]["episode_timeout_seconds"] = 25
    values["evaluation"]["limits"]["step_wall_seconds"] = 2
    config = Config.from_dict(values)
    assert config.uploads.daily_quota == 0
    assert type(config.evaluation.episode_timeout_seconds) is float
    assert type(config.evaluation.limits.step_wall_seconds) is float
    assert EvaluationConfig.from_dict(config.evaluation.as_dict()) == config.evaluation
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()))
    assert load(path) == config
    values["evaluation"]["limits"]["step_fuel"] = 1
    assert config.evaluation.limits.step_fuel != 1
    assert asdict(Limits()) == load().evaluation.as_dict()["limits"]


@pytest.mark.parametrize("path", [(), ("uploads",), ("evaluation",), ("evaluation", "limits")])
@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_config_requires_exact_fields_at_each_level(path, change):
    values = load().as_dict()
    target = values
    for key in path:
        target = target[key]
    if change == "missing":
        target.pop(next(iter(target)))
    else:
        target["unexpected"] = 1
    with pytest.raises(ValueError, match="missing keys.*unknown keys"):
        Config.from_dict(values)


@pytest.mark.parametrize("path, invalid", [
    (("uploads", "daily_quota"), True),
    (("uploads", "daily_quota"), -1),
    (("uploads", "daily_quota"), 1.5),
    (("uploads", "max_upload_mib"), 0),
    (("uploads", "max_upload_mib"), "128"),
    (("evaluation", "episode_timeout_seconds"), False),
    (("evaluation", "episode_timeout_seconds"), 0),
    (("evaluation", "episode_timeout_seconds"), float("inf")),
    (("evaluation", "limits", "step_fuel"), True),
    (("evaluation", "limits", "step_fuel"), 1.5),
    (("evaluation", "limits", "step_fuel"), 2 ** 53),
    (("evaluation", "limits", "memory_bytes"), -1),
    (("evaluation", "limits", "table_elements"), 0),
    (("evaluation", "limits", "init_fuel"), None),
    (("evaluation", "limits", "init_wall_seconds"), float("nan")),
    (("evaluation", "limits", "step_wall_seconds"), "2"),
    (("evaluation", "limits", "step_wall_seconds"), 10 ** 400),
])
def test_config_rejects_invalid_numeric_values(path, invalid):
    values = load().as_dict()
    target = values
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    with pytest.raises(ValueError, match=path[-1]):
        Config.from_dict(values)


@pytest.mark.parametrize("zone", ["", "Not/A_Time_Zone", True, None])
def test_config_rejects_invalid_quota_time_zone(zone):
    values = load().as_dict()
    values["uploads"]["quota_time_zone"] = zone
    with pytest.raises(ValueError, match="quota_time_zone"):
        Config.from_dict(values)


@pytest.mark.parametrize("path", [(), ("uploads",), ("evaluation",), ("evaluation", "limits")])
def test_config_requires_objects(path):
    values = load().as_dict()
    if not path:
        values = []
    elif len(path) == 1:
        values[path[0]] = []
    else:
        values[path[0]][path[1]] = []
    with pytest.raises(ValueError, match="must be an object"):
        Config.from_dict(values)


def test_evaluate_cli_passes_explicit_limits_file(tmp_path, monkeypatch):
    from viobench.tools.runner import run

    values = load().evaluation.as_dict()["limits"]
    values["step_fuel"] = 4321
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(values))
    captured = []

    def evaluate(wasm, episodes, **kwargs):
        captured.append(kwargs["limits"])
        return {"summary": {"episodes": 1, "completed": 1, "sim3_ate_rmse_mean": 0,
                            "fuel_per_frame_max": 1, "fuel_per_frame_mean": 1, "fuel_per_frame_median": 1,
                            "fuel_per_frame_p95": 1, "fuel_per_frame_p99": 1}}

    monkeypatch.setattr(run, "evaluate", evaluate)
    assert main(["evaluate", "unused.wasm", "--limits", str(path)]) == 0
    assert asdict(captured[0]) == values


@pytest.mark.parametrize("content, message", [('{"unexpected": 1}', "unknown limits"),
                                             ('["step_fuel"]', "limits must be a JSON object")])
def test_evaluate_cli_rejects_invalid_limits_file_before_running(tmp_path, monkeypatch, capsys, content, message):
    from viobench.tools.runner import run

    path = tmp_path / "limits.json"
    path.write_text(content)
    monkeypatch.setattr(run, "evaluate", lambda *args, **kwargs: pytest.fail("evaluation must not start"))
    assert main(["evaluate", "unused.wasm", "--limits", str(path)]) == 1
    assert message in capsys.readouterr().err
