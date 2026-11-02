"""The challenge's rules: `viobench/tools/config.json`, checked in with the code and loaded once by the app and the
worker (restart both after editing it). Every key is required, `limits` with every field of
`viobench.tools.runner.sandbox.Limits`, which also takes its defaults from there: local runs apply the server's limits.
GET /config serves it as loaded.

Deployment details stay in the environment: VIOBENCH_SERVER_DIR (uploads, reports, sqlite) and CONTA_CACHE (where the
worker's episodes are cached). What the worker evaluates is code: `worker.EPISODES`.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from zoneinfo import ZoneInfo

from ..runner.sandbox import Limits

CONFIG_PATH = Path(__file__).parents[1] / "config.json"


@dataclass(frozen=True)
class Config:
    daily_quota: int  # submissions per team and day
    daily_failed_quota: int  # failed submissions per team and day that do not count towards daily_quota
    quota_time_zone: str  # a day starts at midnight here (America/Los_Angeles: PST/PDT)
    max_upload_mib: int
    evaluation_timeout_seconds: float  # one submission's whole evaluation, all episodes
    limits: Limits  # the sandbox's, per episode: fuel, memory, wall-clock deadlines

    def as_dict(self):
        return asdict(self)


def load(path=CONFIG_PATH):
    """The Config in a JSON file; ValueError for a missing or unknown key or a bad value."""
    values = json.loads(path.read_text())
    expected = {f.name for f in fields(Config)}
    missing, unknown = sorted(expected - set(values)), sorted(set(values) - expected)
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    missing = sorted({f.name for f in fields(Limits)} - set(values["limits"]))
    if missing:
        raise ValueError(f"{path}: missing limits {missing}")  # unknown ones: Limits.from_dict
    config = Config(**{f.name: Limits.from_dict(values[f.name]) if f.name == "limits" else f.type(values[f.name])
                       for f in fields(Config)})
    ZoneInfo(config.quota_time_zone)  # an unknown time zone fails here rather than at the first upload
    return config
