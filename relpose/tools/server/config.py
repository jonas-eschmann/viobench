"""The challenge's rules: `config.json` in the server directory, loaded once by the app and the worker (restart both
after editing it). Every key is optional and defaults to the values below; GET /config serves the effective whole.

    {"daily_quota": 1, "quota_time_zone": "America/Los_Angeles", "max_upload_mib": 128, "evaluation_timeout_seconds": 14400,
     "limits": {"step_fuel": 20000000000, "memory_bytes": 2147483648}}

`limits` takes any of the fields of `relpose.tools.runner.sandbox.Limits`. Deployment details stay in the environment:
RELPOSE_SERVER_DIR (where this file lives) and RELPOSE_DATASETS (the worker's evaluation sets).
"""
import json
from dataclasses import asdict, dataclass, field, fields
from zoneinfo import ZoneInfo

from ..runner.sandbox import Limits

CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class Config:
    daily_quota: int = 1  # submissions per team and day
    quota_time_zone: str = "America/Los_Angeles"  # a day starts at midnight here (PST/PDT)
    max_upload_mib: int = 128
    evaluation_timeout_seconds: float = 14400.0  # one submission's whole evaluation, all episodes
    limits: Limits = field(default_factory=Limits)  # the sandbox's, per episode: fuel, memory, wall-clock deadlines

    def as_dict(self):
        return asdict(self)


def load(path):
    """The Config in a JSON file, or the defaults when there is none; ValueError for an unknown key or a bad value."""
    values = json.loads(path.read_text()) if path.is_file() else {}
    defaults = Config()
    unknown = sorted(set(values) - {f.name for f in fields(Config)})
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    config = Config(**{key: Limits.from_dict(value) if key == "limits" else type(getattr(defaults, key))(value)
                       for key, value in values.items()})
    ZoneInfo(config.quota_time_zone)  # an unknown time zone fails here rather than at the first upload
    return config
