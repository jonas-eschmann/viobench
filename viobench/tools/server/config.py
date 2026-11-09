"""Validated upload and evaluation settings; config.json seeds a new server and supplies local defaults."""
import json
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..runner.sandbox import Limits

CONFIG_PATH = Path(__file__).parents[1] / "config.json"
MAX_INTEGER = 2 ** 53 - 1  # settings round-trip through JavaScript numbers


def _object(values, cls):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    expected = {field.name for field in fields(cls)}
    missing, unknown = expected - values.keys(), values.keys() - expected
    if missing or unknown:
        raise ValueError(f"{cls.__name__}: missing keys {sorted(missing)}, unknown keys {sorted(map(str, unknown))}")


def _number(value, name, kind, minimum=0, inclusive=False):
    valid_type = type(value) is int if kind is int else type(value) in (int, float)
    try:
        valid = (valid_type and isfinite(value) and (value >= minimum if inclusive else value > minimum)
                 and (kind is not int or value <= MAX_INTEGER))
    except OverflowError:
        valid = False
    if not valid:
        bound = "at least" if inclusive else "greater than"
        maximum = f" and at most {MAX_INTEGER}" if kind is int else ""
        raise ValueError(f"{name} must be a finite {kind.__name__} {bound} {minimum}{maximum}")
    return kind(value)


@dataclass(frozen=True)
class UploadConfig:
    daily_quota: int
    quota_time_zone: str
    max_upload_mib: int

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        _object(values, cls)
        zone = values["quota_time_zone"]
        if not isinstance(zone, str) or not zone:
            raise ValueError("quota_time_zone must be a time zone name")
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown quota_time_zone: {zone}") from error
        return cls(daily_quota=_number(values["daily_quota"], "daily_quota", int, inclusive=True),
                   quota_time_zone=zone, max_upload_mib=_number(values["max_upload_mib"], "max_upload_mib", int))


@dataclass(frozen=True)
class EvaluationConfig:
    episode_timeout_seconds: float
    limits: Limits

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        _object(values, cls)
        _object(values["limits"], Limits)
        limits = Limits(**{field.name: _number(values["limits"][field.name], field.name, field.type)
                           for field in fields(Limits)})
        return cls(episode_timeout_seconds=_number(values["episode_timeout_seconds"], "episode_timeout_seconds", float),
                   limits=limits)


@dataclass(frozen=True)
class Config:
    uploads: UploadConfig
    evaluation: EvaluationConfig

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        _object(values, cls)
        return cls(uploads=UploadConfig.from_dict(values["uploads"]),
                   evaluation=EvaluationConfig.from_dict(values["evaluation"]))


def load(path=CONFIG_PATH):
    return Config.from_dict(json.loads(Path(path).read_text()))
