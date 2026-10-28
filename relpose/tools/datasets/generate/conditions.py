"""Public parameter ranges of the dataset conditions (the held-out sets draw their exact values from these ranges).

A condition fixes how a rendered IMU-rate sub-frame stack becomes a camera frame (relpose.tools.datasets.generate.camera): exposure time
(number of sub-frames integrated), exposure scaling, noise. Students may use `sample(condition, rng)` as training
augmentation; the hidden evaluation seeds are not published.
"""
from dataclasses import dataclass

import numpy as np

from .camera import SensorParams


@dataclass(frozen=True)
class ConditionRange:
    name: str
    description: str
    exposure_seconds: tuple  # [min, max] of the box shutter; sub-frames = round(exposure * imu_rate), at least 1
    exposure_scale: tuple  # [min, max], log-uniform
    read_noise: tuple  # [min, max] linear units
    full_well: tuple  # [min, max] electrons, log-uniform
    digital_gain_targets_mean: bool  # low light: gain chosen so the image mean lands near the nominal one


CONDITIONS = {
    "default": ConditionRange("default", "nominal exposure, short shutter, mild sensor noise",
                            exposure_seconds=(0.001, 0.005), exposure_scale=(0.8, 1.25), read_noise=(0.001, 0.004), full_well=(10000.0, 40000.0),
                            digital_gain_targets_mean=False),
    "blur": ConditionRange("blur", "long shutter: heavy motion blur, otherwise nominal",
                         exposure_seconds=(0.025, 0.05), exposure_scale=(0.8, 1.25), read_noise=(0.001, 0.004), full_well=(10000.0, 40000.0),
                         digital_gain_targets_mean=False),
    "lowlight": ConditionRange("lowlight", "under-exposed, gained up: strong shot and read noise, crushed shadows",
                             exposure_seconds=(0.005, 0.02), exposure_scale=(0.02, 0.1), read_noise=(0.005, 0.02), full_well=(500.0, 3000.0),
                             digital_gain_targets_mean=True),
    "oversaturated": ConditionRange("oversaturated", "over-exposed: large parts of the image clip to white",
                                  exposure_seconds=(0.001, 0.005), exposure_scale=(2.0, 5.0), read_noise=(0.001, 0.004), full_well=(10000.0, 40000.0),
                                  digital_gain_targets_mean=False),
}


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample(condition, rng, imu_rate_hz=200.0):
    """Draw (SensorParams, sub_frames) for one episode of the condition."""
    spec = CONDITIONS[condition] if isinstance(condition, str) else condition
    exposure = rng.uniform(*spec.exposure_seconds)
    sub_frames = max(1, int(round(exposure * imu_rate_hz)))
    scale = log_uniform(rng, *spec.exposure_scale)
    params = SensorParams(
        exposure_scale=scale,
        full_well=log_uniform(rng, *spec.full_well),
        read_noise=float(rng.uniform(*spec.read_noise)),
        digital_gain=(1.0 / scale) * float(rng.uniform(0.7, 1.0)) if spec.digital_gain_targets_mean else 1.0,
    )
    return params, sub_frames
