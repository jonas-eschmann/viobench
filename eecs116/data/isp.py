"""Camera / image-signal-processor model applied to rendered (linear-radiance) frames.

HyperDrone outputs clean sRGB frames at the IMU rate; this module turns a stack of consecutive renders into one
sensor frame: box-shutter integration over the exposure (motion blur), exposure scaling (low light /
oversaturation), photon shot noise, read noise, clipping, sRGB encoding and 8-bit quantisation. The same code serves
the held-out variants and students' training augmentation (eecs116.data.variants has the public parameter ranges).
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ISPParams:
    exposure_scale: float = 1.0  # multiplies linear radiance; 1 = nominal exposure of the rendered scene
    full_well: float = 20000.0  # electrons at saturation; shot noise std = sqrt(signal * full_well) / full_well; inf disables
    read_noise: float = 0.002  # std of additive Gaussian noise in linear units (1 = saturation)
    digital_gain: float = 1.0  # applied after the noise (an under-exposed, gained-up image is the low-light case)
    black_level: float = 0.0  # added before sRGB encoding (sensor pedestal), linear units

    def as_dict(self):
        return asdict(self)


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    high = ((x + np.float32(0.055)) * np.float32(1 / 1.055)) ** np.float32(2.4)
    return np.where(x <= 0.04045, x * np.float32(1 / 12.92), high)


def linear_to_srgb(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0.0), 1 / 2.4) - 0.055)


def luminance(rgb_linear):
    """Rec. 709 luma from linear RGB (..., 3) -> (...)."""
    return rgb_linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def integrate_exposure(frames_srgb):
    """Average a stack (K, H, W[, 3]) of sRGB renders in linear space: a box shutter spanning the K sub-frames."""
    frames_srgb = np.asarray(frames_srgb, dtype=np.float32)
    if frames_srgb.ndim == 3 or (frames_srgb.ndim == 4 and frames_srgb.shape[-1] in (1, 3)):
        pass
    else:
        raise ValueError(f"expected (K, H, W) or (K, H, W, C), got {frames_srgb.shape}")
    return srgb_to_linear(frames_srgb).mean(axis=0)


def apply(linear, params, rng, grayscale=True):
    """linear: (H, W[, 3]) linear radiance in [0, 1]-ish units -> uint8 image (H, W) or (H, W, 3)."""
    linear = np.asarray(linear, dtype=np.float32)
    if grayscale and linear.ndim == 3:
        linear = luminance(linear)
    signal = linear * np.float32(params.exposure_scale)
    if np.isfinite(params.full_well) and params.full_well > 0:
        # photon shot noise: Poisson(signal * full_well) / full_well, Gaussian approximation (std = sqrt(N) electrons);
        # exact Poisson sampling costs 4x more and only differs below ~20 electrons per pixel
        std = np.sqrt(np.clip(signal, 0, None) / np.float32(params.full_well), dtype=np.float32)
        signal = signal + rng.standard_normal(signal.shape, dtype=np.float32) * std
    if params.read_noise > 0:
        signal = signal + rng.standard_normal(signal.shape, dtype=np.float32) * np.float32(params.read_noise)
    signal = signal * np.float32(params.digital_gain) + np.float32(params.black_level)
    encoded = linear_to_srgb(np.clip(signal, 0.0, 1.0))
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def process(frames_srgb_stack, params, rng, grayscale=True):
    """Convenience: integrate the exposure stack and apply the sensor model."""
    return apply(integrate_exposure(frames_srgb_stack), params, rng, grayscale=grayscale)
