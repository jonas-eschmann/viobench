"""numpy forward pass of train.py's TinyNet from weights.npz: the fallback when cv2.dnn is unavailable, and a
transparent reference for what the ONNX graph computes (conv via im2col + matmul)."""
import numpy as np


def conv2d(x, weight, bias, stride, padding):
    """x (C, H, W), weight (O, C, k, k) -> (O, H', W')."""
    channels, height, width = x.shape
    out_channels, _, k, _ = weight.shape
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    out_height = (height + 2 * padding - k) // stride + 1
    out_width = (width + 2 * padding - k) // stride + 1
    columns = np.empty((channels * k * k, out_height * out_width), dtype=np.float32)
    row = 0
    for c in range(channels):
        for i in range(k):
            for j in range(k):
                patch = padded[c, i:i + stride * out_height:stride, j:j + stride * out_width:stride]
                columns[row] = patch.reshape(-1)
                row += 1
    result = weight.reshape(out_channels, -1).astype(np.float32) @ columns + bias[:, None].astype(np.float32)
    return result.reshape(out_channels, out_height, out_width)


def relu(x):
    return np.maximum(x, 0.0)


class TinyNet:
    def __init__(self, weights):
        self.w = {key: np.asarray(value, dtype=np.float32) for key, value in weights.items()}

    def __call__(self, images, imu):
        """images (2, H, W) float32, imu (9,) normalised -> (6,)"""
        x = relu(conv2d(images, self.w["features.0.weight"], self.w["features.0.bias"], 2, 2))
        x = relu(conv2d(x, self.w["features.2.weight"], self.w["features.2.bias"], 2, 1))
        x = relu(conv2d(x, self.w["features.4.weight"], self.w["features.4.bias"], 2, 1))
        features = x.reshape(x.shape[0], -1).mean(axis=1)
        h = np.concatenate([features, imu.astype(np.float32)])
        h = relu(self.w["head.0.weight"] @ h + self.w["head.0.bias"])
        return self.w["head.2.weight"] @ h + self.w["head.2.bias"]
