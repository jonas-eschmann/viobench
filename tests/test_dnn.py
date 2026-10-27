"""ONNX inference inside runtime.wasm via cv2.dnn: a tiny PyTorch CNN exported to ONNX must match PyTorch."""
import io

import numpy as np
import pytest

from relpose import Limits, Sandbox, abi
from relpose.solution.runtime.python.bundler.bundle import bundle
from relpose.tools.abi.config import default_guest_config

torch = pytest.importorskip("torch")

CONFIG = default_guest_config()

SOLUTION = """
import numpy as np, cv2
import solution_bundle as bundle
class Solution:
    def __init__(self, config):
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(bundle.read_bytes("net.onnx"), np.uint8))
        self.imu_vector = np.load(bundle.open("imu.npy"))
    def step(self, t, frame, imu):
        image = frame.transpose(2, 0, 1).astype(np.float32)[None] / 255.0  # RGB HWC -> NCHW
        self.net.setInput(image, "image")
        self.net.setInput(self.imu_vector[None].astype(np.float32), "imu")
        out = self.net.forward()
        return [float(out[0, 0]), float(out[0, 1]), float(out[0, 2]), 0, 0, 0, 1]
"""


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Sequential(torch.nn.Linear(16 + 9, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))

    def forward(self, image, imu):
        x = torch.relu(self.conv1(image))
        x = torch.relu(self.bn(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.head(torch.cat([x, imu], dim=1))


def test_onnx_in_wasm_matches_pytorch(engine, runtime_path, solution_dir, tmp_path):
    torch.manual_seed(0)
    net = TinyNet().eval()
    with torch.no_grad():  # give BatchNorm non-trivial statistics
        net.bn.running_mean.normal_()
        net.bn.running_var.uniform_(0.5, 2.0)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    imu_vector = rng.normal(size=9).astype(np.float32)
    image = torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32)[None] / 255.0)
    with torch.no_grad():
        expected = net(image, torch.from_numpy(imu_vector[None])).numpy()[0]

    onnx_buffer = io.BytesIO()
    torch.onnx.export(net, (image, torch.from_numpy(imu_vector[None])), onnx_buffer, input_names=["image", "imu"], output_names=["pose"], opset_version=13, dynamo=False)
    imu_buffer = io.BytesIO()
    np.save(imu_buffer, imu_vector)
    directory = solution_dir(SOLUTION, {"net.onnx": onnx_buffer.getvalue(), "imu.npy": imu_buffer.getvalue()})
    wasm = tmp_path / "dnn.wasm"
    bundle(directory, wasm, runtime=runtime_path)
    sandbox = Sandbox(wasm, engine, limits=Limits(), log_dir=tmp_path / "logs")
    sandbox.init(CONFIG)
    result = sandbox.step(0.05, frame, np.zeros((0, 7)))
    assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
    np.testing.assert_allclose(result.pose[:3], expected, rtol=1e-4, atol=1e-5)
    print(f"dnn step fuel {result.fuel:,} wall {result.wall_seconds * 1e3:.1f} ms")
