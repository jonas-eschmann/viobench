"""Train a tiny frame-pair + IMU network to predict the relative camera motion between consecutive frames.

    python viobench/solution/examples/python/tiny_cnn/train.py <episode dirs or dataset roots>... --epochs 10 --out viobench/solution/examples/python/tiny_cnn/solution

Writes net.pt (PyTorch), net.onnx (opset 13, loadable by cv2.dnn inside the wasm runtime), weights.npz (for the
numpy fallback) and meta.json (normalisation constants) into the solution directory, which is then bundled with
    viobench bundle viobench/solution/examples/python/tiny_cnn/solution -o tinynn.wasm
The model is a demonstration of the tooling (PyTorch -> ONNX -> cv2.dnn in wasm), not a strong estimator.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "solution"))
from preintegrate import exp_so3, log_so3, preintegrate  # noqa: E402  (shared with the solution)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from viobench.tools.datasets.storage import list_episodes, read_episode  # noqa: E402
from viobench.tools.geometry import invert, poses_to_matrices  # noqa: E402

INPUT_WIDTH, INPUT_HEIGHT = 64, 48


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, 5, stride=2, padding=2), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(torch.nn.Linear(64 + 9, 64), torch.nn.ReLU(), torch.nn.Linear(64, 6))

    def forward(self, images, imu):
        return self.head(torch.cat([self.features(images), imu], dim=1))


def samples_from_episode(episode):
    """(images (N-1, 2, H, W) float, imu (N-1, 9), target (N-1, 6)) for consecutive frame pairs."""
    config, frames = episode.adapt()  # what the solution receives (RGB); it converts to gray exactly like this
    small = np.stack([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                      for frame in frames]).astype(np.float32) / 255.0
    T = poses_to_matrices(episode.ground_truth_relative())
    T_cam_imu = np.asarray(config["T_cam_imu"])
    images, imus, targets = [], [], []
    for k in range(1, episode.num_frames):
        window = episode.imu_window(k)
        if len(window) == 0:
            continue
        relative = invert(T[k - 1]) @ T[k]  # T_c(k-1)_ck
        integrated = preintegrate(window, T_cam_imu)
        # the gyroscope already gives the rotation well: the network predicts the translation and the small
        # rotation residual R_res = exp(imu rotation)^T R_gt (gyro bias and integration error)
        residual = log_so3(exp_so3(integrated[:3].astype(np.float64)).T @ relative[:3, :3])
        images.append(np.stack([small[k - 1], small[k]]))
        imus.append(integrated)
        targets.append(np.concatenate([relative[:3, 3], residual]))
    return np.asarray(images), np.asarray(imus, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="episode directories or dataset roots")
    parser.add_argument("--generate", type=int, default=0, metavar="N",
                        help="also train on N fresh HyperDrone episodes of the default scene (GPU)")
    parser.add_argument("--generate-seed", type=int, default=10_000)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "solution"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-episodes", type=int)
    arguments = parser.parse_args()

    episodes = [path for root in arguments.datasets for path in list_episodes(root)]
    if arguments.max_episodes:
        episodes = episodes[:arguments.max_episodes]
    if not episodes and not arguments.generate:
        parser.error("give episode directories and/or --generate N")
    images, imus, targets = [], [], []

    def add(episode):
        i, m, t = samples_from_episode(episode)
        images.append(i)
        imus.append(m)
        targets.append(t)
        print(f"{episode.name}: {len(i)} pairs")

    for path in episodes:
        add(read_episode(path))
    if arguments.generate:  # "infinite" data: straight from the simulator
        from viobench.tools.datasets.generate.simulation import (
            DEFAULT_SCENE_NAMES,
            SimConfig,
            scene_references,
            simulate,
        )

        for episode in simulate(SimConfig(), scene_references(DEFAULT_SCENE_NAMES), arguments.generate, seed=arguments.generate_seed):
            add(episode)
    images, imus, targets = np.concatenate(images), np.concatenate(imus), np.concatenate(targets)
    imu_mean, imu_std = imus.mean(0), imus.std(0) + 1e-6
    target_std = targets.std(0) + 1e-6
    meta = {"input_width": INPUT_WIDTH, "input_height": INPUT_HEIGHT, "imu_mean": imu_mean.tolist(), "imu_std": imu_std.tolist(), "target_std": target_std.tolist()}

    torch.manual_seed(0)
    net = TinyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=arguments.lr)
    x_images = torch.from_numpy(images)
    x_imu = torch.from_numpy((imus - imu_mean) / imu_std)
    y = torch.from_numpy(targets / target_std)
    for epoch in range(arguments.epochs):
        permutation = torch.randperm(len(y))
        total = 0.0
        for start in range(0, len(y), arguments.batch):
            index = permutation[start:start + arguments.batch]
            loss = torch.nn.functional.smooth_l1_loss(net(x_images[index], x_imu[index]), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        print(f"epoch {epoch + 1}: loss {total / len(y):.4f}")

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    net.eval()
    torch.save(net.state_dict(), out / "net.pt")
    example = (x_images[:1], x_imu[:1])
    torch.onnx.export(net, example, str(out / "net.onnx"), input_names=["images", "imu"], output_names=["motion"], opset_version=13, dynamo=False)
    weights = {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}
    np.savez(out / "weights.npz", **weights)  # uncompressed: the runtime has no zlib
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    with torch.no_grad():
        reference = net(*example).numpy()
    np.save(out / "reference_output.npy", reference)
    print(f"wrote {out}/net.onnx, weights.npz, meta.json ({len(y)} training pairs)")


if __name__ == "__main__":
    main()
