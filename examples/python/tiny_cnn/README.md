# Learned baseline: PyTorch → ONNX → `cv2.dnn` inside the runtime

1. Train on episode directories (any mix of the published sets or your own `simulate` output):

   ```bash
   python examples/python-tinynn/train.py datasets/default-train --epochs 10
   ```

   writes `solution/net.onnx`, `solution/weights.npz`, `solution/meta.json` (and `net.pt`).

2. Bundle and test:

   ```bash
   python -m eecs116 bundle examples/python-tinynn/solution -o tinynn.wasm
   python -m eecs116 check tinynn.wasm
   python -m eecs116 evaluate tinynn.wasm datasets/default-val
   ```

The solution runs the ONNX graph with `cv2.dnn.readNetFromONNX` (bytes from the bundle) and falls back to the numpy
forward pass in `tinynn.py` (same weights, `.npz`) if the graph is rejected — useful when trying operators OpenCV's
importer does not support. Export with `opset_version=13, dynamo=False`; keep the inputs static-shaped.

The model itself (frame pair at 64 × 48 + 9 preintegrated IMU values → relative 6-DoF motion, chained per frame) is
only a demonstration of the tooling: it drifts like a visual odometry without any correction. Ideas that stay inside
the same pipeline: predict in the gravity-aligned frame, feed longer IMU windows, add a learned scale/gravity
initialiser for the hover window, distil a classical tracker's inliers.
