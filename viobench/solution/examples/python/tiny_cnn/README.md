# tiny_cnn: PyTorch → ONNX → `cv2.dnn`

```bash
python viobench/solution/examples/python/tiny_cnn/train.py data/train --epochs 10     # or --generate N: train on fresh simulated episodes
viobench bundle viobench/solution/examples/python/tiny_cnn/solution -o tinynn.wasm
viobench check tinynn.wasm && viobench evaluate tinynn.wasm data/val
```

Training writes `solution/net.onnx`, `solution/weights.npz` and `solution/meta.json`. The solution runs the ONNX graph
with `cv2.dnn` and falls back to the numpy forward pass in `tinynn.py` if OpenCV rejects the graph. Export with
`opset_version=13, dynamo=False` and static input shapes.

The model (a frame pair at 64 × 48 plus 9 preintegrated IMU values → the relative 6-DoF motion) demonstrates the
tooling; chained per frame, it drifts.
