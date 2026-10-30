"""Camera playback with stacked trajectory minimaps in a compact sidebar, encoded one frame at a time."""
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .. import abi
from .metrics import hold_last_state, valid_states

# OpenCV drawing colours are BGR; episode images are RGB.
TRUTH = (105, 225, 125)
ESTIMATE = (75, 175, 255)
TEXT = (235, 240, 245)
BACKGROUND = (23, 29, 37)
MAP_WIDTH, MAP_HEIGHT = 220, 208
HEADER_HEIGHT = 64


def _text(image, text, position, colour=TEXT, scale=0.45):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


class _Minimap:
    def __init__(self, truth, estimates, axes):
        points = truth[:, axes] if estimates is None else np.concatenate([truth[:, axes], estimates[:, axes]])
        lower, upper = points.min(axis=0), points.max(axis=0)
        centre = (lower + upper) / 2
        span = max(float(np.max(upper - lower)) * 1.15, 1.0)

        def project(trajectory):
            xy = (trajectory[:, axes] - centre) * (164 / span)
            return np.rint(xy * [1, -1] + [110, 110]).astype(np.int32)

        self.truth = project(truth)
        self.estimates = None if estimates is None else project(estimates)
        self.trail = np.full((MAP_HEIGHT, MAP_WIDTH, 3), BACKGROUND, dtype=np.uint8)
        for offset in range(0, 165, 41):
            cv2.line(self.trail, (28 + offset, 28), (28 + offset, 192), (60, 68, 80))
            cv2.line(self.trail, (28, 28 + offset), (192, 28 + offset), (60, 68, 80))
        horizontal, vertical = ("XYZ"[axis] for axis in axes)
        _text(self.trail, f"{horizontal}-{vertical} projection", (8, 18), scale=0.5)
        _text(self.trail, f"+{vertical}", (4, 39), scale=0.4)
        _text(self.trail, f"+{horizontal}", (195, 198), scale=0.4)
        _text(self.trail, f"{span:.2g} m across", (8, 203), scale=0.4)

    def draw(self, frame, index, left, top):
        for points, colour in ((self.truth, TRUTH), (self.estimates, ESTIMATE)):
            if index and points is not None:
                cv2.line(self.trail, tuple(points[index - 1]), tuple(points[index]), colour, 2, cv2.LINE_AA)
        panel = self.trail.copy()
        # A ring and a filled dot remain distinguishable when the two positions coincide.
        cv2.circle(panel, tuple(self.truth[index]), 6, TRUTH, 2, cv2.LINE_AA)
        if self.estimates is not None:
            cv2.circle(panel, tuple(self.estimates[index]), 3, ESTIMATE, -1, cv2.LINE_AA)
        frame[top:top + MAP_HEIGHT, left:left + MAP_WIDTH] = panel


def write_video(episode, estimates, path):
    """Stream camera frames and camera-0 trajectory minimaps at the episode's frame rate; estimates=None shows truth only."""
    path = Path(path)
    if episode.num_frames == 0:
        raise ValueError("video needs at least one frame")
    if estimates is not None:
        estimates = np.asarray(estimates, dtype=float)
        if estimates.shape != (episode.num_frames, abi.STATE_VALUES):
            raise ValueError("video needs one estimated state per frame")
        usable = valid_states(estimates)
        estimates = hold_last_state(estimates)[:, :3]
    truth = episode.ground_truth_relative()[:, :3]
    maps = [_Minimap(truth, estimates, axes) for axes in ((0, 2), (1, 2))]
    fps = float(episode.config["episode"]["frame_rate_hz"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish the container before replacing an existing video; clean up partial files on failure.
    with TemporaryDirectory(prefix=".relpose-video-", dir=path.parent) as temporary:
        staged = Path(temporary) / "trajectory.mp4"
        writer = cv2.VideoWriter(str(staged), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640 + MAP_WIDTH, 480))
        try:
            if not writer.isOpened():
                raise RuntimeError(f"cannot encode {path}: OpenCV needs an MP4 (mp4v) encoder")
            for index, rgb in enumerate(episode.frames):
                if rgb.ndim == 2 or rgb.shape[2] == 1:
                    frame = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                else:
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if frame.shape[:2] != (480, 640):
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                canvas = np.full((480, 640 + MAP_WIDTH, 3), BACKGROUND, dtype=np.uint8)
                canvas[:, :640] = frame
                elapsed = episode.frame_times[index] - episode.frame_times[0]
                _text(canvas, f"t={elapsed:.2f}s  {index + 1}/{episode.num_frames}", (648, 17))
                _text(canvas, "Ground truth", (648, 55), TRUTH, scale=0.4)
                if estimates is not None:
                    error = np.linalg.norm(estimates[index] - truth[index])
                    label = "Position error" if usable[index] else "Error (pose held)"
                    _text(canvas, f"{label}: {error:.2f} m", (648, 36), scale=0.4)
                    _text(canvas, "Estimate", (758, 55), ESTIMATE, scale=0.4)
                maps[0].draw(canvas, index, 640, HEADER_HEIGHT)
                maps[1].draw(canvas, index, 640, HEADER_HEIGHT + MAP_HEIGHT)
                writer.write(canvas)
        finally:
            writer.release()
        staged.replace(path)
    return path
