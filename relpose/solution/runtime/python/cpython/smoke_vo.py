import time
import numpy as np
import cv2

print("cv2", cv2.__version__, "numpy", np.__version__)
print(np.random.default_rng(0).multinomial(10, [0.2, 0.8]), np.random.RandomState(0).multinomial(10, [0.2, 0.8]))

rng = np.random.default_rng(0)
width, height, focal = 320, 240, 250.0
K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
points = np.column_stack([rng.uniform(-2, 2, 400), rng.uniform(-1.5, 1.5, 400), rng.uniform(4, 8, 400)])

def render(rotation_vector, translation):
    projected, _ = cv2.projectPoints(points, rotation_vector, translation, K, None)
    image = np.zeros((height, width), np.uint8)
    for x, y in projected.reshape(-1, 2):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (int(x), int(y)), 3, 255, -1)
    return cv2.GaussianBlur(image, (5, 5), 1.0)

true_rotation = np.array([0.0, 0.05, 0.0])
true_translation = np.array([-0.3, 0.0, 0.0])
first = render(np.zeros(3), np.zeros(3))
second = render(true_rotation, true_translation)

start = time.perf_counter()
orb = cv2.ORB_create(500)
keypoints, descriptors = orb.detectAndCompute(first, None)
corners = cv2.goodFeaturesToTrack(first, 300, 0.01, 5)
tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, corners, None)
good_first = corners[status.ravel() == 1].reshape(-1, 2)
good_second = tracked[status.ravel() == 1].reshape(-1, 2)
essential, inliers = cv2.findEssentialMat(good_first, good_second, K, cv2.RANSAC, 0.999, 1.0)
_, rotation, translation, _ = cv2.recoverPose(essential, good_first, good_second, K, mask=inliers)
elapsed = time.perf_counter() - start

print("orb keypoints", len(keypoints), "tracked", len(good_first), "inliers", int(inliers.sum()))
print("rotation vector", cv2.Rodrigues(rotation)[0].ravel().round(4), "true", true_rotation)
print("translation direction", translation.ravel().round(3), "true", (true_translation / np.linalg.norm(true_translation)).round(3))
print(f"pipeline {elapsed * 1e3:.1f} ms")
try:
    cv2.findEssentialMat(np.zeros((2, 2)), np.zeros((2, 2)), K)
except cv2.error as error:
    print("cv2.error raised and caught:", str(error).strip().splitlines()[-1][:80])
