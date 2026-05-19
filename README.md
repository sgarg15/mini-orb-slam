# mini-orb-slam

ORB feature-based monocular visual odometry + sparse mapping inspired by ORB-SLAM.

Given two images from a monocular camera, the pipeline detects and matches ORB features, estimates the relative camera pose via the Essential Matrix, triangulates a sparse 3D point cloud, and validates it with reprojection error.

---

## Project structure

```
mini-orb-slam/
├── data/
│   ├── img1.jpg
│   └── img2.jpg
└── src/
    ├── main.py           # Pipeline orchestration
    ├── camera.py         # Camera intrinsic matrix
    ├── features.py       # ORB detection and matching
    ├── geometry.py       # Pose estimation, triangulation, reprojection error
    └── visualization.py  # Match display and 3D point cloud plot
```

---

## Pipeline

### 1. Feature detection — `features.py`

ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors are extracted from both images.

```python
kp, des = detect(img)
```

Matches are filtered with Lowe's ratio test (`ratio=0.75`): a match is kept only if the best match is significantly closer than the second-best, which removes ambiguous correspondences.

```python
raw_matches, good_matches = match(des1, des2)
```

### 2. Camera intrinsics — `camera.py`

A pinhole camera matrix **K** is constructed from the image dimensions, assuming the principal point is at the image centre and the focal length is `0.9 × width`.

```
K = [[f,  0, cx],
     [0,  f, cy],
     [0,  0,  1]]
```

A sanity check warns if the focal length falls outside the range `[0.3w, 3w]`, which would indicate a bad calibration.

### 3. Pose estimation — `geometry.py`

The Essential Matrix **E** is estimated with RANSAC using the matched point correspondences and **K**:

```
E, inlier_mask = findEssentialMat(pts1, pts2, K)
```

RANSAC suppresses outlier matches that survived the ratio test. `recoverPose` then decomposes **E** into a rotation matrix **R** and unit translation vector **t**, resolving the four-way chirality ambiguity by checking that triangulated points lie in front of both cameras.

### 4. Triangulation — `geometry.py`

Projection matrices for both cameras are formed:

```
P1 = K @ [I | 0]      # camera 1 at world origin
P2 = K @ [R | t]      # camera 2 pose
```

`cv2.triangulatePoints` solves the DLT system for each inlier correspondence to recover homogeneous 3D coordinates. Points with negative depth (behind either camera) are discarded.

### 5. Reprojection error — `geometry.py`

Each triangulated point is projected back into both image planes and compared against the original 2D observations:

```
error = mean(||pts_observed - pts_projected||)
```

| Error range | Interpretation |
|---|---|
| < 3 px | Good — pipeline is healthy |
| 3–8 px | Acceptable — check K or baseline |
| > 8 px | Poor — likely bad K or degenerate scene |

### 6. Visualization — `visualization.py`

- **Inlier match view**: draws only the Essential Matrix RANSAC inliers, confirming the geometric consistency of correspondences.
- **3D point cloud**: matplotlib scatter coloured by depth, with camera positions overlaid (blue = cam 1 at origin, red = cam 2).

---

## Dependencies

- Python 3.8+
- opencv-contrib-python
- numpy
- matplotlib

Install:

```bash
pip install opencv-contrib-python numpy matplotlib
```

## Running

```bash
cd src
python main.py
```
