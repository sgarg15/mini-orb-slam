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

### KITTI odometry

Use a KITTI odometry sequence folder as the input source. The code detects
`calib.txt`, loads the correct projection matrix, and reads the matching image
folder automatically.

Expected layout:

```text
data/kitti/00/
  calib.txt
  image_0/
    000000.png
    000001.png
    ...
```

Run grayscale camera 0:

```bash
python main.py ../data/kitti/00 --kitti-camera 0
```

With the KITTI folders in this repo, run sequence 00 like this:

```bash
cd src
python main.py ../data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 300
```

The final matplotlib dashboard shows the 3D/top-down trajectory, orange
keyframe markers, and a strip of recent keyframe images. Increase or decrease
the image strip with:

```bash
python main.py ../data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 300 --keyframe-thumbs 12
```

For a realtime KITTI demo, stream the image sequence directly and update the
matplotlib dashboard while tracking:

```bash
python src\main.py data\data_odometry_gray\dataset\sequences\00 --kitti-camera 0 --max-frames 300 --live
```

Use `--no-final-viz` if you only want the live dashboard and do not want the
final blocking matplotlib summary after processing finishes.

For a quick terminal-only smoke test:

```bash
cd src
python main.py ../data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 100 --no-viz
```

The source should be the `data_odometry_gray/.../sequences/00` folder because it
contains `image_0/`. The `data_odometry_calib/.../sequences/00` folder contains
calibration only and is useful for reference, but it does not contain frames.

For KITTI color images, use camera 2:

```bash
python main.py ../data/kitti/00 --kitti-camera 2
```

KITTI odometry images are rectified, so distortion coefficients are not needed.
Sequence 00 has 4541 frames, so start with `--max-frames 300` while tuning.

With real camera calibration:

```bash
python main.py ../data/vid1.mp4 --calib ../data/calib.json
```

or explicit intrinsics:

```bash
python main.py ../data/vid1.mp4 --fx 718.856 --fy 718.856 --cx 607.1928 --cy 185.2157
```

Calibration files may be `.npz`, OpenCV `.yaml/.yml/.xml`, `.json`, `.txt`, or `.csv`.
They should contain `K` / `camera_matrix` and may include `dist` / `dist_coeffs`.

Keyframe insertion is intentionally lightweight for this mini version. Tune it with:

```bash
python main.py ../data/vid1.mp4 --keyframe-min-parallax 20 --keyframe-min-translation 0.15 --keyframe-min-frames 2
```

If the trajectory looks unstable, make initialization more conservative so the
first map is not built from a near-pure-rotation / tiny-baseline pair:

```bash
python main.py ../data/vid1.mp4 --init-min-parallax 20 --max-pose-jump 8
```
