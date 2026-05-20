# mini-orb-slam

ORB feature-based monocular visual odometry + sparse mapping inspired by ORB-SLAM.

The pipeline initializes a 3D map from two frames using the Essential Matrix, then tracks subsequent frames with PnP against that map. Keyframes are inserted when the camera has moved far enough, and the map is expanded by triangulating new points between each new keyframe and its predecessor.

---

## Project structure

```
mini-orb-slam/
├── data/
│   └── (KITTI sequence data goes here)
└── src/
    ├── main.py           # Pipeline orchestration
    ├── camera.py         # KITTI calibration loading and undistortion
    ├── features.py       # ORB detection and matching
    ├── geometry.py       # Pose estimation, triangulation, reprojection error
    ├── slam.py           # Map init, PnP tracking, keyframe selection, map expansion
    ├── sources.py        # KITTI image directory helpers
    └── visualization.py  # Match display, 3D point cloud, live dashboard
```

---

## Pipeline

### 1. Feature detection — `features.py`

ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors are extracted from each frame.

Matches are filtered with Lowe's ratio test (`ratio=0.75`): a match is kept only if the best match is significantly closer than the second-best.

### 2. Camera intrinsics — `camera.py`

The camera matrix **K** and distortion coefficients are loaded from the KITTI sequence `calib.txt` file. Each frame is undistorted before processing.

### 3. Map initialization — `slam.py`

The first frame with sufficient parallax is found by scanning the sequence. The Essential Matrix is estimated with RANSAC, and `recoverPose` decomposes it into **R** and **t**. Inlier correspondences are triangulated to create the initial 3D point cloud.

### 4. PnP tracking — `slam.py`

Each subsequent frame is tracked against the current map with PnP+RANSAC. If PnP fails or produces an implausibly large pose jump, the pipeline optionally falls back to frame-to-frame Essential Matrix VO to bridge the gap.

### 5. Keyframe selection and map expansion — `slam.py`

A new keyframe is inserted when median parallax, camera-center translation, and inter-keyframe frame count all exceed configurable thresholds. New 3D points are triangulated between the last keyframe and the new one and merged into the map.

### 6. Visualization — `visualization.py`

- **Inlier match view**: draws Essential Matrix RANSAC inliers from the initialization pair.
- **Trajectory dashboard**: matplotlib plot of the top-down trajectory, 3D point cloud, keyframe markers, and a strip of keyframe thumbnails.
- **Live dashboard** (`--live`): the trajectory dashboard is updated in real time as frames are processed.

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

Expected KITTI layout:

```text
data/kitti/00/
  calib.txt
  image_0/
    000000.png
    000001.png
    ...
```

Run on a KITTI sequence:

```bash
cd src
python main.py ../data/kitti/00 --kitti-camera 0
```

With the KITTI odometry dataset:

```bash
cd src
python main.py ../data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 300
```

For a real-time live dashboard while tracking:

```bash
python src/main.py data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 300 --live
```

Use `--no-final-viz` to skip the blocking summary plot after a live run, or `--no-viz` for a terminal-only smoke test:

```bash
cd src
python main.py ../data/data_odometry_gray/dataset/sequences/00 --kitti-camera 0 --max-frames 100 --no-viz
```

For KITTI color images, use camera 2:

```bash
python main.py ../data/kitti/00 --kitti-camera 2
```

KITTI odometry images are rectified, so distortion coefficients are not needed.
Sequence 00 has 4541 frames; start with `--max-frames 300` while tuning.

### Tuning

Keyframe insertion thresholds:

```bash
python main.py ../data/kitti/00 --keyframe-min-parallax 20 --keyframe-min-translation 0.15 --keyframe-min-frames 2
```

If the trajectory looks unstable, tighten initialization requirements:

```bash
python main.py ../data/kitti/00 --init-min-parallax 20 --max-pose-jump 8
```
