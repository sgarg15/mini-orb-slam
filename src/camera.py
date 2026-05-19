import json

import cv2
import numpy as np


def build_K(img_shape):
    h, w = img_shape[:2]
    focal = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    return np.array([[focal, 0.0, cx],
                     [0.0, focal, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def build_K_from_args(args, img_shape):
    values = (args.fx, args.fy, args.cx, args.cy)
    if any(v is not None for v in values):
        if not all(v is not None for v in values):
            raise ValueError("--fx, --fy, --cx, and --cy must be provided together")
        return np.array([[args.fx, 0.0, args.cx],
                         [0.0, args.fy, args.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    return build_K(img_shape)


def _read_text_matrix(path):
    if _looks_like_kitti_calib(path):
        return load_kitti_calibration(path)

    data = np.loadtxt(path, delimiter="," if path.lower().endswith(".csv") else None)
    data = np.asarray(data, dtype=np.float64)
    if data.shape == (3, 3):
        return data, None

    flat = data.ravel()
    if flat.size >= 4:
        fx, fy, cx, cy = flat[:4]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = flat[4:] if flat.size > 4 else None
        return K, dist

    raise ValueError(f"Could not parse calibration file: {path}")


def _looks_like_kitti_calib(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_lines = [next(f, "") for _ in range(4)]
    except (OSError, StopIteration):
        return False
    return any(line.startswith(("P0:", "P1:", "P2:", "P3:")) for line in first_lines)


def load_kitti_calibration(path, camera_id=0):
    """Load K from a KITTI odometry calib.txt projection matrix.

    KITTI images are already rectified, so distortion is None. For monocular
    use P0/P1 for grayscale cameras or P2/P3 for color cameras.
    """
    target = f"P{camera_id}:"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(target):
                continue
            values = np.fromstring(line[len(target):], sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"Invalid KITTI projection matrix in {path}: {target}")
            P = values.reshape(3, 4)
            return P[:, :3].copy(), None
    raise ValueError(f"KITTI calibration {path} does not contain {target}")


def load_calibration(path):
    """Load K and optional distortion coefficients from common calibration files."""
    lower = path.lower()

    if lower.endswith(".npz"):
        data = np.load(path)
        if "K" in data:
            K = data["K"]
        elif "camera_matrix" in data:
            K = data["camera_matrix"]
        else:
            raise ValueError("NPZ calibration must contain K or camera_matrix")

        dist = None
        for key in ("dist", "dist_coeffs", "distortion_coefficients"):
            if key in data:
                dist = data[key].ravel()
                break
        return np.asarray(K, dtype=np.float64), None if dist is None else np.asarray(dist, dtype=np.float64)

    if lower.endswith((".yaml", ".yml", ".xml")):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f"Could not open calibration file: {path}")

        K = None
        for key in ("K", "camera_matrix"):
            node = fs.getNode(key)
            if not node.empty():
                K = node.mat()
                break
        if K is None:
            fs.release()
            raise ValueError("Calibration file must contain K or camera_matrix")

        dist = None
        for key in ("dist", "dist_coeffs", "distortion_coefficients"):
            node = fs.getNode(key)
            if not node.empty():
                dist = node.mat().ravel()
                break

        fs.release()
        return np.asarray(K, dtype=np.float64), None if dist is None else np.asarray(dist, dtype=np.float64)

    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "K" in data:
            K = np.asarray(data["K"], dtype=np.float64)
        elif "camera_matrix" in data:
            K = np.asarray(data["camera_matrix"], dtype=np.float64)
        else:
            K = np.array([[data["fx"], 0.0, data["cx"]],
                          [0.0, data["fy"], data["cy"]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = data.get("dist", data.get("dist_coeffs"))
        return K, None if dist is None else np.asarray(dist, dtype=np.float64).ravel()

    return _read_text_matrix(path)


def undistort_frame(img, K, dist):
    if dist is None or len(dist) == 0:
        return img
    return cv2.undistort(img, K, dist)


def print_K(K, img_shape, source="estimated", dist=None):
    h, w = img_shape[:2]
    print("--- Camera Intrinsics ---")
    print(f"Source: {source}")
    print(f"Image size: {w}x{h}")
    print(f"fx, fy: {K[0, 0]:.1f}, {K[1, 1]:.1f} px")
    print(f"Principal point: ({K[0, 2]:.1f}, {K[1, 2]:.1f})")
    if dist is not None and len(dist) > 0:
        print(f"Distortion: {np.array2string(np.asarray(dist).ravel(), precision=4)}")
    if K[0, 0] < 0.3 * w or K[0, 0] > 3 * w or K[1, 1] < 0.3 * w or K[1, 1] > 3 * w:
        print("WARNING: focal length looks unreasonable, check K")
    print()
