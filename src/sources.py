import glob
import os

import cv2


def image_paths_from_dir(data_dir):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return sorted(paths)


def frames_from_dir(data_dir, max_frames=None):
    """Yield grayscale frames from sorted images in a directory."""
    paths = image_paths_from_dir(data_dir)
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise IOError(f"No images found in '{data_dir}'")
    print(f"Images: {len(paths)} files in '{data_dir}'")
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            yield img


def kitti_sequence_dir(path):
    """
    Return the sequence directory (the one containing calib.txt) for a KITTI
    path, or None if the path is not a recognised KITTI layout.
    """
    if os.path.exists(os.path.join(path, "calib.txt")):
        return path
    nested = os.path.join(path, "sequences", "00")
    if os.path.exists(os.path.join(nested, "calib.txt")):
        return nested
    return None


def kitti_image_dir(sequence_dir, camera_id):
    """
    Return the image directory for the given camera inside a KITTI sequence
    directory, or None if the directory does not exist.
    """
    seq_dir = kitti_sequence_dir(sequence_dir)
    if seq_dir is None:
        return None
    image_dir = os.path.join(seq_dir, f"image_{camera_id}")
    if os.path.isdir(image_dir):
        return image_dir
    return None
