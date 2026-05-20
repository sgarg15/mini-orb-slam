import glob
import os

import cv2


def frames_from_video(path, step=1, max_frames=None):
    """Yield grayscale frames from a video file, sampling every `step` frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video : {total} frames @ {fps:.1f} fps -> ~{total // step} frames to process")
    idx = 0
    yielded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        idx += 1
    cap.release()


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
    Check if the given path is a KITTI sequence directory (contains calib.txt) or if it contains a nested 'sequences/00' directory with calib.txt. Return the path to the directory containing calib.txt, or None if not found.
    """
    if os.path.exists(os.path.join(path, "calib.txt")):
        return path
    nested = os.path.join(path, "sequences", "00")
    if os.path.exists(os.path.join(nested, "calib.txt")):
        return nested
    return None


def kitti_image_dir(sequence_dir, camera_id):
    """
    Given a KITTI sequence directory (containing calib.txt), return the path to the image directory for the specified camera ID (e.g., '00' for camera 0). If the image directory does not exist, return None.
    """
    seq_dir = kitti_sequence_dir(sequence_dir)
    if seq_dir is None:
        return None
    image_dir = os.path.join(seq_dir, f"image_{camera_id}")
    if os.path.isdir(image_dir):
        return image_dir
    return None


def make_source(src, step, kitti_camera=0, max_frames=None):
    """Return a frame generator for either a video file or an image directory."""
    if os.path.isfile(src):
        return frames_from_video(src, step=step, max_frames=max_frames)
    if os.path.isdir(src):
        image_dir = kitti_image_dir(src, kitti_camera)
        if image_dir is not None:
            print(f"KITTI : using {image_dir}")
            return frames_from_dir(image_dir, max_frames=max_frames)
        if step != 1:
            print("Note: --step is ignored for image directories (use file order instead)")
        return frames_from_dir(src, max_frames=max_frames)
    raise IOError(f"'{src}' is not a file or directory")
