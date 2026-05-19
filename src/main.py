import os
import sys
import glob
import argparse

import cv2
import numpy as np

from camera import build_K, print_K
from features import detect, match, matched_points, match_to_map
from geometry import estimate_pose, triangulate, triangulate_poses, solve_pnp, reprojection_error
from visualization import show_inlier_matches, plot_3d, plot_trajectory

# ── Frame sources ─────────────────────────────────────────────────────────────

def frames_from_video(path, step=1):
    """Yield grayscale frames from a video file, sampling every `step` frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video : {total} frames @ {fps:.1f} fps  →  ~{total // step} frames to process")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()


def frames_from_dir(data_dir):
    """Yield grayscale frames from sorted images in a directory."""
    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    paths = sorted(paths)
    if not paths:
        raise IOError(f"No images found in '{data_dir}'")
    print(f"Images: {len(paths)} files in '{data_dir}'")
    for p in paths:
        yield cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def make_source(src, step):
    """Return a frame generator for either a video file or an image directory."""
    if os.path.isfile(src):
        return frames_from_video(src, step=step)
    if os.path.isdir(src):
        if step != 1:
            print(f"Note: --step is ignored for image directories (use file order instead)")
        return frames_from_dir(src)
    raise IOError(f"'{src}' is not a file or directory")

# ── Pipeline helpers ──────────────────────────────────────────────────────────

MIN_INIT_POINTS  = 10  # minimum triangulated map points to accept initialisation
MIN_INIT_INLIERS = 8   # minimum Essential-Matrix inliers


def init_map(img0, img1, K):
    """Bootstrap the 3D map from two frames using Essential Matrix + triangulation.

    Returns (result_tuple, None) on success or (None, reason_string) on failure,
    so callers can surface diagnostic information.

    Success tuple:
        map3d, map_des, R01, t01, kp0, kp1, des1, good01, inlier_mask
    """
    kp0, des0 = detect(img0)
    kp1, des1 = detect(img1)

    # Use a more permissive ratio during init to maximise match recall
    _, good01 = match(des0, des1, ratio=0.80)
    if len(good01) < MIN_INIT_INLIERS:
        return None, f"only {len(good01)} good matches (need {MIN_INIT_INLIERS})"

    pts0, pts1 = matched_points(kp0, kp1, good01)

    try:
        R01, t01, inlier_mask = estimate_pose(pts0, pts1, K)
    except cv2.error as exc:
        return None, f"Essential Matrix failed: {exc}"

    n_inliers = int(inlier_mask.sum())
    if n_inliers < MIN_INIT_INLIERS:
        return None, f"only {n_inliers} RANSAC inliers (need {MIN_INIT_INLIERS})"

    # Thread descriptors from frame 1 through the inlier + validity filters
    inlier_idx = np.where(inlier_mask)[0]
    train_idx  = np.array([good01[i].trainIdx for i in inlier_idx])
    des1_in    = des1[train_idx]

    pts0_in = pts0[inlier_mask]
    pts1_in = pts1[inlier_mask]
    map3d, pts0_tri, pts1_tri, valid = triangulate(K, R01, t01, pts0_in, pts1_in)

    if len(map3d) < MIN_INIT_POINTS:
        return None, (f"only {len(map3d)} triangulated points (need {MIN_INIT_POINTS}); "
                      f"{n_inliers} inliers → likely pure rotation or insufficient parallax")

    map_des = des1_in[valid]

    e1, e2 = reprojection_error(map3d, pts0_tri, pts1_tri, K, R01, t01)
    print(f"Init  : {len(map3d):5d} map points  |  reprojection error  cam0={e1:.2f}px  cam1={e2:.2f}px")

    return (map3d, map_des, R01, t01, kp0, kp1, des1, good01, inlier_mask), None


def track_frame(img, K, map3d, map_des):
    """Estimate pose of a new frame by matching features to the 3D map (PnP).

    Returns (R, t, n_inliers, kp, des).  R/t are None on tracking failure.
    """
    kp, des = detect(img)
    if des is None or len(des) == 0:
        return None, None, 0, kp, des

    frame_idx, map_idx = match_to_map(des, map_des)
    if len(frame_idx) < 6:
        return None, None, 0, kp, des

    pts2d = np.array([kp[j].pt for j in frame_idx], dtype=np.float64)
    pts3d = map3d[map_idx]

    R, t, inliers = solve_pnp(pts3d, pts2d, K)
    n_inliers = len(inliers) if inliers is not None else 0
    return R, t, n_inliers, kp, des


def expand_map(map3d, map_des, K, prev_kp, prev_des, R_prev, t_prev, curr_kp, curr_des, R_curr, t_curr):
    """Triangulate new 3D points from consecutive frames and append them to the map."""
    _, good = match(prev_des, curr_des)
    if len(good) < 8:
        return map3d, map_des

    pts_prev, pts_curr = matched_points(prev_kp, curr_kp, good)
    new_pts3d, valid = triangulate_poses(K, R_prev, t_prev, pts_prev, R_curr, t_curr, pts_curr)

    if len(new_pts3d) == 0:
        return map3d, map_des

    train_idx = np.array([good[j].trainIdx for j in range(len(good)) if valid[j]])
    new_des   = curr_des[train_idx]

    map3d   = np.vstack([map3d, new_pts3d])
    map_des = np.vstack([map_des, new_des])
    return map3d, map_des

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Mini ORB-SLAM: track camera over a video or image sequence')
    parser.add_argument('source', nargs='?', default='../data',
                        help='Video file (mp4/avi/…) or directory of images (default: ../data)')
    parser.add_argument('--step', type=int, default=1,
                        help='Process every Nth video frame (default: 1 = every frame)')
    args = parser.parse_args()

    try:
        gen = make_source(args.source, args.step)
    except IOError as e:
        print(e); sys.exit(1)

    # Pull the anchor frame (frame 0)
    img0 = next(gen, None)
    if img0 is None:
        print("No frames found"); sys.exit(1)

    K = build_K(img0.shape)
    print_K(K, img0.shape)

    # ── Find a second frame with enough baseline to bootstrap the map ─────────
    # Consecutive frames from a high-fps video often have near-zero parallax, so
    # we keep advancing until triangulation yields MIN_INIT_POINTS map points.
    MAX_INIT_SEARCH = 300  # give up after this many candidates
    # If the anchor frame loses visual overlap with the current video content
    # (e.g. after a cut or fast pan), advance it so we follow the scene.
    LOST_OVERLAP_THRESHOLD = 8
    result = None
    img1 = None
    skipped = 0
    anchor_frame = 0
    last_reason = "no frames available"
    for candidate in gen:
        result, last_reason = init_map(img0, candidate, K)
        skipped += 1
        if result is not None:
            img1 = candidate
            if skipped > 1:
                print(f"  (searched {skipped} frame(s), anchor at frame {anchor_frame})")
            break
        # When matches drop below the lost-overlap threshold the reference frame
        # has drifted out of view — advance the anchor to follow the video content.
        if "good matches" in last_reason:
            try:
                n_matches = int(last_reason.split()[2])
            except (ValueError, IndexError):
                n_matches = 0
            if n_matches < LOST_OVERLAP_THRESHOLD:
                img0 = candidate
                anchor_frame = skipped
        if skipped % 30 == 0:
            print(f"  Searching... [{skipped:3d}] anchor=frame {anchor_frame:3d}: {last_reason}")
        if skipped >= MAX_INIT_SEARCH:
            print(f"\nCould not initialise after {MAX_INIT_SEARCH} frames.")
            print(f"Last failure: {last_reason}")
            # Detect the pure-rotation pattern: many inliers but no triangulated points
            if "triangulated points" in last_reason and "inliers" in last_reason:
                print("\nDiagnosis: the camera appears to be ROTATING without translating.")
                print("Monocular 3D reconstruction is mathematically impossible with pure rotation —")
                print("triangulation requires two distinct camera positions, not just two orientations.")
                print("\nHow to fix — record a new video where the camera MOVES through space:")
                print("  • Walk forward into the scene")
                print("  • Walk sideways along a wall while facing it")
                print("  • Any motion where the camera physically travels (not just turns)")
                print("\nPanning/tilting/spinning in place will never work for monocular SLAM.")
            else:
                print("Suggestions:")
                print("  • Try '--step 2' — step=5 can be too large if the camera moves fast,")
                print("    causing features to shift too far between frames to match reliably")
                print("  • Check that the scene has visible texture (avoid plain walls, sky, etc.)")
            sys.exit(1)

    if result is None:
        print("Not enough frames to initialise the map"); sys.exit(1)

    map3d, map_des, R01, t01, kp0, kp1, des1, good01, inlier_mask = result

    # Pose convention: X_cam = R @ X_world + t  (frame 0 = world origin)
    trajectory = [
        (np.eye(3), np.zeros((3, 1))),  # frame 0
        (R01, t01),                      # frame 1
    ]

    prev_kp, prev_des = kp1, des1
    R_prev, t_prev    = R01, t01
    total = 2

    # ── Frame 2 onward: PnP tracking ─────────────────────────────────────────
    for img_i in gen:
        i = total
        R_i, t_i, n_inliers, kp_i, des_i = track_frame(img_i, K, map3d, map_des)

        if R_i is None:
            print(f"Frame {i:4d}: FAILED  (too few map matches)")
        else:
            trajectory.append((R_i, t_i))
            print(f"Frame {i:4d}: PnP inliers={n_inliers:3d}  |  map size={len(map3d):6d}")
            map3d, map_des = expand_map(map3d, map_des, K,
                                        prev_kp, prev_des, R_prev, t_prev,
                                        kp_i, des_i, R_i, t_i)
            R_prev, t_prev = R_i, t_i

        prev_kp, prev_des = kp_i, des_i
        total += 1

    print(f"\nTracked {len(trajectory)} / {total} frames")
    print(f"Final map: {len(map3d)} 3D points")

    # ── Visualise ─────────────────────────────────────────────────────────────
    show_inlier_matches(img0, kp0, img1, kp1, good01, inlier_mask)

    if len(trajectory) == 2:
        plot_3d(map3d, R01, t01)
    else:
        plot_trajectory(trajectory, map3d)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
