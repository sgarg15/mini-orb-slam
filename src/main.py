import argparse
import glob
import os
import sys

import cv2
import numpy as np

from camera import build_K_from_args, load_calibration, load_kitti_calibration, print_K, undistort_frame
from features import detect, match, matched_points, match_to_map
from geometry import estimate_pose, reprojection_error, solve_pnp, triangulate, triangulate_poses
from visualization import plot_3d, plot_trajectory, show_inlier_matches


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
    if os.path.exists(os.path.join(path, "calib.txt")):
        return path
    nested = os.path.join(path, "sequences", "00")
    if os.path.exists(os.path.join(nested, "calib.txt")):
        return nested
    return None


def kitti_image_dir(sequence_dir, camera_id):
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


MIN_INIT_POINTS = 10
MIN_INIT_INLIERS = 8
MIN_PNP_INLIERS = 20
MIN_INIT_PARALLAX = 15.0
MIN_VO_INLIERS = 40


def init_map(img0, img1, K, min_parallax=MIN_INIT_PARALLAX):
    """Bootstrap the 3D map from two frames using Essential Matrix + triangulation."""
    kp0, des0 = detect(img0)
    kp1, des1 = detect(img1)

    _, good01 = match(des0, des1, ratio=0.80)
    if len(good01) < MIN_INIT_INLIERS:
        return None, f"only {len(good01)} good matches (need {MIN_INIT_INLIERS})"

    pts0, pts1 = matched_points(kp0, kp1, good01)
    median_parallax = float(np.median(np.linalg.norm(pts0 - pts1, axis=1)))
    if median_parallax < min_parallax:
        return None, (f"only {median_parallax:.1f}px median parallax "
                      f"(need {min_parallax:.1f}px)")

    try:
        R01, t01, inlier_mask = estimate_pose(pts0, pts1, K)
    except cv2.error as exc:
        return None, f"Essential Matrix failed: {exc}"

    n_inliers = int(inlier_mask.sum())
    if n_inliers < MIN_INIT_INLIERS:
        return None, f"only {n_inliers} RANSAC inliers (need {MIN_INIT_INLIERS})"

    inlier_idx = np.where(inlier_mask)[0]
    train_idx = np.array([good01[i].trainIdx for i in inlier_idx])
    des1_in = des1[train_idx]

    pts0_in = pts0[inlier_mask]
    pts1_in = pts1[inlier_mask]
    map3d, pts0_tri, pts1_tri, valid = triangulate(K, R01, t01, pts0_in, pts1_in)

    if len(map3d) < MIN_INIT_POINTS:
        return None, (f"only {len(map3d)} triangulated points (need {MIN_INIT_POINTS}); "
                      f"{n_inliers} inliers -> likely pure rotation or insufficient parallax")

    map_des = des1_in[valid]

    e1, e2 = reprojection_error(map3d, pts0_tri, pts1_tri, K, R01, t01)
    print(f"Init  : {len(map3d):5d} map points  |  parallax={median_parallax:.1f}px  |  "
          f"reprojection error  cam0={e1:.2f}px  cam1={e2:.2f}px")

    return (map3d, map_des, R01, t01, kp0, kp1, des1, good01, inlier_mask), None


def track_frame(img, K, map3d, map_des):
    """Estimate pose of a new frame by matching features to the 3D map."""
    kp, des = detect(img)
    if des is None or len(des) == 0:
        return None, None, 0, 0, kp, des

    frame_idx, map_idx = match_to_map(des, map_des)
    if len(frame_idx) < 6:
        return None, None, 0, len(frame_idx), kp, des

    pts2d = np.array([kp[j].pt for j in frame_idx], dtype=np.float64)
    pts3d = map3d[map_idx]

    R, t, inliers = solve_pnp(pts3d, pts2d, K)
    n_inliers = len(inliers) if inliers is not None else 0
    if n_inliers < MIN_PNP_INLIERS:
        return None, None, n_inliers, len(frame_idx), kp, des
    return R, t, n_inliers, len(frame_idx), kp, des


def camera_center(R, t):
    return (-R.T @ t).ravel()


def compose_pose(R_prev, t_prev, R_rel, t_rel, scale):
    """Compose a relative prev->curr pose with the previous world pose."""
    R_curr = R_rel @ R_prev
    t_curr = R_rel @ t_prev + scale * t_rel
    return R_curr, t_curr


def estimate_relative_pose(prev_kp, prev_des, curr_kp, curr_des, K,
                           min_inliers=MIN_VO_INLIERS, min_parallax=2.0):
    """Estimate frame-to-frame motion from Essential Matrix as a VO fallback."""
    _, good = match(prev_des, curr_des, ratio=0.75)
    if len(good) < min_inliers:
        return None, None, 0, 0.0

    pts_prev, pts_curr = matched_points(prev_kp, curr_kp, good)
    parallax = float(np.median(np.linalg.norm(pts_prev - pts_curr, axis=1)))
    if parallax < min_parallax:
        return None, None, 0, parallax

    try:
        R_rel, t_rel, inlier_mask = estimate_pose(pts_prev, pts_curr, K)
    except cv2.error:
        return None, None, 0, parallax

    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers:
        return None, None, n_inliers, parallax
    return R_rel, t_rel, n_inliers, parallax


def median_match_parallax(kp1, des1, kp2, des2):
    _, good = match(des1, des2)
    if len(good) < 8:
        return 0.0, len(good)
    pts1, pts2 = matched_points(kp1, kp2, good)
    return float(np.median(np.linalg.norm(pts1 - pts2, axis=1))), len(good)


def should_add_keyframe(key_kp, key_des, key_R, key_t,
                        curr_kp, curr_des, curr_R, curr_t,
                        min_parallax, min_translation,
                        frames_since_keyframe, min_frames):
    parallax, n_matches = median_match_parallax(key_kp, key_des, curr_kp, curr_des)
    delta = np.linalg.norm(camera_center(curr_R, curr_t) - camera_center(key_R, key_t))
    add = (frames_since_keyframe >= min_frames and
           parallax >= min_parallax and
           delta >= min_translation)
    return add, parallax, delta, n_matches


def expand_map(map3d, map_des, K, key_kp, key_des, key_R, key_t, curr_kp, curr_des, curr_R, curr_t):
    """Triangulate new 3D points from the last keyframe and append them to the map."""
    _, good = match(key_des, curr_des)
    if len(good) < 8:
        return map3d, map_des, 0

    pts_key, pts_curr = matched_points(key_kp, curr_kp, good)
    new_pts3d, valid = triangulate_poses(K, key_R, key_t, pts_key, curr_R, curr_t, pts_curr)
    if len(new_pts3d) == 0:
        return map3d, map_des, 0

    train_idx = np.array([good[j].trainIdx for j in range(len(good)) if valid[j]])
    new_des = curr_des[train_idx]

    map3d = np.vstack([map3d, new_pts3d])
    map_des = np.vstack([map_des, new_des])
    return map3d, map_des, len(new_pts3d)


def parse_args():
    parser = argparse.ArgumentParser(description="Mini ORB-SLAM: monocular visual odometry + sparse mapping")
    parser.add_argument("source", nargs="?", default="../data",
                        help="Video file or directory of images (default: ../data)")
    parser.add_argument("--step", type=int, default=1,
                        help="Process every Nth video frame (default: 1 = every frame)")
    parser.add_argument("--calib",
                        help="Calibration file: npz/yaml/json/txt with K and optional distortion coefficients")
    parser.add_argument("--kitti-camera", type=int, choices=(0, 1, 2, 3), default=0,
                        help="KITTI camera stream/projection to use when source is a KITTI sequence")
    parser.add_argument("--fx", type=float, help="Camera focal length x in pixels")
    parser.add_argument("--fy", type=float, help="Camera focal length y in pixels")
    parser.add_argument("--cx", type=float, help="Camera principal point x in pixels")
    parser.add_argument("--cy", type=float, help="Camera principal point y in pixels")
    parser.add_argument("--dist", type=float, nargs="*",
                        help="Optional distortion coefficients when using explicit intrinsics")
    parser.add_argument("--keyframe-min-parallax", type=float, default=25.0,
                        help="Median keypoint displacement in pixels required for a new keyframe")
    parser.add_argument("--keyframe-min-translation", type=float, default=0.15,
                        help="Minimum camera-center translation in arbitrary monocular scale")
    parser.add_argument("--keyframe-min-frames", type=int, default=5,
                        help="Minimum tracked frames between keyframes")
    parser.add_argument("--init-min-parallax", type=float, default=MIN_INIT_PARALLAX,
                        help="Minimum median pixel displacement for initialization")
    parser.add_argument("--max-pose-jump", type=float, default=10.0,
                        help="Reject PnP poses whose camera center jumps by more than this monocular-scale distance")
    parser.add_argument("--max-frames", type=int,
                        help="Limit the number of frames loaded from the source")
    parser.add_argument("--no-viz", action="store_true",
                        help="Run without opening OpenCV/matplotlib visualization windows")
    parser.add_argument("--disable-vo-fallback", action="store_true",
                        help="Do not fall back to frame-to-frame Essential Matrix tracking when PnP is weak")
    parser.add_argument("--vo-scale", type=float, default=1.0,
                        help="Arbitrary monocular scale for each frame-to-frame VO fallback step")
    parser.add_argument("--vo-min-inliers", type=int, default=MIN_VO_INLIERS,
                        help="Minimum Essential Matrix inliers for VO fallback")
    return parser.parse_args()


def load_camera(args, first_img):
    if args.calib:
        K, dist = load_calibration(args.calib)
        source = args.calib
    else:
        seq_dir = kitti_sequence_dir(args.source) if os.path.isdir(args.source) else None
        if seq_dir is None:
            K = build_K_from_args(args, first_img.shape)
            dist = np.asarray(args.dist, dtype=np.float64) if args.dist else None
            source = "CLI" if args.fx is not None else "estimated"
            return K, dist, source

        calib_path = os.path.join(seq_dir, "calib.txt")
        K, dist = load_kitti_calibration(calib_path, camera_id=args.kitti_camera)
        source = f"{calib_path}:P{args.kitti_camera}"
    return K, dist, source


def main():
    args = parse_args()

    try:
        gen = make_source(args.source, args.step,
                          kitti_camera=args.kitti_camera,
                          max_frames=args.max_frames)
    except IOError as exc:
        print(exc)
        sys.exit(1)

    img0 = next(gen, None)
    if img0 is None:
        print("No frames found")
        sys.exit(1)

    try:
        K, dist, k_source = load_camera(args, img0)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Calibration error: {exc}")
        sys.exit(1)

    print_K(K, img0.shape, source=k_source, dist=dist)
    img0 = undistort_frame(img0, K, dist)

    max_init_search = 300
    lost_overlap_threshold = 8
    result = None
    img1 = None
    skipped = 0
    anchor_frame = 0
    last_reason = "no frames available"

    for candidate in gen:
        candidate = undistort_frame(candidate, K, dist)
        result, last_reason = init_map(img0, candidate, K, min_parallax=args.init_min_parallax)
        skipped += 1
        if result is not None:
            img1 = candidate
            if skipped > 1:
                print(f"  (searched {skipped} frame(s), anchor at frame {anchor_frame})")
            break

        if "good matches" in last_reason:
            try:
                n_matches = int(last_reason.split()[1])
            except (ValueError, IndexError):
                n_matches = 0
            if n_matches < lost_overlap_threshold:
                img0 = candidate
                anchor_frame = skipped

        if skipped % 30 == 0:
            print(f"  Searching... [{skipped:3d}] anchor=frame {anchor_frame:3d}: {last_reason}")

        if skipped >= max_init_search:
            print(f"\nCould not initialise after {max_init_search} frames.")
            print(f"Last failure: {last_reason}")
            print("Suggestions:")
            print("  - Use real camera intrinsics and distortion with --calib or --fx/--fy/--cx/--cy.")
            print("  - Record translating camera motion; pure rotation cannot triangulate monocular depth.")
            print("  - Try a smaller --step if features move too far between processed frames.")
            sys.exit(1)

    if result is None:
        print("Not enough frames to initialise the map")
        sys.exit(1)

    map3d, map_des, R01, t01, kp0, kp1, des1, good01, inlier_mask = result

    trajectory = [
        (np.eye(3), np.zeros((3, 1))),
        (R01, t01),
    ]
    key_kp, key_des = kp1, des1
    key_R, key_t = R01, t01
    keyframes = 2
    last_keyframe_frame = 1
    prev_R, prev_t = R01, t01
    prev_kp, prev_des = kp1, des1
    total = 2

    for img_i in gen:
        img_i = undistort_frame(img_i, K, dist)
        i = total
        R_i, t_i, n_inliers, n_matches, kp_i, des_i = track_frame(img_i, K, map3d, map_des)
        mode = "PnP"
        pose_jump = None

        if R_i is not None:
            pose_jump = np.linalg.norm(camera_center(R_i, t_i) - camera_center(prev_R, prev_t))
            if pose_jump > args.max_pose_jump:
                R_i, t_i = None, None

        if R_i is None and not args.disable_vo_fallback:
            R_rel, t_rel, vo_inliers, vo_parallax = estimate_relative_pose(
                prev_kp, prev_des, kp_i, des_i, K,
                min_inliers=args.vo_min_inliers)
            if R_rel is not None:
                R_i, t_i = compose_pose(prev_R, prev_t, R_rel, t_rel, args.vo_scale)
                pose_jump = np.linalg.norm(camera_center(R_i, t_i) - camera_center(prev_R, prev_t))
                n_inliers = vo_inliers
                n_matches = 0
                mode = "VO "
            else:
                print(f"Frame {i:4d}: FAILED  (map matches={n_matches}, PnP inliers={n_inliers}, "
                      f"VO inliers={vo_inliers}, VO parallax={vo_parallax:.1f}px)")
                total += 1
                continue

        if R_i is None:
            print(f"Frame {i:4d}: FAILED  (map matches={n_matches}, PnP inliers={n_inliers})")
        else:
            trajectory.append((R_i, t_i))
            add_keyframe, parallax, delta, _ = should_add_keyframe(
                key_kp, key_des, key_R, key_t,
                kp_i, des_i, R_i, t_i,
                args.keyframe_min_parallax, args.keyframe_min_translation,
                i - last_keyframe_frame, args.keyframe_min_frames)

            if add_keyframe:
                map3d, map_des, n_new = expand_map(
                    map3d, map_des, K,
                    key_kp, key_des, key_R, key_t,
                    kp_i, des_i, R_i, t_i)
                key_kp, key_des = kp_i, des_i
                key_R, key_t = R_i, t_i
                keyframes += 1
                last_keyframe_frame = i
                print(f"Frame {i:4d}: KEYFRAME  {mode} inliers={n_inliers:3d}  "
                      f"parallax={parallax:5.1f}px  trans={delta:.2f}  +{n_new} pts  map={len(map3d):6d}")
            else:
                print(f"Frame {i:4d}: tracked   {mode} inliers={n_inliers:3d}  "
                      f"matches={n_matches:3d}  parallax={parallax:5.1f}px  "
                      f"jump={pose_jump:.2f}  trans={delta:.2f}  map={len(map3d):6d}")

            prev_R, prev_t = R_i, t_i
            prev_kp, prev_des = kp_i, des_i

        total += 1

    print(f"\nTracked {len(trajectory)} / {total} frames")
    print(f"Keyframes: {keyframes}")
    print(f"Final map: {len(map3d)} 3D points")

    if args.no_viz:
        return

    show_inlier_matches(img0, kp0, img1, kp1, good01, inlier_mask)

    if len(trajectory) == 2:
        plot_3d(map3d, R01, t01)
    else:
        plot_trajectory(trajectory, map3d)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
