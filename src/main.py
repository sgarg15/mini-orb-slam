import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from camera import load_kitti_calibration, print_K, undistort_frame
from sources import frames_from_dir, kitti_image_dir, kitti_sequence_dir
from slam import (
    DUPLICATE_POINT_RADIUS, MAX_MAP_POINT_DISTANCE, MIN_INIT_PARALLAX, MIN_VO_INLIERS,
    camera_center, compose_pose, estimate_relative_pose,
    cull_map_points, expand_map, init_map, should_add_keyframe, track_frame,
)
from visualization import LiveSlamViewer, plot_3d, plot_trajectory, show_inlier_matches


def parse_args():
    parser = argparse.ArgumentParser(description="Mini ORB-SLAM: monocular visual odometry + sparse mapping")
    parser.add_argument("source", nargs="?", default="../data",
                        help="KITTI sequence directory (default: ../data)")
    parser.add_argument("--kitti-camera", type=int, choices=(0, 1, 2, 3), default=0,
                        help="KITTI camera stream to use (default: 0)")
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
    parser.add_argument("--live", action="store_true",
                        help="Update a matplotlib SLAM dashboard while frames are processed")
    parser.add_argument("--no-final-viz", action="store_true",
                        help="Skip the final blocking matplotlib dashboard after a live run")
    parser.add_argument("--disable-vo-fallback", action="store_true",
                        help="Do not fall back to frame-to-frame Essential Matrix tracking when PnP is weak")
    parser.add_argument("--vo-scale", type=float, default=1.0,
                        help="Arbitrary monocular scale for each frame-to-frame VO fallback step")
    parser.add_argument("--vo-min-inliers", type=int, default=MIN_VO_INLIERS,
                        help="Minimum Essential Matrix inliers for VO fallback")
    parser.add_argument("--max-map-point-distance", type=float, default=MAX_MAP_POINT_DISTANCE,
                        help="Reject new map points farther than this world-frame distance from the origin")
    parser.add_argument("--duplicate-point-radius", type=float, default=DUPLICATE_POINT_RADIUS,
                        help="Reject new map points closer than this distance to an existing/new map point")
    parser.add_argument("--min-map-observations", type=int, default=2,
                        help="Cull map points observed fewer than this many times after keyframe insertion")
    return parser.parse_args()


def load_camera(args):
    """
    Load camera intrinsics and distortion from the KITTI sequence calibration file.

    Returns:
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients (k1, k2, p1, p2, k3)
        k_source: string describing the source of the intrinsics (for display purposes)
    """
    seq_dir = kitti_sequence_dir(args.source)
    if seq_dir is None:
        raise ValueError(f"'{args.source}' is not a KITTI sequence directory (no calib.txt found)")
    calib_path = os.path.join(seq_dir, "calib.txt")
    K, dist = load_kitti_calibration(calib_path, camera_id=args.kitti_camera)
    return K, dist, f"{calib_path}:P{args.kitti_camera}"


def _find_init_frame(gen, img0, K, dist, args):
    """Search gen for a second frame with enough parallax to initialize the map.

    May advance img0 to a later anchor frame if feature overlap is lost.
    Returns (img0_anchor, img1, result, frames_consumed) or calls sys.exit on failure.
    """
    max_init_search = 300
    lost_overlap_threshold = 8
    result = None
    img1 = None
    skipped = 0
    anchor_frame = 0
    last_reason = "no frames available"

    # We search for an initialization frame with enough parallax from img0, but if we see a frame with some feature matches but not enough parallax, we move the anchor to that frame, since it may be more likely to find good parallax with subsequent frames.
    for candidate in gen:
        candidate = undistort_frame(candidate, K, dist)
        # Attempt to init the map from img0 and the candidate frame. If it fails, we check the reason for failure to decide whether to move the anchor frame or just keep searching.
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
            # Move the anchor forward when feature overlap is nearly lost so subsequent
            # frames are more likely to share enough features for initialization.
            if n_matches < lost_overlap_threshold:
                img0 = candidate
                anchor_frame = skipped

        if skipped % 30 == 0:
            print(f"  Searching... [{skipped:3d}] anchor=frame {anchor_frame:3d}: {last_reason}")

        if skipped >= max_init_search:
            print(f"\nCould not initialise after {max_init_search} frames.")
            print(f"Last failure: {last_reason}")
            print("Suggestions:")
            print("  - Record translating camera motion; pure rotation cannot triangulate monocular depth.")
            sys.exit(1)

    if result is None:
        print("Not enough frames to initialise the map")
        sys.exit(1)

    return img0, img1, result, skipped + 1


def _track_frames(gen, K, dist, img0, img1, result, args):
    """Run the main PnP tracking loop over remaining frames.

    Returns (trajectory, map3d, keyframe_views, keyframes, total, live_stopped_by_user, metrics).
    """
    map3d, map_des, R01, t01, _, kp1, des1, _, inlier_mask = result
    map_obs = np.full(len(map3d), 2, dtype=np.int32)

    trajectory = [
        (np.eye(3), np.zeros((3, 1))),
        (R01, t01),
    ]
    keyframe_views = [
        {"frame": 0, "image": img0.copy(), "mode": "init", "trajectory_index": 0},
        {"frame": 1, "image": img1.copy(), "mode": "init", "trajectory_index": 1},
    ]

    key_kp, key_des = kp1, des1
    key_R, key_t = R01, t01
    keyframes = 2
    last_keyframe_frame = 1
    prev_R, prev_t = R01, t01
    prev_kp, prev_des = kp1, des1
    total = 2
    live_enabled = args.live and not args.no_viz
    live_viewer = None
    live_min_dt = 1.0 / 10.0
    last_live_update = 0.0
    live_stopped_by_user = False
    metrics = {
        "tracking_failures": 0,
        "keyframes_inserted": 0,
        "map_points_created": int(len(map3d)),
        "map_points_culled": 0,
        "rejected_far_points": 0,
        "rejected_duplicate_points": 0,
        "pnp_inliers": [int(inlier_mask.sum())],
        "pnp_reprojection_errors": [],
        "tracking_seconds": 0.0,
    }

    if args.live and args.no_viz:
        print("Note: --live is ignored because --no-viz is set")

    if live_enabled:
        live_viewer = LiveSlamViewer(max_keyframe_thumbs=8)
        status = {
            "frame": 1,
            "mode": "init",
            "inliers": int(inlier_mask.sum()),
            "map_points": len(map3d),
            "keyframes": keyframes,
            "message": "initialized",
        }
        live_viewer.update(trajectory, map3d, img1, keyframe_views, status)
        last_live_update = time.perf_counter()

    tracking_started = time.perf_counter()

    for img_i in gen:
        img_i = undistort_frame(img_i, K, dist)
        i = total
        R_i, t_i, n_inliers, n_matches, kp_i, des_i, map_inliers, reproj_errors = track_frame(
            img_i, K, map3d, map_des)
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
                map_inliers = np.array([], dtype=int)
                reproj_errors = np.array([], dtype=np.float64)
                mode = "VO "
            else:
                print(f"Frame {i:4d}: FAILED  (map matches={n_matches}, PnP inliers={n_inliers}, "
                      f"VO inliers={vo_inliers}, VO parallax={vo_parallax:.1f}px)")
                metrics["tracking_failures"] += 1
                total += 1
                continue

        if R_i is None:
            print(f"Frame {i:4d}: FAILED  (map matches={n_matches}, PnP inliers={n_inliers})")
            metrics["tracking_failures"] += 1
        else:
            trajectory.append((R_i, t_i))
            trajectory_index = len(trajectory) - 1
            if mode.strip() == "PnP":
                metrics["pnp_inliers"].append(int(n_inliers))
                metrics["pnp_reprojection_errors"].extend([float(e) for e in reproj_errors if np.isfinite(e)])
                if len(map_inliers) > 0:
                    np.add.at(map_obs, map_inliers, 1)

            add_keyframe, parallax, delta, _ = should_add_keyframe(
                key_kp, key_des, key_R, key_t,
                kp_i, des_i, R_i, t_i,
                args.keyframe_min_parallax, args.keyframe_min_translation,
                i - last_keyframe_frame, args.keyframe_min_frames)

            if add_keyframe:
                map3d, map_des, map_obs, n_new, hygiene = expand_map(
                    map3d, map_des, map_obs, K,
                    key_kp, key_des, key_R, key_t,
                    kp_i, des_i, R_i, t_i,
                    max_point_distance=args.max_map_point_distance,
                    duplicate_radius=args.duplicate_point_radius)
                map3d, map_des, map_obs, n_culled = cull_map_points(
                    map3d, map_des, map_obs, args.min_map_observations)
                metrics["keyframes_inserted"] += 1
                metrics["map_points_created"] += int(n_new)
                metrics["map_points_culled"] += int(n_culled)
                metrics["rejected_far_points"] += int(hygiene["rejected_far"])
                metrics["rejected_duplicate_points"] += int(hygiene["rejected_duplicate"])

                key_kp, key_des = kp_i, des_i
                key_R, key_t = R_i, t_i
                keyframes += 1
                last_keyframe_frame = i
                keyframe_views.append({
                    "frame": i,
                    "image": img_i.copy(),
                    "mode": mode.strip(),
                    "trajectory_index": trajectory_index,
                })
                print(f"Frame {i:4d}: KEYFRAME  {mode} inliers={n_inliers:3d}  "
                      f"parallax={parallax:5.1f}px  trans={delta:.2f}  "
                      f"+{n_new} pts  -{n_culled} culled  map={len(map3d):6d}")
            else:
                print(f"Frame {i:4d}: tracked   {mode} inliers={n_inliers:3d}  "
                      f"matches={n_matches:3d}  parallax={parallax:5.1f}px  "
                      f"jump={pose_jump:.2f}  trans={delta:.2f}  map={len(map3d):6d}")

            if live_enabled:
                now = time.perf_counter()
                sleep_for = live_min_dt - (now - last_live_update)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                status = {
                    "frame": i,
                    "mode": "KEYFRAME " + mode.strip() if add_keyframe else mode.strip(),
                    "inliers": n_inliers,
                    "map_points": len(map3d),
                    "keyframes": keyframes,
                    "message": f"parallax={parallax:.1f}px  trans={delta:.2f}",
                }
                if not live_viewer.update(trajectory, map3d, img_i, keyframe_views, status):
                    print("Live viewer closed; stopping run.")
                    live_stopped_by_user = True
                    total += 1
                    break
                last_live_update = time.perf_counter()

            prev_R, prev_t = R_i, t_i
            prev_kp, prev_des = kp_i, des_i

        total += 1

    metrics["tracking_seconds"] = time.perf_counter() - tracking_started
    metrics["final_observation_counts"] = map_obs
    return trajectory, map3d, keyframe_views, keyframes, total, live_stopped_by_user, metrics


def _run_save_paths():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(repo_root, "docs", "runs")
    os.makedirs(save_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(save_dir, f"run_{stamp}")
    return f"{stem}.png", f"{stem}_summary.json"


def _show_final_viz(args, trajectory, map3d, keyframe_views,
                    img0, kp0, img1, kp1, good01, inlier_mask, R01, t01,
                    live_stopped_by_user, save_path):
    if args.no_viz or live_stopped_by_user:
        return
    if args.live and args.no_final_viz:
        return

    if not args.live:
        show_inlier_matches(img0, kp0, img1, kp1, good01, inlier_mask)

    if len(trajectory) == 2:
        plot_3d(map3d, R01, t01, save_path=save_path)
    else:
        plot_trajectory(trajectory, map3d,
                        keyframes=keyframe_views,
                        max_keyframe_thumbs=8,
                        save_path=save_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _build_summary(total, init_frames_processed, trajectory, keyframes, map3d, metrics, args):
    pnp_inliers = metrics["pnp_inliers"]
    reproj_errors = metrics["pnp_reprojection_errors"]
    tracking_seconds = metrics["tracking_seconds"]
    mean_fps = (max(total - 2, 0) / tracking_seconds) if tracking_seconds > 0 else 0.0
    obs = metrics["final_observation_counts"]
    frames_processed = init_frames_processed + max(total - 2, 0)
    return {
        "frames_processed": int(frames_processed),
        "accepted_poses": int(len(trajectory)),
        "tracking_failures": int(metrics["tracking_failures"]),
        "keyframes_inserted": int(metrics["keyframes_inserted"]),
        "keyframes_total": int(keyframes),
        "map_points_created": int(metrics["map_points_created"]),
        "map_points_final": int(len(map3d)),
        "map_points_culled": int(metrics["map_points_culled"]),
        "min_map_observations": int(args.min_map_observations),
        "map_points_observed_fewer_than_min": int(np.count_nonzero(obs < args.min_map_observations)) if len(obs) else 0,
        "rejected_far_points": int(metrics["rejected_far_points"]),
        "rejected_duplicate_points": int(metrics["rejected_duplicate_points"]),
        "mean_pnp_inliers": float(np.mean(pnp_inliers)) if pnp_inliers else 0.0,
        "median_reprojection_error_px": float(np.median(reproj_errors)) if reproj_errors else 0.0,
        "mean_tracking_fps": float(mean_fps),
    }


def _print_summary(summary):
    print("\nFinal run summary")
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Accepted poses: {summary['accepted_poses']}")
    print(f"Tracking failures: {summary['tracking_failures']}")
    print(f"Keyframes inserted: {summary['keyframes_inserted']}")
    print(f"Map points created: {summary['map_points_created']}")
    print(f"Mean PnP inliers: {summary['mean_pnp_inliers']:.1f}")
    print(f"Median reprojection error: {summary['median_reprojection_error_px']:.2f} px")
    print(f"Mean tracking FPS: {summary['mean_tracking_fps']:.1f}")


def _save_summary(summary, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Saved run summary -> {save_path}")


def main():
    args = parse_args()

    image_dir = kitti_image_dir(args.source, args.kitti_camera)
    if image_dir is None:
        print(f"'{args.source}' is not a KITTI sequence directory")
        sys.exit(1)
    gen = frames_from_dir(image_dir, max_frames=args.max_frames)

    img0 = next(gen, None)
    if img0 is None:
        print("No frames found")
        sys.exit(1)

    try:
        K, dist, k_source = load_camera(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Calibration error: {exc}")
        sys.exit(1)

    print_K(K, img0.shape, source=k_source, dist=dist)
    img0 = undistort_frame(img0, K, dist)

    img0, img1, result, init_frames_processed = _find_init_frame(gen, img0, K, dist, args)
    map3d, _, R01, t01, kp0, kp1, _, good01, inlier_mask = result

    plot_path, summary_path = _run_save_paths()

    trajectory, map3d, keyframe_views, keyframes, total, live_stopped_by_user, metrics = _track_frames(
        gen, K, dist, img0, img1, result, args)

    summary = _build_summary(total, init_frames_processed, trajectory, keyframes, map3d, metrics, args)
    _print_summary(summary)
    _save_summary(summary, summary_path)

    _show_final_viz(args, trajectory, map3d, keyframe_views,
                    img0, kp0, img1, kp1, good01, inlier_mask, R01, t01,
                    live_stopped_by_user, plot_path)


if __name__ == "__main__":
    main()
