import argparse
import os
import sys
import time

import cv2
import numpy as np

from camera import load_kitti_calibration, print_K, undistort_frame
from sources import frames_from_dir, kitti_image_dir, kitti_sequence_dir
from slam import (
    MIN_INIT_PARALLAX, MIN_VO_INLIERS,
    camera_center, compose_pose, estimate_relative_pose,
    expand_map, init_map, should_add_keyframe, track_frame,
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


def main():
    args = parse_args()

    image_dir = kitti_image_dir(args.source, args.kitti_camera)
    if image_dir is None:
        print(f"'{args.source}' is not a KITTI sequence directory")
        sys.exit(1)
    # Get a generator for the undistorted frames, so we can stop early if there are no frames or if we reach max_frames
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

    # Print the loaded intrinsics and distortion for confirmation, and undistort the first frame
    print_K(K, img0.shape, source=k_source, dist=dist)
    img0 = undistort_frame(img0, K, dist)

    # Initialization: find the first frame with enough parallax to img0 to initialize the map.
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
            # if we see a frame with some feature matches but not enough paralax to init, we move the anchor to that frame
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

    # Create an init trajectory with the R01 and t01 which are the pose of the second frame relative to the first frame
    trajectory = [
        (np.eye(3), np.zeros((3, 1))),
        (R01, t01),
    ]

    # Create a list of keyframe views for visualization, starting with the two init frame.
    keyframe_views = [
        {"frame": 0, "image": img0.copy(), "mode": "init", "trajectory_index": 0},
        {"frame": 1, "image": img1.copy(), "mode": "init", "trajectory_index": 1},
    ]

    # Setup the variables for the main tracking loop
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

    # Main tracking loop: for each subsequent frame, track it with PnP against the map. If that fails, optionally fall back to frame-to-frame VO using Essential Matrix. If tracking succeeds, decide whether to add a new keyframe and expand the map with triangulation.
    for img_i in gen:
        # Undistort the frame before processing, since both tracking and visualization assume undistorted frames. We need to do this before tracking so that the keypoints are detected in the undistorted image and match the map points which were triangulated in undistorted pixel coordinates.
        img_i = undistort_frame(img_i, K, dist)
        i = total
        # Using the track_frame, we attempt to track the current frame against the map with PnP
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
            # Add the pose to the trajectory
            trajectory.append((R_i, t_i))
            trajectory_index = len(trajectory) - 1

            # Decide whether to add a new keyframe. We check the median parallax of the matches to the current keyframe, the translation of the camera center from the last keyframe, and how many frames have passed since the last keyframe. If we add a new keyframe, we also expand the map with new triangulated points from the last keyframe to the current frame.
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
                keyframe_views.append({
                    "frame": i,
                    "image": img_i.copy(),
                    "mode": mode.strip(),
                    "trajectory_index": trajectory_index,
                })
                print(f"Frame {i:4d}: KEYFRAME  {mode} inliers={n_inliers:3d}  "
                      f"parallax={parallax:5.1f}px  trans={delta:.2f}  +{n_new} pts  map={len(map3d):6d}")
            else:
                print(f"Frame {i:4d}: tracked   {mode} inliers={n_inliers:3d}  "
                      f"matches={n_matches:3d}  parallax={parallax:5.1f}px  "
                      f"jump={pose_jump:.2f}  trans={delta:.2f}  map={len(map3d):6d}")

            if live_enabled:
                should_update_live = True
                if should_update_live:
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

    print(f"\nTracked {len(trajectory)} / {total} frames")
    print(f"Keyframes: {keyframes}")
    print(f"Final map: {len(map3d)} 3D points")

    if args.no_viz:
        return

    if live_stopped_by_user:
        return

    if args.live and args.no_final_viz:
        return

    if not args.live:
        show_inlier_matches(img0, kp0, img1, kp1, good01, inlier_mask)

    if len(trajectory) == 2:
        plot_3d(map3d, R01, t01)
    else:
        plot_trajectory(trajectory, map3d,
                        keyframes=keyframe_views,
                        max_keyframe_thumbs=8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
