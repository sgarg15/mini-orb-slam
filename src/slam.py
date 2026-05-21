import cv2
import numpy as np

from features import detect, match, matched_points, match_to_map
from geometry import estimate_pose, project_points, reprojection_error, solve_pnp, triangulate, triangulate_poses

MIN_INIT_POINTS = 10
MIN_INIT_INLIERS = 8
MIN_PNP_INLIERS = 20
MIN_INIT_PARALLAX = 15.0
MIN_VO_INLIERS = 40
MAX_MAP_POINT_DISTANCE = 80.0
DUPLICATE_POINT_RADIUS = 0.25


def init_map(img0, img1, K, min_parallax=MIN_INIT_PARALLAX):
    """
    Bootstrap the 3D map from two frames using Essential Matrix + triangulation.
    
    Returns:
        If successful, a tuple containing:
        - map3d: Nx3 array of 3D point positions in the world frame
        - map_des: NxD array of descriptors for the 3D points (from the second frame)
        - R01: 3x3 rotation matrix from frame 0 to frame
        - t01: 3x1 translation vector from frame 0 to frame
        - kp0: list of keypoints in img0
        - kp1: list of keypoints in img1
        - des1: descriptors for the keypoints in img1
        - good01: list of good matches between des0 and des1 used for initialization
        - inlier_mask: boolean array indicating which matches were inliers to the Essential Matrix
        None, reason: if initialization failed, with a string reason for failure
    """
    kp0, des0 = detect(img0)
    kp1, des1 = detect(img1)

    _, good01 = match(des0, des1, ratio=0.80)
    # We require a minimum number of matches and parallax to avoid initializing from a pure rotation or very distant frames, which would lead to a degenerate map and bad tracking.
    if len(good01) < MIN_INIT_INLIERS:
        return None, f"only {len(good01)} good matches (need {MIN_INIT_INLIERS})"

    pts0, pts1 = matched_points(kp0, kp1, good01)
    median_parallax = float(np.median(np.linalg.norm(pts0 - pts1, axis=1)))
    # If the parallax is too small, the Essential Matrix estimation will be unstable and likely fail to find a valid pose, so we check this before attempting pose estimation.
    if median_parallax < min_parallax:
        return None, (f"only {median_parallax:.1f}px median parallax "
                      f"(need {min_parallax:.1f}px)")

    try:
        R01, t01, inlier_mask = estimate_pose(pts0, pts1, K)
    except cv2.error as exc:
        return None, f"Essential Matrix failed: {exc}"

    n_inliers = int(inlier_mask.sum())
    # If the number of inliers is too small, the pose estimation is likely unreliable and the triangulated map will be poor, so we check this before attempting triangulation.
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
    """
    Estimate pose of a new frame by matching features to the 3D map.
    
    Returns:
        R: 3x3 rotation matrix of the current frame in the world coordinate system

        t: 3x1 translation vector of the current frame in the world coordinate system
        
        n_inliers: number of inlier matches used for the PnP pose estimation
        
        n_matches: total number of matches found between the frame and the map (including outliers)
        
        kp: list of keypoints detected in the current frame
        
        des: descriptors for the keypoints detected in the current frame

        map_inliers: indices of map points that were PnP inliers

        reproj_errors: reprojection errors for PnP inlier correspondences
    """

    kp, des = detect(img)
    if des is None or len(des) == 0:
        return None, None, 0, 0, kp, des, np.array([], dtype=int), np.array([], dtype=np.float64)

    frame_idx, map_idx = match_to_map(des, map_des)
    if len(frame_idx) < 6:
        return None, None, 0, len(frame_idx), kp, des, np.array([], dtype=int), np.array([], dtype=np.float64)

    # Get the 2D-3D correspondences for PnP. The 2D points are the keypoints in the current frame corresponding to the matched descriptors, and the 3D points are the map points corresponding to those descriptors.
    pts2d = np.array([kp[j].pt for j in frame_idx], dtype=np.float64)
    pts3d = map3d[map_idx]

    # Finally, we solve the PnP problem to get the pose of the current frame. We also get the inliers from RANSAC to evaluate the quality of the pose estimation. If there are too few inliers, we consider the tracking to have failed for this frame.
    R, t, inliers = solve_pnp(pts3d, pts2d, K)
    n_inliers = len(inliers) if inliers is not None else 0
    if n_inliers < MIN_PNP_INLIERS:
        return None, None, n_inliers, len(frame_idx), kp, des, np.array([], dtype=int), np.array([], dtype=np.float64)

    inlier_pts3d = pts3d[inliers]
    inlier_pts2d = pts2d[inliers]
    proj = project_points(inlier_pts3d, K, R, t)
    reproj_errors = np.linalg.norm(inlier_pts2d - proj, axis=1)
    map_inliers = map_idx[inliers]
    
    # Return the pose of the current frame, the number of inliers, the total number of matches to the map, and the detected keypoints and descriptors for potential use in visualization or future matching.
    return R, t, n_inliers, len(frame_idx), kp, des, map_inliers, reproj_errors


def camera_center(R, t):
    """
    Compute the camera center in world coordinates from the pose (R, t).
    """
    return (-R.T @ t).ravel()


def compose_pose(R_prev, t_prev, R_rel, t_rel, scale):
    """Compose a relative prev->curr pose with the previous world pose."""
    R_curr = R_rel @ R_prev
    t_curr = R_rel @ t_prev + scale * t_rel
    return R_curr, t_curr


def estimate_relative_pose(prev_kp, prev_des, curr_kp, curr_des, K,
                            min_inliers=MIN_VO_INLIERS, min_parallax=2.0):
    """Estimate frame-to-frame motion from Essential Matrix as a VO (Visual Odometry) fallback."""
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
    """
    Decide whether to add a new keyframe based on the parallax of matches to the current keyframe, the translation of the camera center from the last keyframe, and how many frames have passed since the last keyframe
    
    Returns:
    add: boolean indicating whether to add a new keyframe
    parallax: the median parallax of the matches between the current frame and the last key
    delta: the translation of the camera center from the last keyframe
    """
    # First check the parallax of the matches between the curr frame and the last frame
    parallax, n_matches = median_match_parallax(key_kp, key_des, curr_kp, curr_des)
    delta = np.linalg.norm(camera_center(curr_R, curr_t) - camera_center(key_R, key_t))
    add = (frames_since_keyframe >= min_frames and
           parallax >= min_parallax and
           delta >= min_translation)
    return add, parallax, delta, n_matches


def _distance_filter(points_3d, max_distance):
    if len(points_3d) == 0:
        return np.array([], dtype=bool)
    return np.linalg.norm(points_3d, axis=1) <= max_distance


def _duplicate_filter(points_3d, existing_points, radius):
    """Keep points that are not near existing points or earlier new points."""
    if len(points_3d) == 0:
        return np.array([], dtype=bool)
    keep = np.ones(len(points_3d), dtype=bool)
    radius2 = radius * radius
    for i, point in enumerate(points_3d):
        if len(existing_points) > 0 and np.any(np.sum((existing_points - point) ** 2, axis=1) < radius2):
            keep[i] = False
            continue
        if i > 0 and np.any(np.sum((points_3d[:i][keep[:i]] - point) ** 2, axis=1) < radius2):
            keep[i] = False
    return keep


def cull_map_points(map3d, map_des, map_obs, min_observations):
    if min_observations <= 1 or len(map3d) == 0:
        return map3d, map_des, map_obs, 0
    keep = map_obs >= min_observations
    culled = int(np.count_nonzero(~keep))
    return map3d[keep], map_des[keep], map_obs[keep], culled


def expand_map(map3d, map_des, map_obs, K,
               key_kp, key_des, key_R, key_t,
               curr_kp, curr_des, curr_R, curr_t,
               max_point_distance=MAX_MAP_POINT_DISTANCE,
               duplicate_radius=DUPLICATE_POINT_RADIUS):
    """
    Triangulate new 3D points from the last keyframe and append them to the map.
    
    Returns:
    map3d: updated Nx3 array of 3D point positions in the world frame
    map_des: updated NxD array of descriptors for the 3D points (from the current frame)
    map_obs: updated observation counts for each map point
    n_new: number of new points added to the map
    stats: filtering counts from the map hygiene pass
    """
    _, good = match(key_des, curr_des)
    stats = {
        "triangulated": 0,
        "rejected_far": 0,
        "rejected_duplicate": 0,
    }
    if len(good) < 8:
        return map3d, map_des, map_obs, 0, stats

    pts_key, pts_curr = matched_points(key_kp, curr_kp, good)
    new_pts3d, valid = triangulate_poses(K, key_R, key_t, pts_key, curr_R, curr_t, pts_curr)
    if len(new_pts3d) == 0:
        return map3d, map_des, map_obs, 0, stats

    train_idx = np.array([good[j].trainIdx for j in range(len(good)) if valid[j]])
    stats["triangulated"] = int(len(new_pts3d))

    near_enough = _distance_filter(new_pts3d, max_point_distance)
    stats["rejected_far"] = int(np.count_nonzero(~near_enough))
    new_pts3d = new_pts3d[near_enough]
    train_idx = train_idx[near_enough]
    if len(new_pts3d) == 0:
        return map3d, map_des, map_obs, 0, stats

    unique = _duplicate_filter(new_pts3d, map3d, duplicate_radius)
    stats["rejected_duplicate"] = int(np.count_nonzero(~unique))
    new_pts3d = new_pts3d[unique]
    train_idx = train_idx[unique]
    if len(new_pts3d) == 0:
        return map3d, map_des, map_obs, 0, stats

    new_des = curr_des[train_idx]
    new_obs = np.full(len(new_pts3d), 2, dtype=np.int32)

    map3d = np.vstack([map3d, new_pts3d])
    map_des = np.vstack([map_des, new_des])
    map_obs = np.concatenate([map_obs, new_obs])
    return map3d, map_des, map_obs, len(new_pts3d), stats
