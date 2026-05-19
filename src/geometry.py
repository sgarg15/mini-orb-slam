import cv2
import numpy as np

# Geometry functions: pose estimation, triangulation, PnP, reprojection error

def estimate_pose(pts1, pts2, K):
    # Compute the essential matrix using RANSAC to find inliers. Inliers are those that fit the epipolar constraint well.
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=1.0)
    # mask is a binary array where 1 indicates inliers and 0 indicates outliers. We convert it to a boolean mask.
    inlier_mask = mask.ravel().astype(bool)
    # Recover the relative camera pose (R, t) from the essential matrix using only the inlier matches. This will give us the rotation and translation that best explain the inlier correspondences.
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, inlier_mask


def triangulate(K, R, t, pts1, pts2):
    """Triangulate 3D points with frame 0 at the world origin.

    Returns (points_3d, pts1_valid, pts2_valid, valid_mask).
    valid_mask is a boolean array of length len(pts1).
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts4d = cv2.triangulatePoints(P1, P2,
                                  pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T
    # Check positive depth in both camera frames
    depth2 = (R @ pts3d.T + t)[2]
    valid = (pts4d[3] > 0) & (pts3d[:, 2] > 0) & (depth2 > 0)
    return pts3d[valid], pts1[valid], pts2[valid], valid


def triangulate_poses(K, R1, t1, pts1, R2, t2, pts2):
    """Triangulate 3D points from two frames with absolute camera poses.

    Convention: X_cam = R @ X_world + t.
    Returns (points_3d_valid, valid_mask) where valid_mask has length len(pts1).
    """
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    pts4d = cv2.triangulatePoints(P1, P2,
                                  pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-6
    pts3d = np.zeros((pts4d.shape[1], 3))
    pts3d[valid_w] = (pts4d[:3, valid_w] / w[valid_w]).T
    # Require positive depth in both camera frames
    depth1 = (R1 @ pts3d.T + t1)[2]
    depth2 = (R2 @ pts3d.T + t2)[2]
    valid = valid_w & (depth1 > 0) & (depth2 > 0)
    return pts3d[valid], valid


def solve_pnp(points_3d, pts2d, K):
    """Estimate camera pose from 3D-2D correspondences using PnP + RANSAC.

    Convention: X_cam = R @ X_world + t.
    Returns (R, t, inlier_indices) or (None, None, None) on failure.
    """
    if len(points_3d) < 6:
        return None, None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float64),
        pts2d.astype(np.float64),
        K,
        None,
        confidence=0.999,
        reprojectionError=8.0,
        iterationsCount=200,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, inliers.flatten()


def reprojection_error(points_3d, pts1, pts2, K, R, t):
    if len(points_3d) == 0:
        return 0.0, 0.0
    # Reproject the 3D points back to the image planes of both cameras using the known camera intrinsics and relative pose. We use cv2.projectPoints to project the 3D points into 2D pixel coordinates for both cameras.
    proj1, _ = cv2.projectPoints(points_3d,
                                 np.zeros((3, 1)), np.zeros((3, 1)),
                                 K, None)
    proj2, _ = cv2.projectPoints(points_3d,
                                 cv2.Rodrigues(R)[0], t,
                                 K, None)
    err1 = np.linalg.norm(pts1 - proj1.reshape(-1, 2), axis=1)
    err2 = np.linalg.norm(pts2 - proj2.reshape(-1, 2), axis=1)
    return err1.mean(), err2.mean()
