import cv2
import numpy as np

# Geometry functions: pose estimation, triangulation, reprojection error
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

# Triangulate 3D points from corresponding 2D points in two images given the camera intrinsics and relative pose. We also filter out points that are behind the camera or have invalid triangulation results.
def triangulate(K, R, t, pts1, pts2):
    # Calculate P1 and P2 which are the projection matrices for the two cameras. P1 corresponds to the first camera (identity rotation and zero translation), while P2 corresponds to the second camera with rotation R and translation t.
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # Triangulate the 3D points from the corresponding 2D points in the two images. The function cv2.triangulatePoints expects the points to be in homogeneous coordinates, so we transpose them and convert to float32. The output pts4d is in homogeneous coordinates (x, y, z, w), so we convert it to 3D by dividing by w.
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Filter out points that are behind the camera (negative depth) or have invalid homogeneous coordinates (w <= 0). We only keep points that are in front of the camera and have valid triangulation results.
    valid = (pts4d[3] > 0) & (pts3d[:, 2] > 0)
    return pts3d[valid], pts1[valid], pts2[valid]


def reprojection_error(points_3d, pts1, pts2, K, R, t):
    # Reproject the 3D points back to the image planes of both cameras using the known camera intrinsics and relative pose. We use cv2.projectPoints to project the 3D points into 2D pixel coordinates for both cameras.
    proj1, _ = cv2.projectPoints(points_3d,
                                 np.zeros((3, 1)), np.zeros((3, 1)),
                                 K, None)
    proj2, _ = cv2.projectPoints(points_3d,
                                 cv2.Rodrigues(R)[0], t,
                                 K, None)

    # Calculate the reprojection error as the average pixel distance between the original 2D points and the reprojected points for both cameras. We compute the Euclidean distance between the original points and the projected points, and return the mean error for each camera.
    err1 = np.linalg.norm(pts1 - proj1.reshape(-1, 2), axis=1)
    err2 = np.linalg.norm(pts2 - proj2.reshape(-1, 2), axis=1)
    return err1.mean(), err2.mean()
