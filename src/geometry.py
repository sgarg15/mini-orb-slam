import cv2
import numpy as np


def estimate_pose(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=1.0)
    inlier_mask = mask.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, inlier_mask


def triangulate(K, R, t, pts1, pts2):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    valid = (pts4d[3] > 0) & (pts3d[:, 2] > 0)
    return pts3d[valid], pts1[valid], pts2[valid]


def reprojection_error(points_3d, pts1, pts2, K, R, t):
    proj1, _ = cv2.projectPoints(points_3d,
                                 np.zeros((3, 1)), np.zeros((3, 1)),
                                 K, None)
    proj2, _ = cv2.projectPoints(points_3d,
                                 cv2.Rodrigues(R)[0], t,
                                 K, None)

    err1 = np.linalg.norm(pts1 - proj1.reshape(-1, 2), axis=1)
    err2 = np.linalg.norm(pts2 - proj2.reshape(-1, 2), axis=1)
    return err1.mean(), err2.mean()
