import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    orb = cv2.ORB_create(nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8)

    img1 = cv2.imread('../data/img2.jpg', 0)
    kp1, des1 = orb.detectAndCompute(img1, None)

    # Feature matching with another image
    img2 = cv2.imread('../data/img1.jpg', 0)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Camera intrinsics — assumes principal point at image center, focal length ~0.9 * width
    h, w = img1.shape[:2]
    focal = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]], dtype=np.float64)

    print(f"--- Camera Intrinsics ---")
    print(f"Image size: {w}x{h}")
    print(f"Focal length: {focal:.1f} px  (reasonable range: {0.5*w:.0f}–{2*w:.0f})")
    print(f"Principal point: ({cx:.1f}, {cy:.1f})")
    if focal < 0.3 * w or focal > 3 * w:
        print("WARNING: focal length looks unreasonable — check K!")
    print()

    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Estimate Essential Matrix with RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=1.0)

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())

    # Recover relative rotation and translation
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    print(f"Keypoints image 1: {len(kp1)}")
    print(f"Keypoints image 2: {len(kp2)}")
    print(f"Raw matches: {len(raw_matches)}")
    print(f"Good matches after ratio test: {len(good_matches)}")
    print(f"Essential Matrix inliers: {inlier_count} / {len(good_matches)}")
    print(f"Rotation matrix R:\n{R}")
    print(f"Translation vector t:\n{t}")

    # --- Draw only Essential Matrix inlier matches ---
    inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]
    img_inlier_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_inlier_matches = cv2.resize(img_inlier_matches, (1200, 500))
    cv2.imshow('Essential Matrix Inlier Matches', img_inlier_matches)

    # --- Milestone 3: Triangulate 3D points ---
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points_4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Filter out points behind either camera (negative depth)
    valid = (points_4d[3] > 0) & (points_3d[:, 2] > 0)
    points_3d = points_3d[valid]

    pts1_in = pts1_in[valid]
    pts2_in = pts2_in[valid]

    print(f"\nTriangulated points: {len(points_3d)}  (after depth filter: {valid.sum()})")
    print(f"Depth range: {points_3d[:, 2].min():.2f} – {points_3d[:, 2].max():.2f}")

    # --- Reprojection error ---
    projected1, _ = cv2.projectPoints(
        points_3d,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        None
    )
    projected2, _ = cv2.projectPoints(
        points_3d,
        cv2.Rodrigues(R)[0],
        t,
        K,
        None
    )

    err1 = np.linalg.norm(pts1_in - projected1.reshape(-1, 2), axis=1)
    err2 = np.linalg.norm(pts2_in - projected2.reshape(-1, 2), axis=1)
    mean_err = (err1.mean() + err2.mean()) / 2

    print(f"\nReprojection error — cam1: {err1.mean():.2f} px  cam2: {err2.mean():.2f} px")
    print(f"Mean reprojection error: {mean_err:.2f} px", end="  ")
    if mean_err < 3.0:
        print("(GOOD)")
    elif mean_err < 8.0:
        print("(ACCEPTABLE — check K or baseline)")
    else:
        print("WARNING: large error — K may be wrong or scene is degenerate")

    # --- Plot 3D point cloud ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               s=2, c=points_3d[:, 2], cmap='plasma', alpha=0.7)

    # Draw camera 1 at origin
    ax.scatter(0, 0, 0, c='blue', s=60, marker='^', label='Cam 1')

    # Camera 2 centre: -R^T @ t
    cam2_center = (-R.T @ t).ravel()
    ax.scatter(*cam2_center, c='red', s=60, marker='^', label='Cam 2')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Triangulated 3D Points')
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.tight_layout()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
