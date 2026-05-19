import cv2

from camera import build_K, print_K
from features import detect, match, matched_points
from geometry import estimate_pose, triangulate, reprojection_error
from visualization import show_inlier_matches, plot_3d


def main():
    # Load images in grayscale
    img1 = cv2.imread('../data/img2.jpg', 0)
    img2 = cv2.imread('../data/img1.jpg', 0)

    # Detect keypoints and compute descriptors
    kp1, des1 = detect(img1)
    kp2, des2 = detect(img2)

    # Build camera intrinsics to be used for pose estimation and triangulation
    K = build_K(img1.shape)
    print_K(K, img1.shape)

    # Match descriptors and filter good matches
    raw_matches, good_matches = match(des1, des2)
    pts1, pts2 = matched_points(kp1, kp2, good_matches)

    # Estimate relative pose (R, t) using the essential matrix and RANSAC to find inliers
    R, t, inlier_mask = estimate_pose(pts1, pts2, K)

    print(f"Keypoints img1: {len(kp1)}  img2: {len(kp2)}")
    print(f"Raw matches: {len(raw_matches)}  good: {len(good_matches)}  inliers: {inlier_mask.sum()}")
    print(f"Rotation R:\n{R}")
    print(f"Translation t:\n{t}")

    # Helper function to visualize the inlier matches between the two images. This will show us which keypoint correspondences were considered inliers by the RANSAC algorithm during pose estimation.
    show_inlier_matches(img1, kp1, img2, kp2, good_matches, inlier_mask)

    # Get the inlier points using the inlier mask
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    # Triangulate the 3D points from the inlier points and the estimated pose giving us the 3D structure of the scene.
    points_3d, pts1_tri, pts2_tri = triangulate(K, R, t, pts1_in, pts2_in)

    print(f"\nTriangulated: {len(points_3d)} points")
    print(f"Depth range: {points_3d[:, 2].min():.2f} - {points_3d[:, 2].max():.2f}")

    # Compute the reprojection error of the triangulated 3D points back onto the original images. This will give us a quantitative measure of how well the estimated pose and triangulated points explain the observed keypoint correspondences. 
    # A low reprojection error indicates a good reconstruction, while a high error may suggest issues with the camera intrinsics, pose estimation, or triangulation.
    e1, e2 = reprojection_error(points_3d, pts1_tri, pts2_tri, K, R, t)
    mean_err = (e1 + e2) / 2
    print(f"\nReprojection error — cam1: {e1:.2f} px  cam2: {e2:.2f} px")
    verdict = "GOOD" if mean_err < 3.0 else ("ACCEPTABLE" if mean_err < 8.0 else "WARNING: check K")
    print(f"Mean: {mean_err:.2f} px  ({verdict})")

    # Helper function to visualize the triangulated 3D points in a 3D scatter plot. This will allow us to see the spatial distribution of the reconstructed points and the relative positions of the two cameras. Points are colored by depth to give a sense of the scene structure.
    plot_3d(points_3d, R, t)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
