import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_inlier_matches(img1, kp1, img2, kp2, matches, inlier_mask, max_draw=50):
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    canvas = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:max_draw],
                             None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    canvas = cv2.resize(canvas, (1200, 500))
    cv2.imshow('Essential Matrix Inlier Matches', canvas)


def plot_3d(points_3d, R, t):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               s=2, c=points_3d[:, 2], cmap='plasma', alpha=0.7)

    ax.scatter(0, 0, 0, c='blue', s=60, marker='^', label='Cam 1')

    cam2_center = (-R.T @ t).ravel()
    ax.scatter(*cam2_center, c='red', s=60, marker='^', label='Cam 2')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Triangulated 3D Points')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_trajectory(trajectory, map_points=None):
    """Plot the camera trajectory and sparse map in 3D.

    trajectory: list of (R, t) where X_cam = R @ X_world + t.
    map_points: optional Nx3 array of 3D map points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Camera center in world: C = -R.T @ t
    centers = np.array([(-R.T @ t).ravel() for R, t in trajectory])

    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            'b-o', markersize=3, linewidth=1.2, label='Trajectory')
    ax.scatter(*centers[0], c='green', s=80, marker='^', zorder=5, label='Start')
    ax.scatter(*centers[-1], c='red', s=80, marker='^', zorder=5, label='End')

    if map_points is not None and len(map_points) > 0:
        ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2],
                   s=1, c=map_points[:, 2], cmap='plasma', alpha=0.4, label='Map points')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory  ({len(trajectory)} frames,  {len(map_points) if map_points is not None else 0} map points)')
    ax.legend()
    plt.tight_layout()
    plt.show()
