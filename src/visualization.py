import cv2
import numpy as np
import matplotlib.pyplot as plt


def _robust_points(points, low=1.0, high=99.0):
    if points is None or len(points) == 0:
        return points
    lo = np.percentile(points, low, axis=0)
    hi = np.percentile(points, high, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def _set_equal_3d_axes(ax, points):
    if points is None or len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax_top = fig.add_subplot(122)

    centers = np.array([(-R.T @ t).ravel() for R, t in trajectory])
    map_plot = _robust_points(map_points)

    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            'b-o', markersize=3, linewidth=1.2, label='Trajectory')
    ax.scatter(*centers[0], c='green', s=80, marker='^', zorder=5, label='Start')
    ax.scatter(*centers[-1], c='red', s=80, marker='^', zorder=5, label='End')

    if map_plot is not None and len(map_plot) > 0:
        ax.scatter(map_plot[:, 0], map_plot[:, 1], map_plot[:, 2],
                   s=1, c=map_plot[:, 2], cmap='plasma', alpha=0.4, label='Map points')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory  ({len(trajectory)} frames,  {len(map_points) if map_points is not None else 0} map points)')
    plot_points = centers if map_plot is None or len(map_plot) == 0 else np.vstack([centers, map_plot])
    _set_equal_3d_axes(ax, plot_points)
    ax.legend()

    if map_plot is not None and len(map_plot) > 0:
        ax_top.scatter(map_plot[:, 0], map_plot[:, 2], s=1, alpha=0.25, label='Map points')
    ax_top.plot(centers[:, 0], centers[:, 2], 'b-o', markersize=3, linewidth=1.2, label='Trajectory')
    ax_top.scatter(centers[0, 0], centers[0, 2], c='green', s=60, marker='^', label='Start')
    ax_top.scatter(centers[-1, 0], centers[-1, 2], c='red', s=60, marker='^', label='End')
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Z forward')
    ax_top.set_title('Top-down X/Z view')
    ax_top.axis('equal')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    plt.tight_layout()
    plt.show()
