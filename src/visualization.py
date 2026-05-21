import cv2
import numpy as np
import matplotlib.pyplot as plt


def _camera_centers(trajectory):
    return np.array([(-R.T @ t).ravel() for R, t in trajectory])


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


def plot_3d(points_3d, R, t, save_path=None):
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
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.ioff()
    plt.show()


def _draw_keyframe_strip(fig, grid_slot, keyframes, max_thumbs):
    recent = keyframes[-max_thumbs:]
    subgrid = grid_slot.subgridspec(1, len(recent), wspace=0.08)
    for i, kf in enumerate(recent):
        ax = fig.add_subplot(subgrid[0, i])
        ax.imshow(kf["image"], cmap="gray")
        ax.set_title(f'F{kf["frame"]} {kf["mode"]}', fontsize=9)
        ax.axis("off")


class LiveSlamViewer:
    """Interactive matplotlib dashboard for the running SLAM state."""

    def __init__(self, max_keyframe_thumbs=8):
        self.max_keyframe_thumbs = max_keyframe_thumbs
        self.closed = False
        plt.ion()
        self.fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.grid = self.fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0],
                                          width_ratios=[1.0, 1.0, 1.25],
                                          hspace=0.18, wspace=0.15)
        self.ax_current = self.fig.add_subplot(self.grid[0, 0])
        self.ax_keyframe = self.fig.add_subplot(self.grid[0, 1])
        self.ax_3d = self.fig.add_subplot(self.grid[0, 2], projection="3d")
        self.ax_top = self.fig.add_subplot(self.grid[1, 0])
        self.ax_status = self.fig.add_subplot(self.grid[1, 1])
        self.strip_slot = self.grid[1, 2]
        self.strip_axes = []
        plt.show(block=False)

    def _on_close(self, _event):
        self.closed = True

    def is_open(self):
        return not self.closed and plt.fignum_exists(self.fig.number)

    def update(self, trajectory, map_points, current_image, keyframes, status, pause=0.001):
        if not self.is_open():
            self.closed = True
            return False

        centers = _camera_centers(trajectory)
        map_plot = _robust_points(map_points)
        keyframe_centers = self._keyframe_centers(keyframes, centers)

        self._draw_images(current_image, keyframes)
        self._draw_trajectory_3d(centers, map_points, map_plot, keyframe_centers)
        self._draw_top_down(centers, map_plot, keyframe_centers)
        self._draw_status(status)
        self._draw_live_keyframe_strip(keyframes)

        self.fig.canvas.draw_idle()
        plt.pause(pause)
        return self.is_open()

    def _keyframe_centers(self, keyframes, centers):
        if not keyframes:
            return None
        indices = [kf["trajectory_index"] for kf in keyframes if kf["trajectory_index"] < len(centers)]
        if not indices:
            return None
        return centers[indices]

    def _draw_images(self, current_image, keyframes):
        self.ax_current.clear()
        self.ax_current.imshow(current_image, cmap="gray")
        self.ax_current.set_title("Current frame")
        self.ax_current.axis("off")

        self.ax_keyframe.clear()
        latest = keyframes[-1] if keyframes else None
        if latest is not None:
            self.ax_keyframe.imshow(latest["image"], cmap="gray")
            self.ax_keyframe.set_title(f'Latest keyframe  F{latest["frame"]} {latest["mode"]}')
        else:
            self.ax_keyframe.set_title("Latest keyframe")
        self.ax_keyframe.axis("off")

    def _draw_trajectory_3d(self, centers, map_points, map_plot, keyframe_centers):
        self.ax_3d.clear()
        self.ax_3d.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                        "b-o", markersize=2.5, linewidth=1.1, label="Trajectory")
        self.ax_3d.scatter(*centers[0], c="green", s=55, marker="^", label="Start")
        self.ax_3d.scatter(*centers[-1], c="red", s=55, marker="^", label="Current")
        if keyframe_centers is not None:
            self.ax_3d.scatter(keyframe_centers[:, 0], keyframe_centers[:, 1], keyframe_centers[:, 2],
                               c="orange", s=24, marker="s", label="Keyframes")
        if map_plot is not None and len(map_plot) > 0:
            self.ax_3d.scatter(map_plot[:, 0], map_plot[:, 1], map_plot[:, 2],
                               s=1, c=map_plot[:, 2], cmap="plasma", alpha=0.35,
                               label="Map points")

        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_title(f'3D trajectory ({len(centers)} poses, {len(map_points)} map points)')
        plot_points = centers if map_plot is None or len(map_plot) == 0 else np.vstack([centers, map_plot])
        _set_equal_3d_axes(self.ax_3d, plot_points)
        self.ax_3d.legend(loc="upper right", fontsize=8)

    def _draw_top_down(self, centers, map_plot, keyframe_centers):
        self.ax_top.clear()
        if map_plot is not None and len(map_plot) > 0:
            self.ax_top.scatter(map_plot[:, 0], map_plot[:, 2], s=1, alpha=0.20,
                                label="Map points")
        self.ax_top.plot(centers[:, 0], centers[:, 2], "b-o",
                         markersize=2.5, linewidth=1.1, label="Trajectory")
        self.ax_top.scatter(centers[0, 0], centers[0, 2], c="green", s=45, marker="^",
                            label="Start")
        self.ax_top.scatter(centers[-1, 0], centers[-1, 2], c="red", s=45, marker="^",
                            label="Current")
        if keyframe_centers is not None:
            self.ax_top.scatter(keyframe_centers[:, 0], keyframe_centers[:, 2],
                                c="orange", s=22, marker="s", label="Keyframes")
        self.ax_top.set_xlabel("X")
        self.ax_top.set_ylabel("Z forward")
        self.ax_top.set_title("Top-down X/Z")
        self.ax_top.axis("equal")
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.legend(loc="best", fontsize=8)

    def _draw_status(self, status):
        self.ax_status.clear()
        self.ax_status.axis("off")
        lines = [
            f'Frame: {status.get("frame", "-")}',
            f'Mode: {status.get("mode", "-")}',
            f'Inliers: {status.get("inliers", "-")}',
            f'Map points: {status.get("map_points", "-")}',
            f'Keyframes: {status.get("keyframes", "-")}',
        ]
        if status.get("message"):
            lines.append(status["message"])
        self.ax_status.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                            fontsize=11, family="monospace")
        self.ax_status.set_title("Tracking status")

    def _draw_live_keyframe_strip(self, keyframes):
        for ax in self.strip_axes:
            ax.remove()
        self.strip_axes = []

        if not keyframes:
            ax = self.fig.add_subplot(self.strip_slot)
            ax.set_title("Recent keyframes")
            ax.axis("off")
            self.strip_axes.append(ax)
            return

        recent = keyframes[-self.max_keyframe_thumbs:]
        subgrid = self.strip_slot.subgridspec(1, len(recent), wspace=0.08)
        for i, kf in enumerate(recent):
            ax = self.fig.add_subplot(subgrid[0, i])
            ax.imshow(kf["image"], cmap="gray")
            ax.set_title(f'F{kf["frame"]} {kf["mode"]}', fontsize=8)
            ax.axis("off")
            self.strip_axes.append(ax)

    def close(self):
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def plot_trajectory(trajectory, map_points=None, keyframes=None, max_keyframe_thumbs=8, save_path=None):
    """Plot the camera trajectory and sparse map in 3D.

    trajectory: list of (R, t) where X_cam = R @ X_world + t.
    map_points: optional Nx3 array of 3D map points.
    keyframes: optional list of dicts with frame/image/mode/trajectory_index.
    """
    has_keyframes = keyframes is not None and len(keyframes) > 0
    fig = plt.figure(figsize=(15, 9 if has_keyframes else 6), constrained_layout=True)
    if has_keyframes:
        grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.25], hspace=0.25, wspace=0.15)
        ax = fig.add_subplot(grid[0, 0], projection='3d')
        ax_top = fig.add_subplot(grid[0, 1])
    else:
        grid = fig.add_gridspec(1, 2, wspace=0.15)
        ax = fig.add_subplot(grid[0, 0], projection='3d')
        ax_top = fig.add_subplot(grid[0, 1])

    centers = np.array([(-R.T @ t).ravel() for R, t in trajectory])
    map_plot = _robust_points(map_points)
    keyframe_centers = None
    if has_keyframes:
        kf_indices = [kf["trajectory_index"] for kf in keyframes if kf["trajectory_index"] < len(centers)]
        if kf_indices:
            keyframe_centers = centers[kf_indices]

    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            'b-o', markersize=3, linewidth=1.2, label='Trajectory')
    ax.scatter(*centers[0], c='green', s=80, marker='^', zorder=5, label='Start')
    ax.scatter(*centers[-1], c='red', s=80, marker='^', zorder=5, label='End')
    if keyframe_centers is not None:
        ax.scatter(keyframe_centers[:, 0], keyframe_centers[:, 1], keyframe_centers[:, 2],
                   c='orange', s=28, marker='s', zorder=6, label='Keyframes')

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
    if keyframe_centers is not None:
        ax_top.scatter(keyframe_centers[:, 0], keyframe_centers[:, 2],
                       c='orange', s=28, marker='s', label='Keyframes')
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Z forward')
    ax_top.set_title('Top-down X/Z view')
    ax_top.axis('equal')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    if has_keyframes:
        _draw_keyframe_strip(fig, grid[1, :], keyframes, max_keyframe_thumbs)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.ioff()
    plt.show()
