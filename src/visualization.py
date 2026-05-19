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
