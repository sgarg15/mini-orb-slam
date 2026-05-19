import numpy as np


def build_K(img_shape):
    h, w = img_shape[:2]
    focal = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0,     0,  1]], dtype=np.float64)
    return K


def print_K(K, img_shape):
    h, w = img_shape[:2]
    focal = K[0, 0]
    print("--- Camera Intrinsics ---")
    print(f"Image size: {w}x{h}")
    print(f"Focal length: {focal:.1f} px  (reasonable range: {0.5*w:.0f}–{2*w:.0f})")
    print(f"Principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})")
    if focal < 0.3 * w or focal > 3 * w:
        print("WARNING: focal length looks unreasonable — check K!")
    print()
