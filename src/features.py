import cv2
import numpy as np

# ORB feature detection and descriptor computation
def detect(img, nfeatures=2000, scale_factor=1.2, nlevels=8):
    # Create ORB Detector with specified parameters taken from the paper
    orb = cv2.ORB_create(nfeatures=nfeatures,
                         scaleFactor=scale_factor,
                         nlevels=nlevels)
    # Detect keypoints and compute descriptors
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


# Match descriptors using Brute-Force matcher with Hamming distance and apply Lowe's ratio test
def match(des1, des2, ratio=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) < 2:
        return [], []
    # Use Brute-Force matcher with Hamming distance for ORB descriptors
    # Used Brute-Force matcher because it is simple and effective for small to medium-sized descriptor sets, and Hamming distance is suitable for binary descriptors like ORB.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Perform k-nearest neighbor matching (k=2)
    raw = bf.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test to filter out ambiguous matches. A match is considered good if the distance of the best match is less than a specified ratio (e.g., 0.75) of the distance of the second-best match.
    candidates = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            candidates.append(m)

    candidates.sort(key=lambda x: x.distance)
    good = []
    used_query = set()
    used_train = set()
    for m in candidates:
        if m.queryIdx not in used_query and m.trainIdx not in used_train:
            good.append(m)
            used_query.add(m.queryIdx)
            used_train.add(m.trainIdx)
    return raw, good


def matched_points(kp1, kp2, matches):
    # Extract matched keypoint coordinates from the keypoint lists based on the matches. The queryIdx (index into kp1 which is the query image) and trainIdx (index into kp2 which is the train image)
    # attributes of the match objects are used to index into the keypoint lists to retrieve the (x, y) coordinates of the matched keypoints.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def match_to_map(des_frame, map_descriptors, ratio=0.75):
    """Match new-frame descriptors against the map's stored descriptors.

    Returns (frame_indices, map_indices): parallel integer arrays of matched indices,
    where frame_indices[i] indexes into the new frame's keypoints and
    map_indices[i] indexes into the map points array.
    """
    if des_frame is None or map_descriptors is None or len(des_frame) == 0 or len(map_descriptors) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des_frame, map_descriptors, k=2)
    candidates = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            candidates.append(m)

    candidates.sort(key=lambda m: m.distance)
    frame_idx, map_idx = [], []
    used_frame = set()
    used_map = set()
    for m in candidates:
        if m.queryIdx in used_frame or m.trainIdx in used_map:
            continue
        frame_idx.append(m.queryIdx)
        map_idx.append(m.trainIdx)
        used_frame.add(m.queryIdx)
        used_map.add(m.trainIdx)

    return np.array(frame_idx, dtype=int), np.array(map_idx, dtype=int)
