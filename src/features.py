import cv2
import numpy as np

def detect(img, nfeatures=2000, scale_factor=1.2, nlevels=8):
    """
    ORB feature detection and descriptor computation
    
    Returns:
        kp: list of detected keypoints in the image, where each keypoint has attributes like pt (x, y coordinates), size, angle, etc.
        des: numpy array of shape (number of keypoints, descriptor dimension) containing the binary descriptors for each keypoint, which can
    """
    # Create ORB Detector with specified parameters taken from the paper
    orb = cv2.ORB_create(nfeatures=nfeatures,
                         scaleFactor=scale_factor,
                         nlevels=nlevels)
    # Detect keypoints and compute descriptors
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


# Match descriptors using Brute-Force matcher with Hamming distance and apply Lowe's ratio test
def match(des1, des2, ratio=0.75):
    """
    Match descriptors between two sets using Brute-Force matcher and Lowe's ratio test.
    Returns:
    raw: list of lists of DMatch objects from knnMatch (k=2)
    good: list of DMatch objects that passed Lowe's ratio test and are considered good matches
    """
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
    """
    Given two sets of keypoints and a list of matches between them, return the corresponding matched points as numpy arrays. This is used for estimating the Essential Matrix or performing PnP.
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def match_to_map(des_frame, map_descriptors, ratio=0.75):
    """
    Given the descriptors of the current frame and the descriptors of the map points, find matches between them using Brute-Force matching and Lowe's ratio test. This is used for tracking the current frame against the map with PnP.

    Returns:
        frame_idx: numpy array of indices of the keypoints in the current frame that have good matches to the map descriptors
        map_idx: numpy array of indices of the map descriptors that have good matches to the current frame descriptors
    """
    if des_frame is None or map_descriptors is None or len(des_frame) == 0 or len(map_descriptors) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    #Use Brute-Force matcher with Hamming distance for ORB descriptors and get raw matches with k-nearest neighbor matching (k=2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des_frame, map_descriptors, k=2)
    
    # Apply Lowe's ratio test to filter out ambiguous matches and ensure that each descriptor in the current frame matches with at most one descriptor in the map and vice versa
    candidates = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            candidates.append(m)

    # Sort the candidate matches by distance and enforce a one-to-one matching constraint to ensure that each descriptor in the current frame matches with at most one descriptor in the map and vice versa. 
    candidates.sort(key=lambda m: m.distance)
    frame_idx, map_idx = [], []
    used_frame = set()
    used_map = set()

    # This is done by iterating through the sorted matches and keeping track of which descriptors have already been matched.
    for m in candidates:
        if m.queryIdx in used_frame or m.trainIdx in used_map:
            continue
        frame_idx.append(m.queryIdx)
        map_idx.append(m.trainIdx)
        used_frame.add(m.queryIdx)
        used_map.add(m.trainIdx)

    # In the end return the indices of the matched descriptors in the current frame and the map as numpy arrays. These indices can then be used to retrieve the corresponding keypoint coordinates for PnP tracking.
    return np.array(frame_idx, dtype=int), np.array(map_idx, dtype=int)
