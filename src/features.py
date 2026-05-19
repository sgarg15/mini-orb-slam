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
    # Use Brute-Force matcher with Hamming distance for ORB descriptors
    # Used Brute-Force matcher because it is simple and effective for small to medium-sized descriptor sets, and Hamming distance is suitable for binary descriptors like ORB.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Perform k-nearest neighbor matching (k=2)
    raw = bf.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test to filter out ambiguous matches. A match is considered good if the distance of the best match is less than a specified ratio (e.g., 0.75) of the distance of the second-best match.
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    # Sort matches by distance (best matches first)
    good.sort(key=lambda x: x.distance)
    return raw, good

# 
def matched_points(kp1, kp2, matches):
    # Extract matched keypoint coordinates from the keypoint lists based on the matches. The queryIdx (index into kp1 which is the query image) and trainIdx (index into kp2 which is the train image) 
    # attributes of the match objects are used to index into the keypoint lists to retrieve the (x, y) coordinates of the matched keypoints.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
