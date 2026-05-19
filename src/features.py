import cv2
import numpy as np


def detect(img, nfeatures=2000, scale_factor=1.2, nlevels=8):
    orb = cv2.ORB_create(nfeatures=nfeatures,
                         scaleFactor=scale_factor,
                         nlevels=nlevels)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def match(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    good.sort(key=lambda x: x.distance)
    return raw, good


def matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
