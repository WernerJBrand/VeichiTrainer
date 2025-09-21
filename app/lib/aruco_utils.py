import cv2
import numpy as np

def detect_aruco_scale(image_bgr, marker_size_mm=60.0):
    """
    Returns (px_per_mm, corners, ids) or (None, [], [])
    px_per_mm is computed from detected 4x4 markers' average pixel side length.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dict_, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, [], []
    px_per_mm_vals = []
    for c in corners:
        pts = c[0]
        sides = [
            np.linalg.norm(pts[0]-pts[1]),
            np.linalg.norm(pts[1]-pts[2]),
            np.linalg.norm(pts[2]-pts[3]),
            np.linalg.norm(pts[3]-pts[0]),
        ]
        mean_side_px = float(np.mean(sides))
        px_per_mm_vals.append(mean_side_px / float(marker_size_mm))
    return float(np.mean(px_per_mm_vals)), corners, ids.flatten()
