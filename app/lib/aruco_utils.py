import cv2
import numpy as np

def _detect(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dict_, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    return corners, (ids.flatten() if ids is not None else None)

def detect_aruco_scale(image_bgr, marker_size_mm=60.0):
    """
    Compute px/mm from detected ArUco markers on the plane.
    Returns (px_per_mm, corners, ids) or (None, [], [])
    """
    corners, ids = _detect(image_bgr)
    if ids is None or len(corners) == 0:
        return None, [], []
    px_per_mm_vals = []
    for c in corners:
        pts = c[0]
        sides = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        mean_side_px = float(np.mean(sides))
        px_per_mm_vals.append(mean_side_px / float(marker_size_mm))
    return float(np.mean(px_per_mm_vals)), corners, ids

def _order_centers_tl_tr_br_bl(corners):
    """Order 4 marker centers into TL, TR, BR, BL by position."""
    centers = np.array([c[0].mean(axis=0) for c in corners], dtype=np.float32)  # (N,2)
    # sort by y (top two, bottom two)
    idx_y = np.argsort(centers[:, 1])
    top = centers[idx_y[:2]]
    bottom = centers[idx_y[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def rectify_topdown_with_aruco(image_bgr, marker_size_mm=60.0):
    """
    Use 4 ArUco markers placed near the bench corners to warp to a fronto-parallel view.
    Returns (warped_bgr, H, px_per_mm) or (None, None, None) if not enough tags.
    """
    corners, ids = _detect(image_bgr)
    if ids is None or len(corners) < 4:
        return None, None, None

    # take the best 4 (first 4 detected is fine here)
    four = corners[:4]
    src = _order_centers_tl_tr_br_bl(four)

    # choose output size from max opposite-side lengths
    w = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))
    h = max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))
    w = int(np.ceil(w)); h = int(np.ceil(h))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, H, (w, h))

    # recompute px/mm on the rectified plane for consistent scaling
    px_per_mm, _, _ = detect_aruco_scale(warped, marker_size_mm)
    return warped, H, px_per_mm
