import numpy as np
import cv2

def harris_corners(img, k=0.04, window_size=3, threshold=0.005):
    """
    Harris Corner Detection from scratch
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    gray = np.float32(gray)
    
    # Compute gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute products of derivatives
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian window
    Ixx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    
    # Compute Harris response
    det_M = Ixx * Iyy - Ixy * Ixy
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)
    
    # Normalize and threshold
    R_max = np.max(R)
    if R_max > 0:
        R_norm = R / R_max
    else:
        R_norm = R
    
    # Find corners above threshold
    corners = np.argwhere(R_norm > threshold)
    
    # Non-maximum suppression
    corners_nms = []
    for y, x in corners:
        if R_norm[y, x] == np.max(R_norm[max(0, y-2):min(R_norm.shape[0], y+3), 
                                          max(0, x-2):min(R_norm.shape[1], x+3)]):
            corners_nms.append([y, x])
    
    return np.array(corners_nms), R_norm

def draw_corners(img, corners):
    """Draw corners on image for visualization"""
    img_copy = img.copy()
    for y, x in corners:
        cv2.circle(img_copy, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img_copy
