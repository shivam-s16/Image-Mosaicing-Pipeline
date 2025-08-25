import numpy as np
import cv2

def bilinear_interpolation(img, x, y):
    """
    Bilinear interpolation at non-integer coordinates
    """
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, img.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, img.shape[0] - 1)
    
    # Ensure bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    
    # Weights
    wx = x - x0
    wy = y - y0
    
    # Interpolate
    if len(img.shape) == 3:  # Color image
        result = (1-wx)*(1-wy)*img[y0, x0] + wx*(1-wy)*img[y0, x1] + \
                 (1-wx)*wy*img[y1, x0] + wx*wy*img[y1, x1]
    else:  # Grayscale
        result = (1-wx)*(1-wy)*img[y0, x0] + wx*(1-wy)*img[y0, x1] + \
                 (1-wx)*wy*img[y1, x0] + wx*wy*img[y1, x1]
    
    return result

def warp_and_blend(img1, img2, H):
    """
    Warp img2 to img1's coordinate system and blend
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find corners of img2 in img1's coordinate system
    corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
    corners_img2_homo = np.column_stack([corners_img2, np.ones(4)])
    
    # Transform corners
    warped_corners_homo = (H @ corners_img2_homo.T).T
    warped_corners = warped_corners_homo[:, :2] / warped_corners_homo[:, 2:3]
    
    # Find bounding box of result
    corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    all_corners = np.vstack([warped_corners, corners_img1])
    
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    
    # Translation to positive coordinates
    translation = np.array([[1, 0, -min_x],
                           [0, 1, -min_y],
                           [0, 0, 1]], dtype=np.float32)
    
    # Final canvas size
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    
    # Create result canvas
    if len(img1.shape) == 3:
        result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    else:
        result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Place img1 on canvas
    img1_offset_y = -min_y
    img1_offset_x = -min_x
    result[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = img1
    
    # Warp img2 onto canvas
    H_final = translation @ H
    H_inv = np.linalg.inv(H_final)
    
    for y in range(canvas_h):
        for x in range(canvas_w):
            # Map canvas coordinates back to img2
            pt_homo = H_inv @ np.array([x, y, 1])
            pt = pt_homo[:2] / pt_homo[2]
            
            # Check if point is within img2 bounds
            if 0 <= pt[0] < w2 and 0 <= pt[1] < h2:
                # Interpolate and set pixel
                pixel_value = bilinear_interpolation(img2, pt[0], pt[1])
                if len(img1.shape) == 3:
                    result[y, x] = pixel_value.astype(np.uint8)
                else:
                    result[y, x] = int(pixel_value)
    
    return result
