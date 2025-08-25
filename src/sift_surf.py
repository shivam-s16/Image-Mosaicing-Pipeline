import cv2
import numpy as np

def sift_surf_mosaic(img1, img2):
    """
    Create mosaic using SIFT or SURF features
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute features
    
    detector = cv2.SIFT_create()
    method_name = "SIFT"
    
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None:
        print(f"Could not compute {method_name} descriptors")
        return img1, method_name
    
    # Match features
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        print(f"Not enough good {method_name} matches: {len(good_matches)}")
        return img1, method_name
    
    # Extract point correspondences
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"Could not find homography with {method_name}")
        return img1, method_name
    
    # Calculate output size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Corners of img1
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    # Transform corners of img1
    pts1_transformed = cv2.perspectiveTransform(pts1, H)
    
    # Corners of img2
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # All corners
    all_pts = np.concatenate((pts1_transformed, pts2), axis=0)
    
    # Find bounding rectangle
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel())
    
    # Translation matrix
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], 
                             [0, 1, translation_dist[1]], 
                             [0, 0, 1]])
    
    # Warp img1
    output_img = cv2.warpPerspective(img1, H_translation.dot(H), 
                                    (x_max - x_min, y_max - y_min))
    
    # Place img2
    output_img[translation_dist[1]:h2 + translation_dist[1], 
               translation_dist[0]:w2 + translation_dist[0]] = img2
    
    return output_img, method_name

def draw_matches_preview(img1, img2, kp1, kp2, matches):
    """
    Draw matches between two images
    """
    # Convert keypoints to cv2.KeyPoint format if needed
    if isinstance(kp1, np.ndarray):
        kp1 = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 5) for kp in kp1]
    if isinstance(kp2, np.ndarray):
        kp2 = [cv2.KeyPoint(float(kp[1]), float(kp[0]), 5) for kp in kp2]
    
    # Convert matches to cv2.DMatch format
    match_objects = [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=0) 
                    for m in matches]
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, match_objects, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches
