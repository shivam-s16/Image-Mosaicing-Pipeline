import numpy as np

def compute_homography_dlt(p1, p2):
    """
    Compute homography using Direct Linear Transform (DLT)
    """
    if len(p1) != len(p2) or len(p1) < 4:
        raise ValueError("Need at least 4 point correspondences")
    
    # Build matrix A for Ah = 0
    A = []
    for i in range(len(p1)):
        x1, y1 = p1[i][0], p1[i][1]
        x2, y2 = p2[i][0], p2[i][1]
        
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    
    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)
    
    # Normalize
    if H[2, 2] != 0:
        H = H / H[2, 2]
    
    return H

def ransac_homography(kp1, kp2, matches, num_iterations=1000, threshold=3.0):
    """
    Robust homography estimation using RANSAC
    """
    if len(matches) < 4:
        print(f"Not enough matches for RANSAC: {len(matches)}")
        return None, []
    
    best_H = None
    best_inliers = []
    matches_array = np.array(matches)
    
    for _ in range(num_iterations):
        # Randomly sample 4 matches
        sample_indices = np.random.choice(len(matches), 4, replace=False)
        
        # Get corresponding points
        sample_matches = matches_array[sample_indices]
        p1 = np.array([kp1[m[0]][::-1] for m in sample_matches])  # (x, y)
        p2 = np.array([kp2[m[1]][::-1] for m in sample_matches])  # (x, y)
        
        try:
            # Compute homography
            H = compute_homography_dlt(p1, p2)
            
            # Count inliers
            inliers = []
            for i, match in enumerate(matches):
                # Transform point from image 1
                pt1_homo = np.array([kp1[match[0]][1], kp1[match[0]][0], 1])  # (x, y, 1)
                pt2_pred_homo = H @ pt1_homo
                
                if pt2_pred_homo[2] != 0:
                    pt2_pred = pt2_pred_homo[:2] / pt2_pred_homo[2]
                    pt2_actual = np.array([kp2[match[1]][1], kp2[match[1]][0]])  # (x, y)
                    
                    # Compute reprojection error
                    error = np.linalg.norm(pt2_pred - pt2_actual)
                    
                    if error < threshold:
                        inliers.append(match)
            
            # Keep if better than current best
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H
                
        except np.linalg.LinAlgError:
            continue
    
    print(f"RANSAC found {len(best_inliers)} inliers out of {len(matches)} matches")
    return best_H, best_inliers
