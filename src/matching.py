import numpy as np

def match_descriptors(desc1, desc2, threshold=0.8):
    """
    Match descriptors using Euclidean distance
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    matches = []
    
    for i, d1 in enumerate(desc1):
        # Compute distances to all descriptors in second image
        distances = np.linalg.norm(desc2 - d1, axis=1)
        
        # Find closest match
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Accept match if below threshold
        if min_distance < threshold:
            matches.append((i, min_idx))
    
    return matches

def filter_matches_ratio_test(desc1, desc2, ratio=0.75):
    """
    Filter matches using Lowe's ratio test
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    matches = []
    
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        
        # Find two closest matches
        sorted_indices = np.argsort(distances)
        
        if len(sorted_indices) >= 2:
            closest_dist = distances[sorted_indices[0]]
            second_closest_dist = distances[sorted_indices[1]]
            
            # Ratio test
            if closest_dist < ratio * second_closest_dist:
                matches.append((i, sorted_indices[0]))
    
    return matches
