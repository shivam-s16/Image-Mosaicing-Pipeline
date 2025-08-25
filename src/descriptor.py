import numpy as np

def extract_patch_descriptors(img_gray, keypoints, patch_size=8):
    """
    Extract normalized patch descriptors around keypoints
    """
    descriptors = []
    valid_keypoints = []
    half = patch_size // 2
    H, W = img_gray.shape
    
    for y, x in keypoints:
        # Check bounds
        if (y - half >= 0 and y + half < H and 
            x - half >= 0 and x + half < W):
            
            # Extract patch
            patch = img_gray[y-half:y+half, x-half:x+half].astype(np.float32)
            
            # Normalize patch (zero mean, unit variance)
            mean = np.mean(patch)
            std = np.std(patch)
            
            if std > 1e-6:  # Avoid division by zero
                norm_patch = (patch - mean) / std
            else:
                norm_patch = patch - mean
            
            descriptors.append(norm_patch.flatten())
            valid_keypoints.append((y, x))
    
    return np.array(valid_keypoints), np.array(descriptors)
