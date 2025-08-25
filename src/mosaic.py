import cv2
import argparse
import numpy as np
import os
import glob

from harris import harris_corners, draw_corners
from descriptor import extract_patch_descriptors
from matching import match_descriptors, filter_matches_ratio_test
from homography import ransac_homography
from warp_blend import warp_and_blend
from sift_surf import sift_surf_mosaic, draw_matches_preview

def find_images():
    """
    Find images in data directory with various extensions
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(f"../data/{ext}"))
    
    # Sort to ensure consistent ordering
    image_files.sort()
    
    if len(image_files) < 2:
        return None, None
    print(image_files[0])
    print((image_files[0][:-5] + "2.jpg" ))

    img1 = cv2.imread(image_files[0])
    img2 = cv2.imread(image_files[0][:-5] + "2.jpg" )
    
    print(f"Loaded: {image_files[0]} and {image_files[1]}")
    
    return img1, img2

def create_results_dir():
    """Create results directory if it doesn't exist"""
    if not os.path.exists("../results"):
        os.makedirs("../results")

def run_harris_pipeline(img1, img2):
    """
    Run the from-scratch Harris-based pipeline
    """
    print("Running Harris corner detection pipeline...")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect Harris corners
    print("Detecting Harris corners...")
    kp1, response1 = harris_corners(gray1, threshold=0.01)
    kp2, response2 = harris_corners(gray2, threshold=0.01)
    
    print(f"Found {len(kp1)} corners in image 1")
    print(f"Found {len(kp2)} corners in image 2")
    
    # Extract descriptors
    print("Extracting patch descriptors...")
    kp1, desc1 = extract_patch_descriptors(gray1, kp1)
    kp2, desc2 = extract_patch_descriptors(gray2, kp2)
    
    print(f"Extracted {len(desc1)} descriptors from image 1")
    print(f"Extracted {len(desc2)} descriptors from image 2")
    
    # Match descriptors
    print("Matching descriptors...")
    matches = match_descriptors(desc1, desc2, threshold=0.8)
    
    print(f"Found {len(matches)} initial matches")
    
    if len(matches) < 4:
        print("Not enough matches found. Trying with lower threshold...")
        matches = match_descriptors(desc1, desc2, threshold=1.2)
        print(f"Found {len(matches)} matches with relaxed threshold")
        
        if len(matches) < 4:
            print("Still not enough matches. Cannot proceed with Harris pipeline.")
            return None
    
    # Create match visualization
    print("Creating match visualization...")
    matches_img = draw_matches_preview(img1, img2, kp1, kp2, matches)
    cv2.imwrite("../results/harris_matches_preview.jpg", matches_img)
    
    # RANSAC homography estimation
    print("Estimating homography with RANSAC...")
    H, inliers = ransac_homography(kp1, kp2, matches, num_iterations=2000)
    
    if H is None:
        print("RANSAC failed to find a good homography")
        return None
    
    print(f"RANSAC found {len(inliers)} inliers")
    
    # Warp and blend
    print("Warping and blending images...")
    result = warp_and_blend(img1, img2, H)
    
    return result

def run_sift_surf_pipeline(img1, img2, use_surf=False):
    """
    Run SIFT or SURF pipeline
    """
    method = "SURF" if use_surf else "SIFT"
    print(f"Running {method} pipeline...")
    
    result, method_name = sift_surf_mosaic(img1, img2)
    
    return result, method_name

def main():
    parser = argparse.ArgumentParser(description="Image Mosaicing Pipeline")
    parser.add_argument("--mode", type=str, choices=["harris", "sift"], 
                       default="harris", help="Feature detection method")
    parser.add_argument("--output", type=str, help="Output filename (optional)")
    
    args = parser.parse_args()
    
    # Create results directory
    create_results_dir()
    
    # Load images
    print("Loading images...")
    img1, img2 = find_images()
    
    if img1 is None or img2 is None:
        print("Error: Could not load two images from ../data/ directory")
        print("Please place at least 2 images (jpg, jpeg, or png) in the data folder")
        return
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Run selected pipeline
    if args.mode == "harris":
        result = run_harris_pipeline(img1, img2)
        if result is not None:
            output_path = args.output or "../results/mosaic_harris.jpg"
            cv2.imwrite(output_path, result)
            print(f"Harris mosaic saved to {output_path}")
        else:
            print("Harris pipeline failed")
            
    elif args.mode in ["sift"]:
        
        result, method_name = run_sift_surf_pipeline(img1, img2)
        
        if result is not None:
            output_path = args.output or f"../results/mosaic_{method_name.lower()}.jpg"
            cv2.imwrite(output_path, result)
            print(f"{method_name} mosaic saved to {output_path}")
        else:
            print(f"{args.mode.upper()} pipeline failed")
    
    print("Done!")

if __name__ == "__main__":
    main()
