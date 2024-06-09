import cv2
import numpy as np
import os

# 如果需要额外的图像预处理（如模糊或锐化），可以在这里定义预处理函数

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def apply_sharpening(image, kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])):
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def orb_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def align_images(target_image, ref_image):
    kp1, des1 = orb_keypoints_and_descriptors(target_image)
    kp2, des2 = orb_keypoints_and_descriptors(ref_image)
    matches = match_keypoints(des1, des2)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(target_image, matrix, (ref_image.shape[1], ref_image.shape[0]))
        return aligned_image, True
    else:
        print(f"Not enough matches are found - {len(matches)}/{10}. Cannot align.")
        return target_image, False

folder_path = '2/3'
image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Assuming that the first image is the reference image
ref_image_path = os.path.join(folder_path, image_files[0])
ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the reference image, if necessary
ref_image = apply_gaussian_blur(ref_image)
ref_image = apply_sharpening(ref_image)

# Align all other images to the reference image
for image_file in image_files[1:]:
    target_image_path = os.path.join(folder_path, image_file)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the target image, if necessary
    # target_image = apply_gaussian_blur(target_image)
    # target_image = apply_sharpening(target_image)

    aligned_image, aligned = align_images(target_image, ref_image)

    if aligned:
        # Save or display the aligned image
        aligned_image_path = os.path.join(folder_path, 'aligned_' + image_file)
        cv2.imwrite(aligned_image_path, aligned_image)
        print(f"Aligned image saved: {aligned_image_path}")
    else:
        print(f"Failed to align {target_image_path}")
