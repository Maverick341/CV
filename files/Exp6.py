import cv2
import numpy as np 
import matplotlib.pyplot as plt

# Load images
img_left = cv2.imread("images/left.jpg", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("images/right.jpg", cv2.IMREAD_GRAYSCALE)

# Display both images
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.imshow(img_left, cmap='gray')
plt.title("Left Image")
plt.subplot(122)
plt.imshow(img_right, cmap='gray')
plt.title("Right Image")
plt.show()


# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_left, None)
kp2, des2 = sift.detectAndCompute(img_right, None)


# Match features between left and right images
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)


# Apply Lowe’s ratio test to keep only good matches
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

print("Number of matched feature points used:", len(pts1))


# Compute Fundamental Matrix using the 8-point algorithm
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
print("\nFundamental Matrix (F):\n", F)


# Validate the Epipolar Constraint (x2^T * F * x1 ≈ 0)
pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)

print("\nEpipolar Constraint Validation (x2^T * F * x1):")
for i in range(min(10, len(pts1))):  # check first 10 correspondences
    val = np.dot(pts2_h[i], np.dot(F, pts1_h[i]))
    print(f"Point {i+1}: {val:.6f}")
    