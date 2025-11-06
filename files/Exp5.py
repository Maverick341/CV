import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 1. Define chessboard dimensions (inner corners)
chessboard_size = (9, 6)
square_size = 1.0

# 2. Prepare object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

# 3. Load calibration images
images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if 'img_shape' not in locals():
        img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Corners in {fname}")
        plt.axis('off')
        plt.show()

# 4. Calibrate camera
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    print("✅ Camera Matrix:\n", mtx)
    print("\n✅ Distortion Coefficients:\n", dist)
else:
    print("❌ No valid chessboard corners were detected in any image.")