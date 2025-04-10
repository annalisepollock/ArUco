import pyrealsense2 as rs
import cv2
import numpy as np
import glob

# === RealSense Setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === Calibration Parameters ===
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# === Capture Images for Calibration ===
try:
    print("Press 'c' to capture an image for calibration.")
    print("Press 'q' to quit and start calibration.")

    image_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == ord('c'):  # Press 'c' to capture
            image_filename = f'checkerboard_capture_{image_count}.png'
            cv2.imwrite(image_filename, color_image)
            print(f"Saved {image_filename}")
            image_count += 1
        if key == ord('q'):  # Press 'q' to quit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# === Perform Calibration ===
images = glob.glob('./checkerboard_capture_*.png')  # Load captured images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optionally, draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# === Compute Calibration Parameters ===
if objpoints and imgpoints:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        # Save the calibration results
        np.save('camera_matrix.npy', mtx)
        np.save('dist_coeffs.npy', dist)
        print("Calibration data saved.")
    else:
        print("Calibration failed.")
else:
    print("No valid checkerboard images were found.")