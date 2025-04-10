import pyrealsense2 as rs
import cv2
import numpy as np
import time

# === Load Calibration ===
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# === RealSense Setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === ArUco Setup ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# === Helper functions ===
def control_pan_tilt(center_x, center_y):
    frame_center_x = 320
    frame_center_y = 240
    error_x = center_x - frame_center_x
    error_y = center_y - frame_center_y

    # Pan control
    if abs(error_x) > 20:
        if error_x > 0:
            pan_right()
        else:
            pan_left()

    # Tilt control (optional)
    if abs(error_y) > 20:
        if error_y > 0:
            tilt_down()
        else:
            tilt_up()

def pan_left():
    print("Pan left")

def pan_right():
    print("Pan right")

def tilt_up():
    print("Tilt up")

def tilt_down():
    print("Tilt down")

def move_forward():
    print("Move forward")
    time.sleep(1)

def turn_left():
    print("Turn left")
    time.sleep(0.5)

def turn_right():
    print("Turn right")
    time.sleep(0.5)

def stop_robot():
    print("Stop robot")

def pass_marker(marker_id):
    print(f"Passing marker {marker_id}")
    if marker_id % 2 == 0:
        # Even marker: pass on right
        move_forward()
        turn_right()
    else:
        # Odd marker: pass on left
        move_forward()
        turn_left()

# === Main Loop ===
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.15, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                id = ids[i][0]
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(img, corners)
                cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Marker center
                center_x = np.mean(corners[i][0][:, 0])
                center_y = np.mean(corners[i][0][:, 1])

                # Center the marker
                control_pan_tilt(center_x, center_y)

                # Distance to marker
                z_distance = tvec[2]

                # Move toward marker if it's far enough
                if z_distance > 0.5:  # Marker more than 0.5 meters away
                    move_forward()

                # If close enough, pass it
                elif z_distance <= 0.5:
                    pass_marker(id)
                    stop_robot()
                    time.sleep(1)  # Small pause after passing

        cv2.imshow('RealSense', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()