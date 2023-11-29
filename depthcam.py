# Below is the code that works using a regular MacBook camera.
# The depth estimates are not good and the intrinsics are not calibrated, so it's not great.

# This code calculates a normal vector to the plane of the hand at each timestep.
# The normal vector is visualized, but can also be outputted.
# Remove the visualization code when using this on a robot.

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import svd

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to convert keypoints to 3D coordinates using camera intrinsics.
def keypoints_to_3D(keypoints, intrinsics):
    points_3D = []
    for point in keypoints:
        # Convert from normalized coordinates to pixel coordinates.
        x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
        # Z-value is provided by MediaPipe.
        z = point.z

        # Reproject to 3D space using intrinsics.
        x_3D = (x - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y_3D = (y - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points_3D.append([x_3D, y_3D, z])
    return np.array(points_3D)

# Function to fit a plane using SVD and find its normal.
def fit_plane_svd(points_3D):
    # Mean centering the points
    center = np.mean(points_3D, axis=0)
    centered_points = points_3D - center

    # Applying SVD
    _, _, Vt = svd(centered_points)
    # Plane normal is the last singular vector
    normal = Vt[2, :]

    return normal, center

# Camera setup
camera = cv2.VideoCapture(0)

# NOTE: Replace with RealSense intrinsics
intrinsics = np.array([[1059.81, 0, 960], [0, 1059.81, 540], [0, 0, 1]])

while True:
    # Capture frame-by-frame.
    ret, frame = camera.read()

    # Convert the frame to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract keypoints.
            keypoints = [landmark for landmark in hand_landmarks.landmark]

            # NOTE: Replace the Z components of these keypoints with the RealSense depth for the given (X,Y) keypoints.

            # Convert keypoints to 3D.
            points_3D = keypoints_to_3D(keypoints, intrinsics)

            # Fit a plane and find the normal.
            plane_normal, plane_center = fit_plane_svd(points_3D)

            # Reproject normal vector into camera frame.
            # Assuming the length of the normal vector for visualization.
            normal_end = plane_center + plane_normal * 0.1

            # Convert 3D points back to 2D for visualization.
            normal_start_2D = np.array([plane_center[0] * intrinsics[0, 0] / plane_center[2] + intrinsics[0, 2],
                                        plane_center[1] * intrinsics[1, 1] / plane_center[2] + intrinsics[1, 2]])
            normal_end_2D = np.array([normal_end[0] * intrinsics[0, 0] / normal_end[2] + intrinsics[0, 2],
                                      normal_end[1] * intrinsics[1, 1] / normal_end[2] + intrinsics[1, 2]])

            # Draw the normal vector on the frame.
            cv2.arrowedLine(frame, tuple(normal_start_2D.astype(int)), tuple(normal_end_2D.astype(int)), (255, 0, 0), 2)

    # Display the resulting frame.
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


# Below is the code that should work with the RealSense.
# I am not able to get RealSense to work on my M1 mac.
# Pull the depth frame from this loop and plug it in as the Z value in the code above.

"""
import pyrealsense2 as rs
import numpy as np
import cv2

# NOTE: Replace with RealSense intrinsics
intrinsics = np.array([[1059.81, 0, 960], [0, 1059.81, 540], [0, 0, 1]])

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))


        results = hands.process(color_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract keypoints.
                keypoints = [landmark for landmark in hand_landmarks.landmark]



                # NOTE: Replace the Z components of these keypoints with the RealSense depth for the given (X,Y) keypoints.
                # Attempt is made below:

                # Extract X and Y coordinates
                x_coords = keypoints[:, 0].astype(int)
                y_coords = keypoints[:, 1].astype(int)

                # Ensure coordinates are within image bounds
                valid_indices = (x_coords >= 0) & (x_coords < depth_image.shape[1]) & (y_coords >= 0) & (y_coords < depth_image.shape[0])

                # Vectorized operation to replace Z values
                keypoints[valid_indices, 2] = depth_image[y_coords[valid_indices], x_coords[valid_indices]]


                # Convert keypoints to 3D.
                points_3D = keypoints_to_3D(keypoints, intrinsics)

                # Fit a plane and find the normal.
                plane_normal, plane_center = fit_plane_svd(points_3D)

                # Reproject normal vector into camera frame.
                # Assuming the length of the normal vector for visualization.
                normal_end = plane_center + plane_normal * 0.1

                # Convert 3D points back to 2D for visualization.
                normal_start_2D = np.array([plane_center[0] * intrinsics[0, 0] / plane_center[2] + intrinsics[0, 2],
                                            plane_center[1] * intrinsics[1, 1] / plane_center[2] + intrinsics[1, 2]])
                normal_end_2D = np.array([normal_end[0] * intrinsics[0, 0] / normal_end[2] + intrinsics[0, 2],
                                        normal_end[1] * intrinsics[1, 1] / normal_end[2] + intrinsics[1, 2]])

                # Draw the normal vector on the frame.
                cv2.arrowedLine(frame, tuple(normal_start_2D.astype(int)), tuple(normal_end_2D.astype(int)), (255, 0, 0), 2)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

"""


