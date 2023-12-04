import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyrealsense2 as rs
import rospy
rospy.init_node("random")


RS_GRIPPER_SN = '101622072236'
RS_FRAME_SN = '027322071961'
def init_rs():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(RS_GRIPPER_SN)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgra8, 30)
    # config.enable_stream(rs.stream.infrared, 424, 240, rs.format.y8, 6)
    cfg = pipeline.start(config)

    profile = cfg.get_stream(rs.stream.color)  # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics

    # print(intr)
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coefficients = None

    return pipeline, None, camera_matrix, dist_coefficients


def get_rs_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    return color_image


rs_cam, __, rs_cam_par, rs_dist_coeff = init_rs()

model_path = "/home/louis/Git/camera_calibration/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        img = get_rs_frame(rs_cam)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img) # Need to convert from bgra8 to rgb8?
        frame_timestamp_ms = round(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

print("closing camera")
rs_cam.stop()


