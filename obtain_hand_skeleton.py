import time
from mp_viz import draw_landmarks_on_image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyrealsense2 as rs
import rospy
from threading import Thread, Lock

image_dict = {"img": None, "lock": Lock()}

rospy.init_node("random")
RS_SN = '027322071961'

def init_rs():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(RS_SN)
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


# Setup mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def publish_data(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Publisher callback whenever data is received. This is where the hand skeleton data and corresponding image will be
    published
    Args:
        result: HandLandmarkerResult | Stores the results of handlandmarker software
        output_image: mp.Image | Mediapipe image where the hand skeleton was extracted from
        timestamp_ms: int | ROS timestamp in nanoseconds

    Returns: None

    """
    # print('hand landmarker result: {}'.format(result))
    if len(result.hand_world_landmarks) > 0:
        img_ = draw_landmarks_on_image(output_image.numpy_view(), result)
        with image_dict["lock"]:
            image_dict["img"] = img_.copy()


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=publish_data)

# The landmarker is initialized. Use it here.
with HandLandmarker.create_from_options(options) as landmarker:

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rate.sleep()
        img = get_rs_frame(rs_cam)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Need to convert from bgra8 to rgb8
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        frame_timestamp_ms = round(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        if image_dict["img"] is not None:
            with image_dict["lock"]:
                cv2.imshow("win", image_dict["img"])
                cv2.waitKey(1)

print("closing camera")
cv2.destroyAllWindows()
rs_cam.stop()


