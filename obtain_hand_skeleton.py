import sys
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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose

image_dict = {"img": None, "lock": Lock()}
bridge = CvBridge()
image_pub = rospy.Publisher("mp_rgb_img", Image, queue_size=30)
left_hand_pub = rospy.Publisher("left_hand_skel_data", PoseArray, queue_size=30)
right_hand_pub = rospy.Publisher("right_hand_skel_data", PoseArray, queue_size=30)

def pub_img_data(image_np, ts):
    """
    Publisher for image where the skeleton data was extracted. Uses CvBridge
    Args:
        image_np: np array | corresponding RGB image to when the skeleton was extracted
        ts: int | ros time in nanoseconds corresponding ot when nuitrack.update was called

    Returns: None

    """
    image_msg = bridge.cv2_to_imgmsg(image_np, encoding="bgra8")
    image_pub.publish(image_msg)

rospy.init_node("random")
# RS_SN = '027322071961'
RS_SN = '017322070251'
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
    Publisher callback whenever data is received. This is where the hand skeleton data is published
    Args:
        result: HandLandmarkerResult | Stores the results of handlandmarker software
        output_image: mp.Image | Mediapipe image where the hand skeleton was extracted from
        timestamp_ms: int | ROS timestamp in milliseconds

    Returns: None
    """

    if len(result.hand_world_landmarks) > 0:
        # img_ = draw_landmarks_on_image(output_image.numpy_view(), result)
        # with image_dict["lock"]:
        #     image_dict["img"] = img_.copy()
        hands_list = result.hand_landmarks  # List of hands containing List of hand landmarks
        handedness = result.handedness  # Obtain whether the hand is left or right hand
        for idx, hand in enumerate(hands_list):  # Iterate through each hand
            if handedness[idx][0].category_name == "Left":
                hand_publisher = left_hand_pub
            else:
                hand_publisher = right_hand_pub
            hand_joints = PoseArray()
            # hand_joints.header.stamp.nsecs = timestamp_ms/10000
            for joint in hand:
                hand_pose = Pose()
                hand_pose.position.x = joint.x
                hand_pose.position.y = joint.y
                hand_pose.position.z = joint.z
                hand_pose.orientation.x = 0
                hand_pose.orientation.y = 0
                hand_pose.orientation.z = 0
                hand_pose.orientation.w = 1
                hand_joints.poses.append(hand_pose)
            hand_publisher.publish(hand_joints)

        # for idx in range(len(hand_joints)):
        #     hand_joints = hand_joints[idx]


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=publish_data)
try:
    # The landmarker is initialized. Use it here.
    with HandLandmarker.create_from_options(options) as landmarker:

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            rate.sleep()
            img = get_rs_frame(rs_cam)
            now_s = rospy.get_time()
            pub_img_data(img, now_s)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Need to convert from bgra8 to rgb8
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            # print(rospy.get_rostime().nsecs)
            landmarker.detect_async(mp_image, int(now_s*10000))
            # if image_dict["img"] is not None:
            #     with image_dict["lock"]:
            #         cv2.imshow("win", image_dict["img"])
            #         cv2.waitKey(1)
finally:
    print("closing camera")
    cv2.destroyAllWindows()
    rs_cam.stop()


