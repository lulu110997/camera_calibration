import cv2

from cv_bridge import CvBridge
import pandas as pd
import sys
import rospy
import rosbag
from geometry_msgs.msg import PoseArray, Pose
BAG_PATH = "/home/louis/Data/2023_12_HAR_bags/test0_userB.bag"
bridge = CvBridge()
rospy.init_node("rosbag_reader", anonymous=True)
rate = rospy.Rate(15)


def hs_data_to_csv():
    """
    Reads rosbag and converts hand skeletal data to csv files
    Returns: None
    """
    left_hand = {'time': [], 'data': []}
    right_hand = {'time': [], 'data': []}
    # mp_joint_names = ["WRIST",
    #                   "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TPI",
    #                   "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    #                   "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    #                   "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    #                   "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
# "WRIST" 0-2
# "THUMB_CMC", 3-5
# "THUMB_MCP", 6-8
# "THUMB_IP", 9-11
# "THUMB_TPI", 12-14
# "INDEX_MCP", 15-17
# "INDEX_PIP", 18-20
# "INDEX_DIP", 21-23
# "INDEX_TIP", 24-26
# "MIDDLE_MCP", 27-29
# "MIDDLE_PIP", 30-32
# "MIDDLE_DIP", 33-35
# "MIDDLE_TIP", 36-38
# "RING_MCP", 39-41
# "RING_PIP", 42-44
# "RING_DIP", 45-47
# "RING_TIP", 48-50
# "PINKY_MCP", 51-53
# "PINKY_PIP", 54-56
# "PINKY_DIP", 57-59
# "PINKY_TIP" 60-62
    with rosbag.Bag(BAG_PATH) as bag:
        # Obtain all the images first
        for topic, msg, t in bag.read_messages(topics=['/left_hand_skel_data', '/right_hand_skel_data']):
            if rospy.is_shutdown():
                break
            joint_lists = []
            if "left" in topic:
                hand = left_hand
            else:
                hand = right_hand

            hand["time"].append(msg.header.stamp)
            for idx, joints in enumerate(msg.poses):
                joint_lists.extend([joints.position.x,
                                    joints.position.y,
                                    joints.position.z])
            hand["data"].append(joint_lists)
    df = pd.DataFrame(left_hand["data"], index=left_hand["time"])
    df.to_csv("hs_right.csv", header=False)
    df = pd.DataFrame(right_hand["data"], index=right_hand["time"])
    df.to_csv("hs_left.csv", header=False)

def check_for_matching_ts():
    d = {'mp_rgb_img': [], 'left_hand_skel_data': [], 'right_hand_skel_data': [],
         'nuitrack_rgb_image': [], 'nuitrack_skel_data': []}

    with rosbag.Bag(BAG_PATH) as bag:
        # Obtain all the images first
        for topic, msg, t in bag.read_messages(topics=['/mp_rgb_img', '/left_hand_skel_data', '/right_hand_skel_data',
                                                       '/nuitrack_rgb_image', '/nuitrack_skel_data']):
            if rospy.is_shutdown():
                break
            topic_name = topic[1:]
            d[topic_name].append(t.to_sec())
        df = pd.DataFrame.from_dict(data=d, orient='index').transpose().to_csv("check_ts.csv", index=False)


def change_img(skel_list, img_):
    for j in skel_list:
        x = (round(j[0]), round(j[1]))
        cv2.circle(img=img_, center=x, radius=4, color=(59, 164, 0), thickness=-1)


def check_hand_tracking_data():
    try:
        with rosbag.Bag(BAG_PATH) as bag:
            # Obtain all the images first
            for topic, msg, t in bag.read_messages(topics=['/mp_rgb_img/compressed', '/left_hand_skel_data', '/right_hand_skel_data']):
                if rospy.is_shutdown():
                    break
                if topic == "/mp_rgb_img/compressed":
                    img_time = t.secs + t.nsecs
                    cv2_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    img_list = cv2_img
                    height_row, width_col = cv2_img.shape[:2]
                else:
                    hand_joints = []
                    hand_time = t.secs + t.nsecs
                    try:
                        print(hand_time == img_time)
                        for pose_msg in msg.poses:
                            hand_joints.append([pose_msg.position.x*width_col, pose_msg.position.y*height_row])

                        change_img(hand_joints, img_list)
                        cv2.imshow('img', img_list)
                        cv2.waitKey(1)
                        rate.sleep()
                    except Exception as e:
                        # print(e)
                        pass

    finally:
        cv2.destroyAllWindows()


def check_body_tracking_data():
    img_list = []
    joint_list = []
    try:
        with rosbag.Bag(BAG_PATH) as bag:
            for topic, msg, t in bag.read_messages(topics=['/nuitrack_rgb_image/compressed', '/nuitrack_skel_data']):
                if topic == '/nuitrack_rgb_image/compressed':
                    cv2_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    img_list.append(cv2_img)
                else:
                    joint_now = []
                    for pose_msg in msg.poses:
                        joint_now.append([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
                    joint_list.append(joint_now)
        print("showing images with skeleton data")
        for idx, joint_at_t in enumerate(joint_list):
            if rospy.is_shutdown():
                break
            for j in joint_at_t:
                x = (round(j[0]), round(j[1]))
                cv2.circle(img=img_list[idx], center=x, radius=8, color=(59, 164, 0), thickness=-1)
            cv2.imshow('img', img_list[idx])
            cv2.waitKey(1)
            rate.sleep()
    finally:
        cv2.destroyAllWindows()

hs_data_to_csv()
# check_body_tracking_data()
# check_hand_tracking_data()