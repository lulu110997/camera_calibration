import cv2

from cv_bridge import CvBridge

import sys
import rospy
import rosbag
from geometry_msgs.msg import PoseArray, Pose
BAG_PATH = "/home/louis/test1_userA.bag"
bridge = CvBridge()
rospy.init_node("rosbag_reader", anonymous=True)
rate = rospy.Rate(30)


def change_img(skel_list, img_):
    for j in skel_list:
        x = (round(j[0]), round(j[1]))
        cv2.circle(img=img_, center=x, radius=4, color=(59, 164, 0), thickness=-1)
def check_hand_tracking_data():
    img_time = []
    img_list = []
    try:
        with rosbag.Bag(BAG_PATH) as bag:
            for topic, msg, t in bag.read_messages(topics=['/mp_rgb_img']):
                img_time.append(t.secs + t.nsecs)
                cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                img_list.append(cv2_img)

            for topic, msg, t in bag.read_messages(topics=['/left_hand_skel_data', '/right_hand_skel_data']):
                left_hands_dict = {"skel": 0, "time": 0}
                right_hands_dict = {"skel": 0, "time": 0}\

                if topic == '/left_hand_skel_data':
                    left_hands_dict["time"] = t.secs + t.nsecs
                    temp_ = []
                    for pose_msg in msg.poses:
                        temp_.append([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
                    left_hands_dict["skel"] = temp_
                elif topic == '/right_hand_skel_data':
                    right_hands_dict["time"] = t.secs + t.nsecs
                    temp_ = []
                    for pose_msg in msg.poses:
                        temp_.append([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
                    right_hands_dict["skel"] = temp_

        print("showing images with skeleton data")
        for idx, img in enumerate(img_list):
            if rospy.is_shutdown():
                break
            now = img_time[idx]

            # Was the left hand extracted?
            try:
                l_hand_idx = left_hands_dict["time"].index(now)
                change_img(left_hands_dict["skel"], img)
            except:
                pass

            # Was the right hand extracted?
            try:
                r_hand_idx = right_hands_dict["time"].index(now)
                change_img(right_hands_dict["skel"], img)
            except:
                pass
            cv2.imshow('img', img)
            cv2.waitKey(1)
            rate.sleep()
    finally:
        cv2.destroyAllWindows()


def check_body_tracking_data():
    img_list = []
    joint_list = []
    try:
        with rosbag.Bag(BAG_PATH) as bag:
            for topic, msg, t in bag.read_messages(topics=['/nuitrack_rgb_image', '/nuitrack_skel_data']):
                if topic == '/nuitrack_rgb_image':
                    cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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

check_hand_tracking_data()