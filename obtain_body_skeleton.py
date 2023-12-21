import sys

from PyNuitrack import py_nuitrack
import cv2
import numpy as np
import rospy
from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

MILLISECONDS = 1000.0
VISUALISE_DEPTH = 0
VISUALISE_COLOR = 0
rospy.init_node("body_tracking")
image_pub = rospy.Publisher("nuitrack_rgb_image", Image, queue_size=30)
skel_pub = rospy.Publisher("nuitrack_skel_data", PoseArray, queue_size=30)
bridge = CvBridge()


def create_T_matrix(rmatr, tvec):
    """
    Create a homogenous transformation matrix from the rotation matrix and translation vector
    Args:
        rmatr: 3x3 rotation matrix
        tvec: 3x1 translation vector

    Returns: 4x4 homogenous transformation matrix
    """
    # Check the shapes of the rotation matrix and translation vector is correct
    assert tvec.shape == (3, 1), f"The translation vector is the wrong shape! It is {tvec.shape}"
    assert rmatr.shape == (3, 3), f"The rotation matrix is not 3x3! It is {rmatr.shape}"

    R = np.vstack((rmatr, np.zeros((1, 3))))
    t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))


def draw_skeleton(skel_data, img_vis):
    """
    Visualises the first set of joint identified by the software
    Args:
        skel_data: SkeletonResult | output of get_skeleton
        img_vis: img to plot on

    Returns: None

    """
    s = skel_data.skeletons[0]  # Only plot the first person
    for el in s[1:]:
        if (el.projection == [0, 0, 0]).all():
            # Software cannot find joint. Set to None? or -1? or stay at 0?
            continue
        # if str(el.type) == "right_hand":
        #     print(el.projection)
        x = (round(el.projection[0]), round(el.projection[1]))
        cv2.circle(img=img_vis, center=x, radius=8, color=(59, 164, 0), thickness=-1)
        cv2.imshow('img', img_vis)
        cv2.waitKey(1)


def pub_skel_data(skel, image_np, ts):
    """
    Publisher for the skeleton data and for image where the skeleton data was extracted. Uses CvBridge
    Args:
        skel: output of get_skeleton | contains SkeletonResult object
        image_np: np array | corresponding RGB image to when the skeleton was extracted
        ts: int | ros Time which was converted and rounded to milliseconds

    Returns: None
    """
    time_ros = rospy.Time.from_sec(ts / MILLISECONDS)
    s = skel.skeletons[0]  # We only care about the tracking the joints of the first (hopefully closest) person
    joint_poses = PoseArray()
    joint_poses.header.stamp.secs = time_ros.secs
    joint_poses.header.stamp.nsecs = time_ros.nsecs
    for j in s[1:]:
        pose = Pose()  # create a new Pose message and populate its fields
        pose.position.x, pose.position.y, pose.position.z = j.projection[0], j.projection[1], j.projection[2]
        T = create_T_matrix(j.orientation, j.projection.reshape(3, 1))
        qx, qy, qz, qw = quaternion_from_matrix(T)  # This function takes in a a 4x4 SE(3) matrix
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        joint_poses.poses.append(pose)  # add the new Pose object to the PoseArray list
    skel_pub.publish(joint_poses)

    # Only want an image if we have extracted a skeleton from that img
    if len(joint_poses.poses):
        image_msg = bridge.cv2_to_imgmsg(image_np, encoding="bgr8")
        image_msg.header.stamp.secs = time_ros.secs
        image_msg.header.stamp.nsecs = time_ros.nsecs
        image_pub.publish(image_msg)


def init_nuitrack():
    """
    Initialise nuitrack software
    Returns: nuitrack object

    """
    nuitrack = py_nuitrack.Nuitrack()
    nuitrack.init()

    # Configure settings of /usr/etc/nuitrack/data/nuitrack.config
    nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")  # Not sure what this does
    nuitrack.set_config_value("Skeletonization.Type", "CNN_HPE")  # Uses deep learning for skeleton tracking
    nuitrack.set_config_value("Skeletonization.MaxDistance", "1700")  # Max distance (mm) to look for skeleton
    nuitrack.set_config_value("Segmentation.MAX_DISTANCE", "1700")  # Max distance (mm) to look for skeleton
    nuitrack.set_config_value("Skeletonization.ActiveUsers", "1")  # Number of persons to detect

    # Use the Kinect as the sensor for the nuitrack software
    devices = nuitrack.get_device_list()
    cam_activated = False
    for dev in devices:
        if dev.get_name() == "Kinect V1" and dev.get_serial_number() == "B00361200505042B":
            nuitrack.set_device(dev)
            cam_activated = True
            break

    if not cam_activated:
        raise "Could not find Kinect V1"

    # Create modules based on the config file values and then run the nuitrack software
    nuitrack.create_modules()
    nuitrack.run()

    return nuitrack


def main(nuitrack):
    """
    Main function that extracts body skeleton data
    Returns: None
    """
    try:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()
            nuitrack.update()
            now = round(rospy.get_time()*MILLISECONDS)  # Converted to ms to suit mediapipe
            skel_data = nuitrack.get_skeleton()
            img_color = nuitrack.get_color_data()
            if skel_data.skeletons:
                pub_skel_data(skel_data, img_color, now)
                if VISUALISE_DEPTH:
                    img_vis = nuitrack.get_depth_data()
                    cv2.normalize(img_vis, img_vis, 0, 255, cv2.NORM_MINMAX)
                    img_vis = np.array(cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
                    draw_skeleton(skel_data, img_vis)
                elif VISUALISE_COLOR:
                    draw_skeleton(skel_data, img_color)

            else:
                pass
    finally:
        print("Cleaning up")
        nuitrack.release()
        cv2.destroyAllWindows()

main(init_nuitrack())