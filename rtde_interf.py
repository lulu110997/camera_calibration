# https://github.com/UniversalRobots/RTDE_Python_Client_Library
import sys

from rtde_receive import RTDEReceiveInterface as RTDEReceive
import rtde_control
import cv2
import numpy as np
import time
from rtde_control import RTDEControlInterface as RTDEControl
import math
from spatialmath.base import rotx
T_gripper2cam = np.load("T_gripper2rs.npy")
T_base2gripper = np.load("test_transforms/T_base2gripper_table.npy")
T_cam2world = np.load("test_transforms/T_cam2world_table.npy")


def tcp_pose_scal(pose):
    """
    from https://forum.universal-robots.com/t/state-actual-tcp-pose-results-in-wrong-pose/14498/9
    Mainly for printing out the scaled axis angles (in radians)
    Args:
        pose: ee_pose

    Returns:scaled axis-angle orientation
    """
    v = pose
    pi = 3.1416
    l = math.sqrt(pow(v[3], 2) + pow(v[4], 2) + pow(v[5], 2))
    scale = 1 - 2 * pi / l
    if ((np.linalg.norm(v[3]) >= 0.001 and v[3] < 0.0) or (np.linalg.norm(v[3]) < 0.001 and np.linalg.norm(v[4]) >= 0.001 and v[4] < 0.0) or (
            np.linalg.norm(v[3]) < 0.001 and np.linalg.norm(v[4]) < 0.001 and v[5] < 0.0)):
        tcp_pose = [v[0], v[1], v[2], scale * v[3], scale * v[4], scale * v[5]]
    else:
        tcp_pose = v

    return tcp_pose[3:]


def create_T_matrix(rmatr, tvec):
    """
    Create a homogenous transformation matrix from the rotation matrix and translation vector
    Args:
        rmatr: 3x3 rotation matrix
        tvec: 3x1 translation vector

    Returns: 4x4 homogenous transformation matrix
    """
    R = np.vstack((rmatr, np.zeros((1, 3))))
    t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))


def get_pose_vector():
    """
    Obtain the pose vector ur rtde expects
    Returns: pose vector
    """
    T_cam2world[:3, :3] = rotx(180, 'deg') @ T_cam2world[:3, :3]
    T_base2world = T_base2gripper @ T_gripper2cam @ T_cam2world
    r_vec = cv2.Rodrigues(T_base2world[:3, :3])[0].squeeze().tolist()
    t_vec = T_base2world[:3, 3].tolist()
    return t_vec + r_vec

# Connect to UR
rtde_frequency = 500.0
rtde_c = RTDEControl("172.31.1.200", rtde_frequency)
rtde_r = RTDEReceive("172.31.1.200")

# Obtain initial q and move robot to initial q
# q = rtde_r.getActualQ(); print(q); sys.exit()
init_q = [0.5454618334770203, -1.5271859516254445, 1.3794172445880335, -1.7549616299071253, 4.627932548522949, -0.13782769838442022]
# rtde_c.moveJ(init_q, speed=0.1)

# Obtain desired pose based on charuco board location
des_ee_pose = get_pose_vector()
sol_q = rtde_c.getInverseKinematics(x=des_ee_pose, qnear=init_q)  # Find suitable joint states

# Either move the robot or just print the joint angles
# rtde_c.moveJ(sol_q, speed=0.1)
sol_q_deg = [i * 57.296 for i in sol_q]
print(rtde_c.getTCPOffset())
print(f"x_des: {des_ee_pose[:3] + tcp_pose_scal(des_ee_pose)}")
print(f"q: {sol_q_deg}")

# rtde_c.freedriveMode([1, 1, 1, 1, 1, 1])
#
# start = time.time()
# while time.time()-start < 10:
#     time.sleep(0.5)
#     ee_pose = rtde_r.getActualTCPPose()
#     rvec = np.array(ee_pose[3:])
#     tvec = np.array(ee_pose[:3]).reshape(3, 1)
#     r_matr = cv2.Rodrigues(rvec)[0]
#
#     print(r_matr)
#     print(tvec)
#
# rtde_c.endFreedriveMode()
