# https://github.com/UniversalRobots/RTDE_Python_Client_Library
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import rtde_control
import cv2
import numpy as np
import time
from rtde_control import RTDEControlInterface as RTDEControl
import math
from spatialmath.base import rotx
T_rs2gripper = np.load("T_rs2gripper.npy")

def tcp_pose_scal(pose):
    """
    from https://forum.universal-robots.com/t/state-actual-tcp-pose-results-in-wrong-pose/14498/9
    :param pose: ee_pose
    :return: scaled axis-angle orientation
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
    R = np.vstack((rmatr, np.zeros((1, 3))))
    t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))

def get_pose_vector():
    """
    Obtain the pose vector ur rtde expects
    :return: pose vector
    """
    T_base2gripper = np.load("T_base2gripper.npy")
    T_cam2world = np.load("T_cam2world.npy")
    T_cam2world[:3, :3] = T_cam2world[:3, :3] @ rotx(180, 'deg')
    T_base2world = T_base2gripper @ T_rs2gripper @ T_cam2world
    r_vec = cv2.Rodrigues(T_base2world[:3, :3])[0].squeeze().tolist()
    t_vec = T_base2world[:3, 3].tolist()
    return t_vec + r_vec

def obtain_curr_T(rtde_r):
    ee_pose = rtde_r.getActualTCPPose()
    rvec = np.array(ee_pose[3:])
    tvec = np.array(ee_pose[:3]).reshape(3, 1)
    r_matr = cv2.Rodrigues(rvec)[0]
    T_base2gripper = create_T_matrix(r_matr, tvec)
    np.save("T_base2gripper", T_base2gripper)


rtde_frequency = 500.0
rtde_c = RTDEControl("192.158.5.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_r = RTDEReceive("192.158.5.2")
# q = rtde_r.getActualQ(); print(q)
init_q = [-0.15263349214662725, -1.9617778263487757, 1.8334344069110315, -1.5824738941588343, -1.3699825445758265, -3.1928420702563685]
rtde_c.moveJ(init_q, speed=0.4)
des_ee_pose = get_pose_vector()
des_ee_pose[2] = des_ee_pose[2] + 0.05
sol_q = rtde_c.getInverseKinematics(x=des_ee_pose, qnear=init_q)
rtde_c.moveJ(sol_q, speed=0.1)
sol_q_deg = [i * 57.296 for i in sol_q]
print(sol_q_deg)

# print("connecting to robot...")
# rtde_r = rtde_receive.RTDEReceiveInterface("192.158.5.2")
# rtde_c = rtde_control.RTDEControlInterface("192.158.5.2")
# init_q = [-0.35702735582460576, -0.9135687512210389, 0.656994644795553,
#           -1.312950925236084, -1.5706594626056116, -0.3629344145404261]
# final_q = rtde_c.getInverseKinematics([-0.68025457, -0.06188002, -0.01239602,
#                                        -0.07850479, -1.50668142, -0.30036346], qnear=init_q)
# print(rtde_r.getActualTCPPose())
# print(rtde_r.getActualQ())
# rtde_c.moveJ(init_q, speed=0.4)

# obtain_curr_T(rtde_r)

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


# import numpy as np
# import cv2
#
# c_points = np.loadtxt("ss. ur_waypoints/c_points.txt", delimiter=",")
#
# for idx, c in enumerate(c_points, start=1):
#     rvec = c[3:]
#     tvec = c[:3]
#     rmat = cv2.Rodrigues(rvec)[0]
#     np.save(f"ee_transforms/BASE_TO_EE_{idx}_rot_mat", rmat)
#     np.save(f"ee_transforms/BASE_TO_EE_{idx}_t_vec", tvec)
