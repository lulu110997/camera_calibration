import cv2
import numpy as np
import os
from natsort import natsorted


def create_T_matrix(rmatr, tvec):
    R = np.vstack((rmatr, np.zeros((1, 3))))
    t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))


rs_trans_path = natsorted([os.path.join("rs_transforms", i) for i in os.listdir("rs_transforms")])
ee_trans_path = natsorted([os.path.join("ee_transforms", i) for i in os.listdir("ee_transforms")])

R_world2cam = []
t_world2cam = []
R_base2gripper = []
t_base2gripper = []

for i in range(len(ee_trans_path)):
    curr_rs_file = rs_trans_path[i]
    curr_ee_file = ee_trans_path[i]

    np_rs = np.load(curr_rs_file)
    np_ee = np.load(curr_ee_file)
    if ("rot_mat" in curr_rs_file) and ("rot_mat" in curr_ee_file):
        R_world2cam.append(np.linalg.inv(np_rs))
        R_base2gripper.append(np_ee)
    elif ("t_vec" in curr_rs_file) and ("t_vec" in curr_ee_file):
        t_world2cam.append(-np_rs)
        t_base2gripper.append(np_ee)
    else:
        print(curr_ee_file)
        print(curr_rs_file)
        raise "np arrays do not match"

R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(R_world2cam, t_world2cam,
                                                                      R_base2gripper, t_base2gripper, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
print(t_gripper2cam)
print(t_base2world)
# T_base2gripper = np.load("T_base2gripper.npy")
# T_gripper2cam = create_T_matrix(R_gripper2cam, t_gripper2cam)
# T_cam2world = np.load("T_cam2world.npy")
# # T_base2world = np.matmul(T_base2gripper, T_gripper2cam, T_cam2world)
# T_base2world = T_base2gripper @ T_gripper2cam @ T_cam2world
#
# print(cv2.Rodrigues(T_base2world[:3, :3])[0])
# print(T_base2world)