"""
Assumes that there is a folder called ee_transforms and {CAM_NAME}_transforms which contain matching base2ee and
camera2tgt rot_mat/t_vec respectively. For example, RS_TO_TGT_0_rot_mat.npy and BASE_TO_EE_0_rot_mat.npy are matching
rotation matrices.

The calibrated transformation matrix is saved in the current directory with the following format:
{CAM_NAME}2gripper_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}
"""
import sys
import argparse
import cv2
import numpy as np
import os
from natsort import natsorted
from datetime import datetime
now = datetime.now()
TODAY = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

CAM_NAME = "rs"
METHOD = 4
print(f"camera name is {CAM_NAME}")
print(f"cv2.HandEyeCalibrationMethod used is {METHOD}")

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


cam_transforms = f"{CAM_NAME}_transforms"
cam_trans_path = natsorted([os.path.join(cam_transforms, i) for i in os.listdir(cam_transforms)])
ee_trans_path = natsorted([os.path.join("ee_transforms", i) for i in os.listdir("ee_transforms")])

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

num_transforms = len(ee_trans_path)

for i in range(num_transforms):
    curr_cam_file = cam_trans_path[i]
    curr_ee_file = ee_trans_path[i]
    np_cam = np.load(curr_cam_file)
    np_ee = np.load(curr_ee_file)
    if ("rot_mat" in curr_cam_file) and ("rot_mat" in curr_ee_file):
        R_target2cam.append(np_cam)
        R_gripper2base.append(np_ee)
    elif ("t_vec" in curr_cam_file) and ("t_vec" in curr_ee_file):
        t_target2cam.append(np_cam)
        t_gripper2base.append(np_ee)
    else:
        print(curr_ee_file)
        print(curr_cam_file)
        raise "np arrays do not match. See printouts above"

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base,
                                                    t_gripper2base=t_gripper2base,
                                                    R_target2cam=R_target2cam,
                                                    t_target2cam=t_target2cam,
                                                    method=METHOD)


T_cam2_gripper = create_T_matrix(R_cam2gripper, t_cam2gripper)
np.save(f"T_gripper2{CAM_NAME}_{TODAY}", T_cam2_gripper)
print(t_cam2gripper)
