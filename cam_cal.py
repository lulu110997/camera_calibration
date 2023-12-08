"""
Eye-in-hand calibration script with the following requirements
natsort==8.4.0
numpy==1.24.4
opencv-python==4.8.1.78

Assumes that there is a folder called ee_transforms and {CAM_NAME}_transforms which contain matching base2ee and
camera2tgt rot_mat/t_vec respectively. For example, RS_TO_TGT_0_rot_mat.npy and BASE_TO_EE_0_rot_mat.npy are matching
rotation matrices.

The calibrated transformation matrix is saved in the current directory with the following format:
{CAM_NAME}2gripper_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}

Run the scripts as follows ./cam_cal {camera name} {handeyecalib method, default: CALIB_HAND_EYE_TSAI}
hand-eye-calib methods: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99
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

parser = argparse.ArgumentParser(description='Eye-in-hand calibration script')
parser.add_argument('--CAM_NAME', metavar='camera name', default="rs",  help='name of camera')
parser.add_argument('--METHOD', metavar='cv2.HandEyeCalibrationMethod', default=cv2.CALIB_HAND_EYE_TSAI,
                    help='A cv2.HandEyeCalibrationMethod')

args = parser.parse_args()
CAM_NAME = args.CAM_NAME
METHOD = args.METHOD
print(f"camera name is {CAM_NAME}")
print(f"cv2.HandEyeCalibrationMethod used is {METHOD}")

def create_T_matrix(rmatr, tvec):
    """
    Create a homogenous transformation matrix from the rotation matrix and translation vector
    :param rmatr: 3x3 rotation matrix
    :param tvec: 3x1 translation vector
    :return: 4x4 homogenous transformation matrix
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

num_transforms = 20

for i in range(40):
    curr_cam_file = cam_trans_path[i]
    curr_ee_file = ee_trans_path[i]
    np_cam = np.load(curr_cam_file)
    np_ee = np.load(curr_ee_file)
    if ("rot_mat" in curr_cam_file) and ("rot_mat" in curr_ee_file):
        R_target2cam.append(np_cam)
        rot_ee = np.transpose(np_ee)
        R_gripper2base.append(rot_ee)
    elif ("t_vec" in curr_cam_file) and ("t_vec" in curr_ee_file):
        t_target2cam.append(np_cam)
        t_gripper2base.append(-rot_ee @ np_ee)
    else:
        print(curr_ee_file)
        print(curr_cam_file)
        raise "np arrays do not match. See printouts above"
METHOD=4
TODAY=METHOD
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base,
                                                    t_gripper2base=t_gripper2base,
                                                    R_target2cam=R_target2cam,
                                                    t_target2cam=t_target2cam,
                                                    method=METHOD)


T_cam2_gripper = create_T_matrix(R_cam2gripper, t_cam2gripper)
np.save(f"T_{CAM_NAME}2gripper_{TODAY}", T_cam2_gripper)
print(T_cam2_gripper)