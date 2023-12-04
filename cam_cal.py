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

R_world2cam = []
t_world2cam = []
R_base2gripper = []
t_base2gripper = []

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

for i in range(len(ee_trans_path)):
    curr_cam_file = cam_trans_path[i]
    curr_ee_file = ee_trans_path[i]

    np_cam = np.load(curr_cam_file)
    np_ee = np.load(curr_ee_file)
    if ("rot_mat" in curr_cam_file) and ("rot_mat" in curr_ee_file):
        R_world2cam.append(np.linalg.inv(np_cam))
        R_base2gripper.append(np_ee)

        R_target2cam.append(np_cam)
        R_gripper2base.append(np.linalg.inv(np_ee))
    elif ("t_vec" in curr_cam_file) and ("t_vec" in curr_ee_file):
        t_world2cam.append(-np_cam)
        t_base2gripper.append(np_ee)

        t_target2cam.append(np_cam)
        t_gripper2base.append(-np_ee)
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
np.save(f"T_{CAM_NAME}2gripper_{TODAY}", T_cam2_gripper)

# T_base2gripper = np.load("T_base2gripper.npy")
# T_gripper2cam = create_T_matrix(np.linalg.inv(R_cam2gripper), -t_cam2gripper)
# T_cam2world = np.load("T_cam2world.npy")
# T_base2world = T_base2gripper @ T_gripper2cam @ T_cam2world
# print(T_base2world)
# R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(R_world2cam=R_world2cam,
#                                                                                           t_world2cam=t_world2cam,
#                                                                                           R_base2gripper=R_base2gripper,
#                                                                                           t_base2gripper=t_base2gripper,
#                                                                                           method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

# R_base2world = np.empty((3, 3))
# t_base2world = np.empty((3, 1))
# R_gripper2cam = np.empty((3, 3))
# t_gripper2cam = np.empty((3, 1))
# cv2.calibrateRobotWorldHandEye(R_world2cam=R_world2cam,
#                                t_world2cam=t_world2cam,
#                                R_base2gripper=R_base2gripper,
#                                t_base2gripper=t_base2gripper,
#                                R_base2world=R_base2world,
#                                t_base2world=t_base2world,
#                                R_gripper2cam=R_gripper2cam,
#                                t_gripper2cam=t_gripper2cam,
#                                method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
# print(t_gripper2cam)
# print(t_base2world)
# T_base2gripper = np.load("T_base2gripper.npy")
# T_gripper2cam = create_T_matrix(R_gripper2cam, t_gripper2cam)
# T_cam2world = np.load("T_cam2world.npy")
# # T_base2world = np.matmul(T_base2gripper, T_gripper2cam, T_cam2world)
# T_base2world = T_base2gripper @ T_gripper2cam @ T_cam2world
#
# print(cv2.Rodrigues(T_base2world[:3, :3])[0])
# print(T_base2world)