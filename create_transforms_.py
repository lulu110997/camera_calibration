import argparse
import cv2
import numpy as np
import os
from natsort import natsorted
from datetime import datetime

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

# CAM_NAME = "rs"
# cam_transforms = f"{CAM_NAME}_transforms"
# cam_trans_path = natsorted([os.path.join(cam_transforms, i) for i in os.listdir(cam_transforms)])

root_dir = "rs_transforms/"
save_dir = "rs_t/"

for pt_num in range(1,21):
    rot = np.load(f"{root_dir}RS_TO_TGT_{pt_num}_rot_mat.npy")
    trans = np.load(f"{root_dir}RS_TO_TGT_{pt_num}_t_vec.npy")
    T = create_T_matrix(rot, trans)
    np.save(f"{save_dir}RS_TO_TGT_{pt_num}_transform.npy", T)