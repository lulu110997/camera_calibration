import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np


class CamTransforms:
    def __init__(self):
        self.zed = sl.Camera()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD1080 video mode
        init_params.camera_fps = 15  # Set fps at 30
        # init_params.enable_image_enhancement = True

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        self.runtime_parameters = sl.RuntimeParameters()

        # cat /usr/local/zed/settings/SN13699.conf
        fx = 1398.93
        fy = 1398.93
        cx = 1000.53
        cy = 575.924
        k1 = -0.169456
        k2 = 0.022525
        p1 = 0
        p2 = 0
        k3 = 0
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.dist_coefficients = np.array([[k1, k2, p1, p2, k3]])
