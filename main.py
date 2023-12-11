"""
Obtain transforms from ZED to world or RS to world as well as base2gripper transforms

Uses the ur_rtde interface
"""

import sys
import time

from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np
import pyrealsense2 as rs

ROBOT_IP = "172.31.1.200"
# For viewing the created charuco board. https://calib.io/pages/camera-calibration-pattern-generator was used to
# generate the charuco board. The following params were used (squaresX=6, squaresY=8, squareLength=0.03,
# markerLength=0.022, dictionary=aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SHOW_CHARUCO_BOARD = False

# Debugging:
PRINT_CAM2WORLD = False  # Print values to check the rot and transl from cam2world
VIEW_BOARD_ONCE = False  # If you only want see the board once, mostly for when capturing the rot and transl elements

# Calibration:
SAVE_ROT_TRANSL = False  # Saves the individual rot and transl elements
SAVE_TEST_POSE = False  # Saves a pose (transformation matrix) for testing the calibration values
SAVE_ZED2WORLD = False  # The transform from the ZED camera to a point in the world (tgt) frame

# Testing:
tgt_loc = "table"  # Location of the charuco board (target)

if SAVE_ZED2WORLD:
    from datetime import datetime
    now = datetime.now()
    TODAY = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    CAMERA = "ZED"
else:
    wp_num = 101
    CAMERA = "rs"
    RS_GRIPPER_SN = '101622072236'
    RS_FRAME_SN = '027322071961'

def save_base2ee():
    """
    Saves the rotation matrix and translation vector of the UR from base to EE
    Returns: None if being used for camera calibration. If for testing,
        r_matr: rotation matrix that transforms point from EE frame to base frame
        tvec: translation vector that transforms point from EE frame to base frame
    """

    # Connect to the robot
    rtde_r = RTDEReceive(ROBOT_IP)
    rtde_c = RTDEControl(ROBOT_IP, 500)
    time.sleep(1)
    rtde_c.setTcp([0.0, 0.0, 0.15769999999999998, 0.0, 0.0, 0.0])
    print(rtde_c.getTCPOffset())

    # Obtain EE pose and convert rotation vector to rotation matrix
    ee_pose = rtde_r.getActualTCPPose()
    rvec = np.array(ee_pose[3:])
    tvec = np.array(ee_pose[:3]).reshape(3, 1)
    r_matr = cv2.Rodrigues(rvec)[0]

    if SAVE_TEST_POSE:
        return r_matr, tvec
    else:
        # Save rotation matrix and translation vector
        np.save(f"ee_transforms/BASE_TO_EE_{wp_num}_rot_mat", r_matr)
        np.save(f"ee_transforms/BASE_TO_EE_{wp_num}_t_vec", tvec)


def close_cam(cam):
    """
    Closes camera when script terminates
    """
    if CAMERA == "ZED":
        cam.close()
    else:
        cam.stop()


def init_cam():
    """
    Initialises the camera to be calibrated with the robot base
    Returns: Object for obtaining image from camera

    """
    if CAMERA == "ZED":
        return _init_zed()
    else:
        return _init_rs()


def _init_rs():
    """
    Initialises rs camera for use
    Returns:
        zed: o/p of rs.pipeline()
        runtime_parameters: None. Only required for ZED camera
        camera_matrix: np array | Contains the location of the cameras focal length and principal points
        dist_coefficients: np array | Distortion coefficients of the camera
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(RS_GRIPPER_SN)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgra8, 30)
    # config.enable_stream(rs.stream.infrared, 424, 240, rs.format.y8, 6)
    cfg = pipeline.start(config)

    profile = cfg.get_stream(rs.stream.color)  # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics

    # print(intr)
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coefficients = None

    return pipeline, None, camera_matrix, dist_coefficients


def _get_rs_frame(pipeline):
    """
    Outputs the image from rs camera's colour stream
    Args:
        zed: output of sl.Camera()
        runtime_parameters: output of sl.RuntimeParameters()

    Returns: np array that represents a bgra8 image
    """
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    return color_image


def _init_zed():
    """
    Initialises ZED camera for use
    Returns:
        zed: o/p of sl.Camera()
        runtime_parameters: o/p of sl.RuntimeParameters()
        camera_matrix: np array | Contains the location of the cameras focal length and principal points
        dist_coefficients: np array | Distortion coefficients of the camera
    """
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD1080 video mode
    init_params.camera_fps = 15  # Set fps at 30
    # init_params.enable_image_enhancement = True


    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    runtime_parameters = sl.RuntimeParameters()

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
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coefficients = np.array([[k1, k2, p1, p2, k3]])

    return zed, runtime_parameters, camera_matrix, dist_coefficients


def _get_zed_frame(zed, runtime_parameters):
    """
    Outputs the image from left ZED camera
    Args:
        zed: output of sl.Camera()
        runtime_parameters: output of sl.RuntimeParameters()

    Returns: np array that represents a bgra8 image
    """
    # A new image is available if grab() returns ERROR_CODE.SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_z = sl.Mat()
        zed.retrieve_image(image_z, sl.VIEW.LEFT)

        # Use get_data() to get the numpy array
        image_cv = np.array(image_z.get_data())
        return image_cv


def get_vid_frame(camera, runtime_parameters=None):
    """
    Wrapper used to obtain the video frame from a camera
    Args:
        camera: output of the init_cam() function
        runtime_parameters: used for ZED only. None for rs

    Returns: np array that represents a bgra8 image

    """
    if CAMERA == "ZED":
        return _get_zed_frame(camera, runtime_parameters)
    elif CAMERA == "rs":
        return _get_rs_frame(camera)
    else:
        raise "Chosen camera must be 'ZED' or 'rs'"


def get_charuco_board():
    """
    Obtains a charuco board with the given params
    Returns:
        board: the output CharucoBoard object |
        params: output of cv2.aruco.DetectorParameters_create | contains marker for the detectMarker process
    """
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = aruco.CharucoBoard_create(6, 8, 0.03, 0.022, dictionary)
    params = aruco.DetectorParameters_create()

    # Show calibration board
    if SHOW_CHARUCO_BOARD:
        img = board.draw((200*3, 200*3))
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board, params


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


def save_cam_arr(R_matr, tvec):
    """
    Saves the rotation matrix and target vector for the camera
    Args:
        R_matr: Rotation matrix of the target relative to the camera
        tvec: Translation vector of the target relative to the camera

    Returns: None

    """
    np.save(f"{CAMERA}_transforms/{CAMERA}_TO_TGT_{wp_num}_rot_mat", R_matr)
    np.save(f"{CAMERA}_transforms/{CAMERA}_TO_TGT_{wp_num}_t_vec", tvec)


def save_cam2world():
    try:
        cam, runtime_params, cam_matr, dist_coeff = init_cam()
        board, char_params = get_charuco_board()

        while True:
            frame = get_vid_frame(cam, runtime_params)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            res = aruco.detectMarkers(frame, board.dictionary, parameters=char_params)  # detects the individual aruco markers

            if not res[1] is None:
                # if there is at least one marker detected interpolate charuco corners, draw charuco corners on board
                # and estimate the charuco board's pose as well as showing it on the image
                char_retval, char_corners, char_ids = aruco.interpolateCornersCharuco(res[0], res[1], frame, board)
                frame = aruco.drawDetectedCornersCharuco(frame, char_corners, char_ids, (0, 255, 0))
                # The output of estimatePoseCharucoBoard is a transform that transforms a point in the target frame to
                # the camera frame. It can also be viewed as the pose of the target relative to the camera frame
                # rvec: rotation vector of the board(see cv::Rodrigues)
                # tvec: translation vector of the board
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, cam_matr, dist_coeff,
                                                                    np.empty(1), np.empty(1))

                if retval: # If charuco board is found
                    # Draw axis with length 0.1 units
                    frame = aruco.drawAxis(frame, cam_matr, dist_coeff, rvec, tvec, 0.1)

                    R_cam2target = cv2.Rodrigues(rvec)[0]  # Obtain rotation matrix
                    T_target2cam = create_T_matrix(R_cam2target, tvec)

                    if PRINT_CAM2WORLD:
                        print("Rot matrix")
                        print(R_cam2target)
                        print("Translation vector")
                        print(tvec)
                        print("Transformation matrix")
                        print(T_target2cam)

                    if SAVE_TEST_POSE:
                        np.save(f"test_transforms/T_cam2world_{tgt_loc}.npy", T_target2cam)
                        T_base2gripper = create_T_matrix(*save_base2ee())
                        np.save(f"test_transforms/T_base2gripper_{tgt_loc}.npy", T_base2gripper)

                    if SAVE_ROT_TRANSL:
                        save_cam_arr(R_cam2target, tvec)
                        save_base2ee()

                    if not VIEW_BOARD_ONCE:
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        cv2.imshow('frame', frame)
                        if cv2.waitKey(0):
                            cv2.destroyAllWindows()
                            break

        close_cam(cam)




    except Exception as err:
        close_cam(cam)
        cv2.destroyAllWindows()
        raise err


save_cam2world()