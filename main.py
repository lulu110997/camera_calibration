"""
Obtain transforms from ZED to world or RS to world as well as base2gripper transforms

Uses the ur_rtde interface
"""

import sys
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np
import pyrealsense2 as rs
import math

PRINT_CAM2WORLD = False  # Print values to check the rot and transl from cam2world
VIEW_BOARD_ONCE = False  # If you only want see the board once, mostly for when capturing the rot and transl elements
SAVE_ROT_TRANSL = False  # Saves the individual rot and transl elements for camera calibration
SAVE_TEST_POSE = False  # Saves a pose for testing the calibration values
SAVE_ZED2WORLD = False  # The transform from the ZED camera to a point in the world (tgt) frame

if SAVE_ZED2WORLD:
    from datetime import datetime
    # now = datetime.now()
    # TODAY = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    # CAMERA = "ZED"
else:
    wp_num = 11
    CAMERA = "RS"

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


def save_base2ee():
    """
    Saves the rotation matrix and translation vector of the UR from base to EE
    :return: None
    """
    # Connect to the robot
    rtde_r = RTDEReceive("192.158.5.2")

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
    # print([i*(180/3.14) for i in rtde_r.getActualQ()])
    # print(tvec)

def close_cam(cam):
    if CAMERA == "ZED":
        cam.close()
    else:
        cam.stop()


def init_cam():
    if CAMERA == "ZED":
        return init_zed()
    else:
        return init_rs()


def init_rs():
    pipeline = rs.pipeline()
    config = rs.config()
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


def get_rs_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    return color_image


def init_zed():
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


def get_zed_frame(zed, runtime_parameters):
    # A new image is available if grab() returns ERROR_CODE.SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_z = sl.Mat()
        zed.retrieve_image(image_z, sl.VIEW.LEFT)

        # Use get_data() to get the numpy array
        image_cv = np.array(image_z.get_data())
        return image_cv


def get_vid_frame(camera, runtime_parameters=None):
    if CAMERA == "ZED":
        return get_zed_frame(camera, runtime_parameters)
    else:
        return get_rs_frame(camera)


def get_charuco_board():
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = aruco.CharucoBoard_create(6, 8, 0.03, 0.022, dictionary)
    params = aruco.DetectorParameters_create()
    # Show calibration board
    # img = board.draw((200*3,200*3))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Dump the calibration board to a file
    # cv2.imwrite('charuco.png',img)

    return board, params


def create_T_matrix(rmatr, tvec):
    R = np.vstack((rmatr, np.zeros((1, 3))))
    t = np.vstack((tvec, np.ones((1, 1))))
    return np.hstack((R, t))


def save_np_arr(R_cam_target, tvec):
    np.save(f"rs_transforms/{CAMERA}_TO_TGT_{wp_num}_rot_mat", R_cam_target)
    np.save(f"rs_transforms/{CAMERA}_TO_TGT_{wp_num}_t_vec", tvec)


def save_cam2world():
    try:
        cam, runtime_params, cam_matr, dist_coeff = init_cam()
        board, char_params = get_charuco_board()
        # sys.exit()
        while True:
            frame = get_vid_frame(cam, runtime_params)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            res = aruco.detectMarkers(frame, board.dictionary, parameters=char_params)  # detects the individual aruco markers

            if not res[1] is None:
                # if there is at least one marker detected interpolate charuco corners, draw charuco corners on board and
                # estimate the charuco board's pose as well as showing it on the image
                char_retval, char_corners, char_ids = aruco.interpolateCornersCharuco(res[0], res[1], frame, board)
                frame = aruco.drawDetectedCornersCharuco(frame, char_corners, char_ids, (0, 255, 0))
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, cam_matr, dist_coeff,
                                                                    np.empty(1), np.empty(1))
                # rvec: rotation vector of the board(see cv::Rodrigues). Need to convert to rotation matrix
                # tvec: translation vector of the board

                if retval: # If charuco board is found
                    # Draw axis with length 0.1 units
                    frame = aruco.drawAxis(frame, cam_matr, dist_coeff, rvec, tvec, 0.1)

                    R_cam_target = cv2.Rodrigues(rvec)[0]  # Obtain rotation matrix
                    T_cam_target = create_T_matrix(R_cam_target, tvec)

                    if PRINT_CAM2WORLD:
                        print("Rot matrix")
                        print(R_cam_target)
                        print("tvec below")
                        print(tvec)
                        print("Transformation matrix below")
                        print(T_cam_target)

                    if SAVE_TEST_POSE:
                        np.save("T_cam2world.npy", T_cam_target)
                        T_base2gripper = create_T_matrix(*save_base2ee())
                        np.save("T_base2gripper.npy", T_base2gripper)

                    if SAVE_ROT_TRANSL:
                        save_np_arr(R_cam_target, tvec)
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