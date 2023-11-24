import sys

import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
# now = datetime.now()
# TODAY = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
# CAMERA = "ZED"

wp_num = 1
TODAY = f"WP_{wp_num}"
CAMERA = "RS"

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


cam, runtime_params, cam_matr, dist_coeff = init_cam()
board, char_params = get_charuco_board()
# sys.exit()

# Calibration fails for lots of reasons. Release the video if we do
try:

    while True:
        frame = get_vid_frame(cam, runtime_params)
        # frame = get_rs_frame(rs_pipeline)
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

            if retval:
                # Draw axis with length 0.1 units
                frame = aruco.drawAxis(frame, cam_matr, dist_coeff, rvec, tvec, 0.1)

        R_cam_target = cv2.Rodrigues(rvec)[0]
        T_cam_target = create_T_matrix(R_cam_target, tvec)
        # print("Rot matrix")
        # print(R_cam_target)
        # print("tvec below")
        # print(tvec)
        # print("Transformation matrix below")
        print(T_cam_target)
        # break

        # np.save(f"{CAMERA}_TO_TGT_{TODAY}", T_cam_target)
        np.save(f"{CAMERA}_TO_TGT_{TODAY}_rot_mat", R_cam_target)
        np.save(f"{CAMERA}_TO_TGT_{TODAY}_t_vec", tvec)
        # print(np.load(f"ZED_TO_TGT_{TODAY}.npy"))
        break
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    close_cam(cam)

    cv2.imshow('frame', frame)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


except Exception as err:
    close_cam(cam)
    cv2.destroyAllWindows()
    raise err
