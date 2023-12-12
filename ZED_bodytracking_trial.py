import sys

from PyNuitrack import py_nuitrack
import cv2
from itertools import cycle
import numpy as np
import rospy

rospy.init_node("skeleton_tracking")

def draw_face(image):
    if not data_instance:
        return
    for instance in data_instance["Instances"]:
        line_color = (59, 164, 225)
        text_color = (59, 255, 255)
        if 'face' in instance.keys():
            bbox = instance["face"]["rectangle"]
        else:
            return
        x1 = (round(bbox["left"]), round(bbox["top"]))
        x2 = (round(bbox["left"]) + round(bbox["width"]), round(bbox["top"]))
        x3 = (round(bbox["left"]), round(bbox["top"]) + round(bbox["height"]))
        x4 = (round(bbox["left"]) + round(bbox["width"]), round(bbox["top"]) + round(bbox["height"]))
        cv2.line(image, x1, x2, line_color, 3)
        cv2.line(image, x1, x3, line_color, 3)
        cv2.line(image, x2, x4, line_color, 3)
        cv2.line(image, x3, x4, line_color, 3)
        cv2.putText(image, "User {}".format(instance["id"]),
        x1, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(image, "{} {}".format(instance["face"]["gender"],int(instance["face"]["age"]["years"])),
        x3, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

def draw_skeleton(image):
    point_color = (59, 164, 0)
    for skel in data.skeletons:
        for el in skel[1:]:
            x = (round(el.projection[0]), round(el.projection[1]))
            cv2.circle(image, x, 8, point_color, -1)

nuitrack = py_nuitrack.Nuitrack()
nuitrack.init()

# ---enable if you want to use face tracking---
# nuitrack.set_config_value("Faces.ToUse", "true")
nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
nuitrack.set_config_value("Skeletonization.Type", "CNN_HPE")
# nuitrack.set_config_value("Skeletonization.MaxDistance", "2000")
# nuitrack.set_config_value("Skeletonization.ActiveUsers", "1")

devices = nuitrack.get_device_list()
cam_activated = False
for dev in devices:
    if dev.get_name() == "Kinect V1" and dev.get_serial_number() == "B00361200505042B":
        #dev.activate("ACTIVATION_KEY") #you can activate device using python api
        print(nuitrack.get_version())
        print(nuitrack.get_license())
        nuitrack.set_device(dev)
        cam_activated = True

if not cam_activated:
    raise "Could not find Kinect V1"


nuitrack.create_modules()
nuitrack.run()

modes = cycle(["depth", "color"])
mode = next(modes)
rate = rospy.Rate(30)
while not rospy.is_shutdown():
    rate.sleep()
    key = cv2.waitKey(1)
    nuitrack.update()
    data = nuitrack.get_skeleton()
    data_instance=nuitrack.get_instance()
    img_depth = nuitrack.get_depth_data()
    if img_depth.size:
        cv2.normalize(img_depth, img_depth, 0, 255, cv2.NORM_MINMAX)
        img_depth = np.array(cv2.cvtColor(img_depth,cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        img_color = nuitrack.get_color_data()
        draw_skeleton(img_depth)
        draw_skeleton(img_color)

        if key == 32:
            mode = next(modes)
        if mode == "depth":
            cv2.imshow('Image', img_depth)
        if mode == "color":
            if img_color.size:
                cv2.imshow('Image', img_color)
    if key == 27:
        break

nuitrack.release()