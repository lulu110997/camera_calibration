T_gripper2rs is the output of the handeye calibration function. This is used to transform a point expressed in
the camera frame to the gripper frame. Rotation is slightly off, 20 points were used and method 4 was used to obtain
this transform
Useful functions:
- [calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
- [estimatePoseCharucoBoard](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#ga21b51b9e8c6422a4bac27e48fa0a150b)
- [estimatePoseBoard](https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gabb2578b9e18b13913b1d3e0ab1b554f9)
- [Eye-to-hand calibration](https://forum.opencv.org/t/eye-to-hand-calibration/5690/1)