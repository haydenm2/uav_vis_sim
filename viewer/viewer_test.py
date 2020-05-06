#!/usr/bin/env python3

from viewer import vision_uav_viewer
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initialize viewer
    x0 = np.array([[-10], [0], [100]])
    xt0 = np.array([[0], [0], [0]])
    ypr = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    ypr_g = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    h_fov = np.deg2rad(45)
    v_fov = np.deg2rad(45)
    vis = vision_uav_viewer(x0, xt0, ypr, ypr_g, h_fov, v_fov)

    ##########################################################################
    ########################### Simulator Animations #########################
    ##########################################################################
    # example of cycling through UAV orientations
    print("Displaying UAV Orientation Animation Sample...")
    for i in range(0, 20, 1):
        vis.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    for j in range(0, 20, 1):
        vis.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    for k in range(0, 20, 1):
        vis.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))

    # example of cycling through gimbal orientations
    print("Displaying UAV Gimbal Orientation Animation Sample...")
    for i in range(0, 30, 1):
        vis.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    for j in range(0, -10, -1):
        vis.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    for k in range(0, -10, -1):
        vis.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))

    # example of cycling through UAV positions
    print("Displaying UAV Position Animation Sample...")
    for i in range(-10, 20, 1):
        vis.UpdateX(np.array((i, 0, 100)))
    for j in range(0, 20, 1):
        vis.UpdateX(np.array((i, j, 100)))
    for k in range(100, 40, -1):
        vis.UpdateX(np.array((i, j, k)))

    # example of cycling through target positions
    print("Displaying Target Position Animation Sample...")
    for i in range(0, 3, 1):
        vis.UpdateTargetX(np.array((i, 0, 0)))
    for j in range(0, 25, 1):
        vis.UpdateTargetX(np.array((i, j, 0)))
    for k in range(0, 10, 1):
        vis.UpdateTargetX(np.array((i, j, k)))

    # example of cycling through camera FOV angles
    print("Displaying UAV Camera Field of View Animation Sample...")
    for i in range(45, 20, -1):
        vis.UpdateHFOV(np.deg2rad(i))
    for i in range(45, 25, -1):
        vis.UpdateVFOV(np.deg2rad(i))

    plt.show()
