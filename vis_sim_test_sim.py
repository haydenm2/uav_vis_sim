#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # Initialize simulator
    x0 = np.array([[-10], [0], [100]])
    xt0 = np.array([[0], [0], [0]])
    ypr = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    ypr_g = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    h_fov = np.deg2rad(45)
    v_fov = np.deg2rad(45)
    sim = UAV_simulator(x0, xt0, ypr, ypr_g, h_fov, v_fov)

    ##########################################################################
    ########################### Simulator Animations #########################
    ##########################################################################
    # example of cycling through UAV orientations
    print("Displaying UAV Orientation Animation Sample...")
    for i in range(0, 20, 1):
        sim.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    for j in range(0, 20, 1):
        sim.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    for k in range(0, 20, 1):
        sim.UpdateYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))

    # example of cycling through gimbal orientations
    print("Displaying UAV Gimbal Orientation Animation Sample...")
    for i in range(0, 30, 1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    for j in range(0, -10, -1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    for k in range(0, -10, -1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))

    # example of cycling through UAV positions
    print("Displaying UAV Position Animation Sample...")
    for i in range(-10, 20, 1):
        sim.UpdateX(np.array((i, 0, 100)))
    for j in range(0, 20, 1):
        sim.UpdateX(np.array((i, j, 100)))
    for k in range(100, 40, -1):
        sim.UpdateX(np.array((i, j, k)))

    # example of cycling through target positions
    print("Displaying Target Position Animation Sample...")
    for i in range(0, 3, 1):
        sim.UpdateTargetX(np.array((i, 0, 0)))
    for j in range(0, 25, 1):
        sim.UpdateTargetX(np.array((i, j, 0)))
    for k in range(0, 10, 1):
        sim.UpdateTargetX(np.array((i, j, k)))

    # example of cycling through camera FOV angles
    print("Displaying UAV Camera Field of View Animation Sample...")
    for i in range(45, 20, -1):
        sim.UpdateHFOV(np.deg2rad(i))
    for i in range(45, 25, -1):
        sim.UpdateVFOV(np.deg2rad(i))

    plt.show()
