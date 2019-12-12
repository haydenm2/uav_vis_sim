#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # init simulator
    sim = UAV_simulator(np.array([[-10], [0], [40]]))

    # example of cycling through UAV orientations
    for i in range(0, 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
        sim.UpdatePlots()
    for j in range(0, 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
        sim.UpdatePlots()
    for k in range(0, 20, 1):
        sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
        sim.UpdatePlots()

    # example of cycling through gimbal orientations
    for i in range(0, 30, 1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
        sim.UpdatePlots()
    for j in range(0, -10, -1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
        sim.UpdatePlots()
    for k in range(0, -10, -1):
        sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
        sim.UpdatePlots()

    # example of cycling through UAV positions
    for i in range(-10, 20, 1):
        sim.UpdateX(np.array((i, 0, 40)))
        sim.UpdatePlots()
    for j in range(0, 20, 1):
        sim.UpdateX(np.array((i, j, 40)))
        sim.UpdatePlots()
    for k in range(40, 80, 1):
        sim.UpdateX(np.array((i, j, k)))
        sim.UpdatePlots()

    # example of cycling through target positions
    for i in range(0, -10, -1):
        sim.UpdateTargetX(np.array((i, 0, 0)))
        sim.UpdatePlots()
    for j in range(0, 10, 1):
        sim.UpdateTargetX(np.array((i, j, 0)))
        sim.UpdatePlots()
    for k in range(0, 10, 1):
        sim.UpdateTargetX(np.array((i, j, k)))
        sim.UpdatePlots()

    # example of cycling through camera FOV angles
    for i in range(45, 15, -1):
        sim.UpdateHFOV(np.deg2rad(i))
        sim.UpdatePlots()
    for i in range(45, 20, -1):
        sim.UpdateVFOV(np.deg2rad(i))
        sim.UpdatePlots()

    plt.show()
