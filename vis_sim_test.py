#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Init simulator
    x0 = np.array([[-10], [0], [1000]])
    xt0 = np.array([[0], [0], [0]])
    rpy = np.array([np.deg2rad(10), np.deg2rad(-8), np.deg2rad(17)])
    ypr_g = np.array([np.deg2rad(20), np.deg2rad(15), np.deg2rad(6)])
    h_fov = np.deg2rad(45)
    v_fov = np.deg2rad(45)
    sim = UAV_simulator(x0, xt0, rpy, ypr_g, h_fov, v_fov)

    ##########################################################################
    ########################### Simulator Animations #########################
    ##########################################################################

    # example of cycling through UAV orientations
    print("Displaying UAV Orientation Animation Sample...")
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
    print("Displaying UAV Gimbal Orientation Animation Sample...")
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
    print("Displaying UAV Position Animation Sample...")
    for i in range(-10, 20, 1):
        sim.UpdateX(np.array((i, 0, 100)))
        sim.UpdatePlots()
    for j in range(0, 20, 1):
        sim.UpdateX(np.array((i, j, 100)))
        sim.UpdatePlots()
    for k in range(100, 40, -1):
        sim.UpdateX(np.array((i, j, k)))
        sim.UpdatePlots()

    # example of cycling through target positions
    print("Displaying Target Position Animation Sample...")
    for i in range(0, 3, 1):
        sim.UpdateTargetX(np.array((i, 0, 0)))
        sim.UpdatePlots()
    for j in range(0, 25, 1):
        sim.UpdateTargetX(np.array((i, j, 0)))
        sim.UpdatePlots()
    for k in range(0, 10, 1):
        sim.UpdateTargetX(np.array((i, j, k)))
        sim.UpdatePlots()

    # example of cycling through camera FOV angles
    print("Displaying UAV Camera Field of View Animation Sample...")
    for i in range(45, 20, -1):
        sim.UpdateHFOV(np.deg2rad(i))
        sim.UpdatePlots()
    for i in range(45, 25, -1):
        sim.UpdateVFOV(np.deg2rad(i))
        sim.UpdatePlots()

    ###########################################################################
    ########################## CRITICAL ANGLE ANALYSIS ########################
    ###########################################################################

    v_r = sim.R_gc @ sim.R_bg @ sim.e1
    v_p = sim.R_gc @ sim.R_bg @ sim.e2
    v_y = sim.R_gc @ sim.R_bg @ sim.e3
    v_yg = sim.R_gc @ sim.e3
    v_pg = sim.R_gc @ sim.e2
    v_rg = sim.R_gc @ sim.e1

    # critical roll angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_r)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang1))
    sim.UpdatePlots()
    print("Displaying Right Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang2))
    sim.UpdatePlots()
    print("Displaying Left Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang3))
    sim.UpdatePlots()
    print("Displaying Top Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Roll Constraint...")
    plt.pause(1.0)

    # critical pitch angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_p)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang1))
    sim.UpdatePlots()
    print("Displaying Right Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang2))
    sim.UpdatePlots()
    print("Displaying Left Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang3))
    sim.UpdatePlots()
    print("Displaying Top Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Pitch Constraint...")
    plt.pause(1.0)

    # critical yaw angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_y)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang1))
    sim.UpdatePlots()
    print("Displaying Right Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang2))
    sim.UpdatePlots()
    print("Displaying Left Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang3))
    sim.UpdatePlots()
    print("Displaying Top Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Yaw Constraint...")
    plt.pause(1.0)

    # critical gimbal yaw angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_yg)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang1))
    sim.UpdatePlots()
    print("Displaying Right Gimbal Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang2))
    sim.UpdatePlots()
    print("Displaying Left Gimbal Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang3))
    sim.UpdatePlots()
    print("Displaying Top Gimbal Yaw Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Gimbal Yaw Constraint...")
    plt.pause(1.0)

    # critical gimbal pitch angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_pg)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang1))
    sim.UpdatePlots()
    print("Displaying Right Gimbal Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang2))
    sim.UpdatePlots()
    print("Displaying Left Gimbal Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang3))
    sim.UpdatePlots()
    print("Displaying Top Gimbal Pitch Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Gimbal Pitch Constraint...")
    plt.pause(1.0)

    # critical gimbal roll angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_rg)  # right, left, top, bottom
    print(ang1, ang2, ang3, ang4)

    sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang1))
    sim.UpdatePlots()
    print("Displaying Right Gimbal Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang2))
    sim.UpdatePlots()
    print("Displaying Left Gimbal Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang3))
    sim.UpdatePlots()
    print("Displaying Top Gimbal Roll Constraint...")
    plt.pause(1.0)
    sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang4))
    sim.UpdatePlots()
    print("Displaying Bottom Gimbal Roll Constraint...")
    plt.pause(1.0)

    plt.show()
