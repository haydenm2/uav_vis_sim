#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # ##########################################################################
    # ########################### Simulator Animations #########################
    # ##########################################################################
    # Initialize simulator
    # x0 = np.array([[-10], [0], [100]])
    # xt0 = np.array([[0], [0], [0]])
    # rpy = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    # ypr_g = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    # h_fov = np.deg2rad(45)
    # v_fov = np.deg2rad(45)
    # sim = UAV_simulator(x0, xt0, rpy, ypr_g, h_fov, v_fov)
    #
    # # example of cycling through UAV orientations
    # print("Displaying UAV Orientation Animation Sample...")
    # for i in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    # for j in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    # for k in range(0, 20, 1):
    #     sim.UpdateRPY(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
    #
    # # example of cycling through gimbal orientations
    # print("Displaying UAV Gimbal Orientation Animation Sample...")
    # for i in range(0, 30, 1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(0), np.deg2rad(0))))
    # for j in range(0, -10, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(0))))
    # for k in range(0, -10, -1):
    #     sim.UpdateGimbalYPR(np.array((np.deg2rad(i), np.deg2rad(j), np.deg2rad(k))))
    #
    # # example of cycling through UAV positions
    # print("Displaying UAV Position Animation Sample...")
    # for i in range(-10, 20, 1):
    #     sim.UpdateX(np.array((i, 0, 100)))
    # for j in range(0, 20, 1):
    #     sim.UpdateX(np.array((i, j, 100)))
    # for k in range(100, 40, -1):
    #     sim.UpdateX(np.array((i, j, k)))
    #
    # # example of cycling through target positions
    # print("Displaying Target Position Animation Sample...")
    # for i in range(0, 3, 1):
    #     sim.UpdateTargetX(np.array((i, 0, 0)))
    # for j in range(0, 25, 1):
    #     sim.UpdateTargetX(np.array((i, j, 0)))
    # for k in range(0, 10, 1):
    #     sim.UpdateTargetX(np.array((i, j, k)))
    #
    # # example of cycling through camera FOV angles
    # print("Displaying UAV Camera Field of View Animation Sample...")
    # for i in range(45, 20, -1):
    #     sim.UpdateHFOV(np.deg2rad(i))
    # for i in range(45, 25, -1):
    #     sim.UpdateVFOV(np.deg2rad(i))


    ###########################################################################
    ########################## CRITICAL ANGLE ANALYSIS ########################
    ###########################################################################
    # Initialize simulator
    x0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(20, 100)]])
    xt0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(0, 10)]])
    rpy = np.array([np.deg2rad(np.random.randint(-20, 20)), np.deg2rad(np.random.randint(-20, 20)),
                    np.deg2rad(np.random.randint(-20, 20))])
    ypr_g = np.array([np.deg2rad(np.random.randint(-20, 20)), np.deg2rad(np.random.randint(-20, 20)),
                      np.deg2rad(np.random.randint(-20, 20))])
    h_fov = np.deg2rad(np.random.randint(20, 60))
    v_fov = np.deg2rad(np.random.randint(20, 60))
    sim = UAV_simulator(x0, xt0, rpy, ypr_g, h_fov, v_fov)

    v_r = sim.R_gc @ sim.R_bg @ sim.e1
    v_p = sim.R_gc @ sim.R_bg @ sim.e2
    v_y = sim.R_gc @ sim.R_bg @ sim.e3
    v_yg = sim.R_gc @ sim.e3
    v_pg = sim.R_gc @ sim.e2
    v_rg = sim.R_gc @ sim.e1

    display_time = 1.0
    all_constraints = False

    # critical roll angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_r, all=all_constraints)  # right, left, top, bottom
    angs_r = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("UAV Roll Constraints (First, Second, Third, Fourth): ", angs_r)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang1[i]))
        print("Displaying First Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang2[i]))
        print("Displaying Second Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang3[i]))
        print("Displaying Third Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang4[i]))
        print("Displaying Fourth Roll Constraint...")
        plt.pause(display_time)

    # critical pitch angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_p, all=all_constraints)  # right, left, top, bottom
    angs_p = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("UAV Pitch Constraints (First, Second, Third, Fourth): ", angs_p)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang1[i]))
        print("Displaying First Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang2[i]))
        print("Displaying Second Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang3[i]))
        print("Displaying Third Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang4[i]))
        print("Displaying Fourth Pitch Constraint...")
        plt.pause(display_time)

    # critical yaw angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_y, all=all_constraints)  # right, left, top, bottom
    angs_y = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("UAV Yaw Constraints (First, Second, Third, Fourth): ", angs_y)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang1[i]))
        print("Displaying First Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang2[i]))
        print("Displaying Second Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang3[i]))
        print("Displaying Third Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang4[i]))
        print("Displaying Fourth Yaw Constraint...")
        plt.pause(display_time)

    # critical gimbal yaw angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_yg, all=all_constraints)  # right, left, top, bottom
    angs_yg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("Gimbal Yaw Constraints (First, Second, Third, Fourth): ", angs_yg)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang1[i]))
        print("Displaying First Gimbal Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang2[i]))
        print("Displaying Second Gimbal Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang3[i]))
        print("Displaying Third Gimbal Yaw Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang4[i]))
        print("Displaying Fourth Gimbal Yaw Constraint...")
        plt.pause(display_time)

    # critical gimbal pitch angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
    angs_pg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("Gimbal Pitch Constraints (First, Second, Third, Fourth): ", angs_pg)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang1[i]))
        print("Displaying First Gimbal Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang2[i]))
        print("Displaying Second Gimbal Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang3[i]))
        print("Displaying Third Gimbal Pitch Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang4[i]))
        print("Displaying Fourth Gimbal Pitch Constraint...")
        plt.pause(display_time)

    # critical gimbal roll angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
    angs_rg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi
    print("Gimbal Roll Constraints (First, Second, Third, Fourth): ", angs_rg)

    for i in range(len(ang1)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang1[i]))
        print("Displaying First Gimbal Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang2)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang2[i]))
        print("Displaying Second Gimbal Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang3)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang3[i]))
        print("Displaying Third Gimbal Roll Constraint...")
        plt.pause(display_time)
    for i in range(len(ang4)):
        sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang4[i]))
        print("Displaying Fourth Gimbal Roll Constraint...")
        plt.pause(display_time)

    # return to original sim location
    sim.UpdatePertR(np.eye(3))
    print("Displaying Original UAV Pose")

    print("\n-------------------------------------------------------------------")
    print("------------------------- UAV Constraints -------------------------")
    print("-------------------------------------------------------------------")
    print("Roll: ", angs_r)
    print("Pitch: ", angs_p)
    print("Yaw: ", angs_y)
    print("Yaw Gimbal: ", angs_yg)
    print("Pitch Gimbal: ", angs_pg)
    print("Roll Gimbal: ", angs_rg)
    if sim.in_sight:
        print("*** Target Currently in Line of Sight ***")
    else:
        print("*** Target Currently NOT in Line of Sight ***")
    plt.show()
