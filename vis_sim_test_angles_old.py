#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    Display_Intermediate_Outputs = True    # toggle display of all angles during calculation with pauses and outputs

    ###########################################################################
    ########################## CRITICAL ANGLE ANALYSIS ########################
    ###########################################################################
    # Initialize simulator
    x0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(20, 100)]])
    xt0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(0, 10)]])
    ypr = np.array([np.deg2rad(np.random.randint(-90, 90)), np.deg2rad(np.random.randint(-20, 20)),
                    np.deg2rad(np.random.randint(-20, 20))])
    ypr_g = np.array([np.deg2rad(np.random.randint(-90, 90)), np.deg2rad(np.random.randint(-20, 20)),
                      np.deg2rad(np.random.randint(-20, 20))])
    h_fov = np.deg2rad(np.random.randint(20, 60))
    v_fov = np.deg2rad(np.random.randint(20, 60))
    sim = UAV_simulator(x0, xt0, ypr, ypr_g, h_fov, v_fov)
    sim.UpdateX(x0, visualize=False)
    sim.UpdateTargetX(xt0, visualize=False)
    sim.UpdateYPR(ypr, visualize=False)
    sim.UpdateGimbalYPR(ypr_g, visualize=False)
    sim.UpdateHFOV(h_fov, visualize=False)
    sim.UpdateVFOV(v_fov, visualize=False)
    print("**UAV and Target Poses Randomized**")

    v_r = sim.R_gc @ sim.R_bg @ sim.e1
    v_p = sim.R_gc @ sim.R_bg @ sim.e2
    v_y = sim.R_gc @ sim.R_bg @ sim.e3
    v_yg = sim.R_gc @ sim.e3
    v_pg = sim.R_gc @ sim.e2
    v_rg = sim.R_gc @ sim.e1

    if Display_Intermediate_Outputs:
        display_time = 1.0  # display time for critical angle display
    all_constraints = False  # toggle to return all critical angle constraint solutions or just the closest on either side of the current orientation

    start = time.time()
    # critical roll angles
    sim.UpdatePertR(np.eye(3))
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_r, all=all_constraints)  # right, left, top, bottom
    angs_r = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("UAV Roll Constraints (First, Second, Third, Fourth): ", angs_r)
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang1[i]), visualize=True)
            print("Displaying First Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang2[i]), visualize=True)
            print("Displaying Second Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang3[i]), visualize=True)
            print("Displaying Third Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang4[i]), visualize=True)
            print("Displaying Fourth Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_r, ang4[i]), visualize=False)

    # critical pitch angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_p, all=all_constraints)  # right, left, top, bottom
    angs_p = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("UAV Pitch Constraints (First, Second, Third, Fourth): ", angs_p)
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang1[i]), visualize=True)
            print("Displaying First Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang2[i]), visualize=True)
            print("Displaying Second Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang3[i]), visualize=True)
            print("Displaying Third Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang4[i]), visualize=True)
            print("Displaying Fourth Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_p, ang4[i]), visualize=False)

    # critical yaw angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_y, all=all_constraints)  # right, left, top, bottom
    angs_y = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("UAV Yaw Constraints (First, Second, Third, Fourth): ", angs_y)
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang1[i]), visualize=True)
            print("Displaying First Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang2[i]), visualize=True)
            print("Displaying Second Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang3[i]), visualize=True)
            print("Displaying Third Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang4[i]), visualize=True)
            print("Displaying Fourth Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_y, ang4[i]), visualize=False)

    # critical gimbal yaw angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_yg, all=all_constraints)  # right, left, top, bottom
    angs_yg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("Gimbal Yaw Constraints (First, Second, Third, Fourth): ", angs_yg)
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang1[i]), visualize=True)
            print("Displaying First Gimbal Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang2[i]), visualize=True)
            print("Displaying Second Gimbal Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang3[i]), visualize=True)
            print("Displaying Third Gimbal Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang4[i]), visualize=True)
            print("Displaying Fourth Gimbal Yaw Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_yg, ang4[i]), visualize=False)

    # critical gimbal pitch angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
    angs_pg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("Gimbal Pitch Constraints (First, Second, Third, Fourth): ", angs_pg)
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang1[i]), visualize=True)
            print("Displaying First Gimbal Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang2[i]), visualize=True)
            print("Displaying Second Gimbal Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang3[i]), visualize=True)
            print("Displaying Third Gimbal Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang4[i]), visualize=True)
            print("Displaying Fourth Gimbal Pitch Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_pg, ang4[i]), visualize=False)

    # critical gimbal roll angles
    sim.UpdatePertR(np.eye(3), visualize=False)
    [ang1, ang2, ang3, ang4] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
    angs_rg = np.hstack([ang1, ang2, ang3, ang4]) * 180 / np.pi

    for i in range(len(ang1)):
        if Display_Intermediate_Outputs:
            print("Gimbal Roll Constraints (First, Second, Third, Fourth): ", angs_rg)
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang1[i]), visualize=True)
            print("Displaying First Gimbal Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang1[i]), visualize=False)
    for i in range(len(ang2)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang2[i]), visualize=True)
            print("Displaying Second Gimbal Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang2[i]), visualize=False)
    for i in range(len(ang3)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang3[i]), visualize=True)
            print("Displaying Third Gimbal Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang3[i]), visualize=False)
    for i in range(len(ang4)):
        if Display_Intermediate_Outputs:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang4[i]), visualize=True)
            print("Displaying Fourth Gimbal Roll Constraint...")
            plt.pause(display_time)
        else:
            sim.UpdatePertR(sim.axis_angle_to_R(v_rg, ang4[i]), visualize=False)

    # return to original sim location
    sim.UpdatePertR(np.eye(3))
    end = time.time()
    print("Calculation Time: ", end-start, "seconds")
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

    # determine critical action

    try:
        critical_axis = "Roll"
        critical_value = angs_r[np.argmin(np.abs(angs_r))]
    except:
        critical_axis = "No single rotation axis can achieve line of sight"
        critical_value = np.inf
    try:
        if np.abs(angs_p[np.argmin(np.abs(angs_p))]) < np.abs(critical_value):
            critical_axis = "Pitch"
            critical_value = angs_p[np.argmin(np.abs(angs_p))]
    except:
        pass
    try:
        if np.abs(angs_y[np.argmin(np.abs(angs_y))]) < np.abs(critical_value):
            critical_axis = "Yaw"
            critical_value = angs_y[np.argmin(np.abs(angs_y))]
    except:
        pass
    try:
        if np.abs(angs_yg[np.argmin(np.abs(angs_yg))]) < np.abs(critical_value):
            critical_axis = "Yaw Gimbal"
            critical_value = angs_yg[np.argmin(np.abs(angs_yg))]
    except:
        pass
    try:
        if np.abs(angs_pg[np.argmin(np.abs(angs_pg))]) < np.abs(critical_value):
            critical_axis = "Pitch Gimbal"
            critical_value = angs_pg[np.argmin(np.abs(angs_pg))]
    except:
        pass
    try:
        if np.abs(angs_rg[np.argmin(np.abs(angs_rg))]) < np.abs(critical_value):
            critical_axis = "Roll Gimbal"
            critical_value = angs_rg[np.argmin(np.abs(angs_rg))]
    except:
        pass

    if sim.in_sight:
        print("*** Target Currently in Line of Sight ***")
        print("Action resulting in fastest LOST line of sight is:", critical_axis, critical_value, "degrees")
    else:
        print("*** Target Currently NOT in Line of Sight ***")
        print("Action resulting in fastest ACHIEVED line of sight is:", critical_axis, critical_value, "degrees")


    plt.show()
