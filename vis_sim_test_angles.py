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

    v_combined = sim.GetAxes()
    v_y = v_combined[:, 0].reshape(-1, 1)
    v_p = v_combined[:, 1].reshape(-1, 1)
    v_r = v_combined[:, 2].reshape(-1, 1)
    v_yg = v_combined[:, 3].reshape(-1, 1)
    v_pg = v_combined[:, 4].reshape(-1, 1)
    v_rg = v_combined[:, 5].reshape(-1, 1)

    if Display_Intermediate_Outputs:
        display_time = 1.0  # display time for critical angle display
    all_constraints = False  # toggle to return all critical angle constraint solutions or just the closest on either side of the current orientation

    start = time.time()
    # critical roll angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_r, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_r = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("UAV Roll Constraints: ", angs)
            sim.UpdatePert_ypr_UAV(np.array([0, 0, angs[i]]), visualize=True)
            print("Displaying Roll Constraint: ", i+1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_UAV(np.array([0, 0, angs[i]]), visualize=False)

    # critical pitch angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_p, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_p = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("UAV Pitch Constraints: ", angs)
            sim.UpdatePert_ypr_UAV(np.array([0, angs[i], 0]), visualize=True)
            print("Displaying Pitch Constraint: ", i + 1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_UAV(np.array([0, angs[i], 0]), visualize=False)

    # critical yaw angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_y, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_y = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("UAV Yaw Constraints: ", angs)
            sim.UpdatePert_ypr_UAV(np.array([angs[i], 0, 0]), visualize=True)
            print("Displaying Yaw Constraint: ", i + 1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_UAV(np.array([angs[i], 0, 0]), visualize=False)

    # critical gimbal roll angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_rg, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_rg = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("Gimbal Roll Constraints: ", angs)
            sim.UpdatePert_ypr_Gimbal(np.array([0, 0, angs[i]]), visualize=True)
            print("Displaying Gimbal Roll Constraint: ", i+1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_Gimbal(np.array([0, 0, angs[i]]), visualize=False)

    # critical gimbal pitch angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_pg, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_pg = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("Gimbal Pitch Constraints: ", angs)
            sim.UpdatePert_ypr_Gimbal(np.array([0, angs[i], 0]), visualize=True)
            print("Displaying Gimbal Pitch Constraint: ", i+1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_Gimbal(np.array([0, angs[i], 0]), visualize=False)

    # critical gimbal yaw angles
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=False)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=False)
    angs = sim.CalculateCriticalAngles(v_yg, all=all_constraints)
    angs = angs[~np.isnan(angs)]
    angs_yg = angs * 180 / np.pi
    for i in range(len(angs)):
        if Display_Intermediate_Outputs:
            print("Gimbal Roll Constraints: ", angs)
            sim.UpdatePert_ypr_Gimbal(np.array([angs[i], 0, 0]), visualize=True)
            print("Displaying Gimbal Roll Constraint: ", i+1)
            plt.pause(display_time)
        else:
            sim.UpdatePert_ypr_Gimbal(np.array([angs[i], 0, 0]), visualize=False)

    # return to original sim location
    sim.UpdatePert_ypr_UAV(np.array([0, 0, 0]), visualize=True)
    sim.UpdatePert_ypr_Gimbal(np.array([0, 0, 0]), visualize=True)
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
