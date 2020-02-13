#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt
import time

def sub_vstack(a):
    if len(a[0]) == 0:
        b = np.array([0.0])
    else:
        b = a[0]
    for i in a[1:]:
        try:
            b = np.vstack([b, i])
        except:
            b = np.vstack([b, np.array([0.0])])
    return b


if __name__ == "__main__":

    # plotting vectors
    t_end = 10.0
    dt = 0.1
    _t = np.linspace(0.0, t_end, num=t_end / dt)
    z_line = _t*0.0  # zero reference line

    ##########################################################################
    ########################### Live Plotting Bounds #########################
    ##########################################################################

    show_sim = False                    # toggle display of simulated course
    all_constraints = False             # toggle to return all critical angle constraint solutions or just the closest on either side of the current orientation

    # Initialize simulator
    x0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(20, 100)]])
    xt0 = np.array([[np.random.randint(-20, 20)], [np.random.randint(-20, 20)], [np.random.randint(0, 10)]])
    ypr0 = np.array([np.deg2rad(np.random.randint(-90, 90)), np.deg2rad(np.random.randint(-20, 20)),
                    np.deg2rad(np.random.randint(-20, 20))])
    ypr_g0 = np.array([np.deg2rad(np.random.randint(-90, 90)), np.deg2rad(np.random.randint(-20, 20)),
                      np.deg2rad(np.random.randint(-20, 20))])
    h_fov = np.deg2rad(np.random.randint(20, 60))
    v_fov = np.deg2rad(np.random.randint(20, 60))

    sim = UAV_simulator(x0, xt0, ypr0, ypr_g0, h_fov, v_fov)

    v_r = sim.R_gc @ sim.R_bg @ sim.e1
    v_p = sim.R_gc @ sim.R_bg @ sim.e2
    v_y = sim.R_gc @ sim.R_bg @ sim.e3
    v_yg = sim.R_gc @ sim.e3
    v_pg = sim.R_gc @ sim.e2
    v_rg = sim.R_gc @ sim.e1

    [ang1_r, ang2_r, ang3_r, ang4_r] = sim.CalculateCriticalAngles(v_r, all=all_constraints)  # right, left, top, bottom
    angs_r = sub_vstack([ang1_r, ang2_r, ang3_r, ang4_r])
    [ang1_p, ang2_p, ang3_p, ang4_p] = sim.CalculateCriticalAngles(v_p, all=all_constraints)  # right, left, top, bottom
    angs_p = sub_vstack([ang1_p, ang2_p, ang3_p, ang4_p])
    [ang1_y, ang2_y, ang3_y, ang4_y] = sim.CalculateCriticalAngles(v_y, all=all_constraints)  # right, left, top, bottom
    angs_y = sub_vstack([ang1_y, ang2_y, ang3_y, ang4_y])
    [ang1_rg, ang2_rg, ang3_rg, ang4_rg] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
    angs_rg = sub_vstack([ang1_rg, ang2_rg, ang3_rg, ang4_rg])
    [ang1_pg, ang2_pg, ang3_pg, ang4_pg] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
    angs_pg = sub_vstack([ang1_pg, ang2_pg, ang3_pg, ang4_pg])
    [ang1_yg, ang2_yg, ang3_yg, ang4_yg] = sim.CalculateCriticalAngles(v_yg, all=all_constraints)  # right, left, top, bottom
    angs_yg = sub_vstack([ang1_yg, ang2_yg, ang3_yg, ang4_yg])

    _in_sight = np.array([sim.in_sight])
    _pi = sim.p_i
    _a1 = angs_r   # axis 1 limits
    _a2 = angs_p   # axis 2 limits
    _a3 = angs_y   # axis 3 limits
    _a4 = angs_rg  # axis 4 limits
    _a5 = angs_pg  # axis 5 limits
    _a6 = angs_yg  # axis 6 limits

    x = np.array([20.0*np.sin(_t), 20.0*np.cos(_t), 2.0*np.sin(_t)]) + x0.reshape(-1, 1)
    xt = np.array([-1.0*np.sin(_t), -3.0*np.cos(_t), np.sin(_t) + 1.0]) + xt0.reshape(-1, 1)
    ypr = np.array([np.deg2rad(90*np.cos(_t)), np.deg2rad(20.0*np.sin(_t/3.0)), np.deg2rad(20.0*np.sin(_t/2.0))]) + ypr0.reshape(-1, 1)
    ypr_g = np.array([np.deg2rad(90*np.sin(_t)), np.deg2rad(20.0*np.sin(_t*2.0)), np.deg2rad(20.0*np.sin(_t/2.0))]) + ypr_g0.reshape(-1, 1)

    for i in range(len(x[0])):
        if i == 0:
            continue  #skip redundant first term
        sim.UpdateX(x[:, i], visualize=show_sim)
        sim.UpdateTargetX(xt[:, i], visualize=show_sim)
        sim.UpdateYPR(ypr[:, i], visualize=show_sim)
        sim.UpdateGimbalYPR(ypr_g[:, i], visualize=show_sim)

        v_r = sim.R_gc @ sim.R_bg @ sim.e1
        v_p = sim.R_gc @ sim.R_bg @ sim.e2
        v_y = sim.R_gc @ sim.R_bg @ sim.e3
        v_yg = sim.R_gc @ sim.e3
        v_pg = sim.R_gc @ sim.e2
        v_rg = sim.R_gc @ sim.e1

        [ang1_r, ang2_r, ang3_r, ang4_r] = sim.CalculateCriticalAngles(v_r, all=all_constraints)  # right, left, top, bottom
        angs_r = sub_vstack([ang1_r, ang2_r, ang3_r, ang4_r])
        [ang1_p, ang2_p, ang3_p, ang4_p] = sim.CalculateCriticalAngles(v_p, all=all_constraints)  # right, left, top, bottom
        angs_p = sub_vstack([ang1_p, ang2_p, ang3_p, ang4_p])
        [ang1_y, ang2_y, ang3_y, ang4_y] = sim.CalculateCriticalAngles(v_y, all=all_constraints)  # right, left, top, bottom
        angs_y = sub_vstack([ang1_y, ang2_y, ang3_y, ang4_y])
        [ang1_rg, ang2_rg, ang3_rg, ang4_rg] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
        angs_rg = sub_vstack([ang1_rg, ang2_rg, ang3_rg, ang4_rg])
        [ang1_pg, ang2_pg, ang3_pg, ang4_pg] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
        angs_pg = sub_vstack([ang1_pg, ang2_pg, ang3_pg, ang4_pg])
        [ang1_yg, ang2_yg, ang3_yg, ang4_yg] = sim.CalculateCriticalAngles(v_yg, all=all_constraints)  # right, left, top, bottom
        angs_yg = sub_vstack([ang1_yg, ang2_yg, ang3_yg, ang4_yg])

        _in_sight = np.hstack((_in_sight, sim.in_sight))
        _pi = np.hstack((_pi, sim.p_i))
        _a1 = np.hstack((_a1, angs_r))
        _a2 = np.hstack((_a2, angs_p))
        _a3 = np.hstack((_a3, angs_y))
        _a4 = np.hstack((_a4, angs_rg))
        _a5 = np.hstack((_a5, angs_pg))
        _a6 = np.hstack((_a6, angs_yg))

    # convert angle limits to degrees
    _a1 *= 180.0/np.pi
    _a2 *= 180.0/np.pi
    _a3 *= 180.0/np.pi
    _a4 *= 180.0/np.pi
    _a5 *= 180.0/np.pi
    _a6 *= 180.0/np.pi

    plt.figure(3)

    # plot rotation bounds for roll
    plt.subplot(231)
    for j in range(len(_a1[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a1[0, j], _a1[1, j]]), 'b')
        else:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a1[0, j], _a1[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Roll Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # plot rotation bounds for pitch
    plt.subplot(232)
    for j in range(len(_a2[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a2[0, j], _a2[1, j]]), 'b')
        else:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a2[0, j], _a2[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Pitch Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # plot rotation bounds for yaw
    plt.subplot(233)
    for j in range(len(_a3[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a3[0, j], _a3[1, j]]), 'b')
        else:
            if np.minimum(_a3[0, j], _a3[1, j]) < 0 and np.maximum(_a3[0, j], _a3[1, j]) > 0:  # wrapping bound angles
                plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a3[0, j], _a3[1, j]), -180.0]), 'r')
                plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a3[0, j], _a3[1, j]), 180.0]), 'r')
            else:
                plt.plot(np.array([_t[j], _t[j]]), np.array([_a3[0, j], _a3[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Yaw Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # plot rotation bounds for roll gimbal
    plt.subplot(234)
    for j in range(len(_a4[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a4[0, j], _a4[1, j]]), 'b')
        else:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a4[0, j], _a4[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Roll Gimbal Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # plot rotation bounds for pitch gimbal
    plt.subplot(235)
    for j in range(len(_a5[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a5[0, j], _a5[1, j]]), 'b')
        else:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a5[0, j], _a5[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Pitch Gimbal Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # plot rotation bounds for yaw gimbal
    plt.subplot(236)
    for j in range(len(_a6[0])):
        if _in_sight[j]:
            plt.plot(np.array([_t[j], _t[j]]), np.array([_a6[0, j], _a6[1, j]]), 'b')
        else:
            if np.minimum(_a6[0, j], _a6[1, j]) < 0 and np.maximum(_a6[0, j], _a6[1, j]) > 0:  # wrapping bound angles
                plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a6[0, j], _a6[1, j]), -180.0]), 'r')
                plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a6[0, j], _a6[1, j]), 180.0]), 'r')
            else:
                plt.plot(np.array([_t[j], _t[j]]), np.array([_a6[0, j], _a6[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.title('Yaw Gimbal Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    plt.pause(0.1)

    plt.show()
