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

def sub_vstack2(a):
    if len(a[0]) == 0:
        b = np.array([[0.0], [0.0]])
    elif len(a[0]) == 1:
        b = np.array([[a[0].item(0)], [0.0]])
    else:
        b = a[0].reshape(-1, 1)
    for i in a[1:]:
        if len(i) == 0:
            b = np.vstack([b, np.array([[0.0], [0.0]])])
        elif len(i) == 1:
            b = np.vstack([b, np.array([[i.item(0)], [0.0]])])
        else:
            b = np.vstack([b, i.reshape(-1, 1)])
    return b

def zone_plot(in_sight, t, pts, bounds=180):
    pts2 = np.vstack([np.array([-bounds]), pts, np.array([bounds])])
    f_pts = pts2[pts2 != np.inf]
    mid_ind = len(f_pts[f_pts < 0])
    if in_sight:
        if np.mod(mid_ind, 2):  #odd middle index
            for i in range(int((len(f_pts)-1)/2) + 1): # counting every other space between points
                if 2*i+1 == mid_ind:
                    plt.plot(np.array([t, t]), np.array([f_pts[2*i], f_pts[2*i+1]]), 'b')
                else:
                    plt.plot(np.array([t, t]), np.array([f_pts[2*i], f_pts[2*i+1]]), 'r')
        else: #even middle index
            for ii in range(int((len(f_pts)-1)/2)):
                if 2*(ii+1) == mid_ind:
                    plt.plot(np.array([t, t]), np.array([f_pts[2*(ii+1)-1], f_pts[2*(ii+1)]]), 'b')
                else:
                    plt.plot(np.array([t, t]), np.array([f_pts[2*(ii+1)-1], f_pts[2*(ii+1)]]), 'r')
    else:
        if np.mod(mid_ind, 2):  # odd middle index
            for iii in range(int((len(f_pts) - 1) / 2)):
                plt.plot(np.array([t, t]), np.array([f_pts[2 * (iii + 1) - 1], f_pts[2 * (iii + 1)]]), 'r')
        else:  # even middle index
            for iiii in range(int((len(f_pts) - 1) / 2) + 1):  # counting every other space between points
                plt.plot(np.array([t, t]), np.array([f_pts[2 * iiii], f_pts[2 * iiii + 1]]), 'r')  #TODO: maybe skipping last set??



if __name__ == "__main__":

    # plotting vectors
    t_end = 10.0
    dt = 0.1
    _t = np.linspace(0.0, t_end, num=int(t_end / dt))
    z_line = _t*0.0  # zero reference line
    u_line = z_line + 180.0  # upper limit reference line
    l_line = z_line - 180.0  # lower limit reference line

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
    angs_y = sub_vstack2([ang1_y, ang2_y, ang3_y, ang4_y])
    [ang1_rg, ang2_rg, ang3_rg, ang4_rg] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
    angs_rg = sub_vstack([ang1_rg, ang2_rg, ang3_rg, ang4_rg])
    [ang1_pg, ang2_pg, ang3_pg, ang4_pg] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
    angs_pg = sub_vstack([ang1_pg, ang2_pg, ang3_pg, ang4_pg])
    [ang1_yg, ang2_yg, ang3_yg, ang4_yg] = sim.CalculateCriticalAngles(v_yg, all=all_constraints)  # right, left, top, bottom
    angs_yg = sub_vstack2([ang1_yg, ang2_yg, ang3_yg, ang4_yg])

    _in_sight = np.array([sim.in_sight])
    _pi = sim.p_i
    _a1 = angs_r   # axis 1 limits
    _a2 = angs_p   # axis 2 limits
    _a3 = angs_y   # axis 3 limits
    _a4 = angs_rg  # axis 4 limits
    _a5 = angs_pg  # axis 5 limits
    _a6 = angs_yg  # axis 6 limits

    x = np.array([20.0*np.sin(_t), 20.0*np.sin(_t), 2.0*np.sin(_t)]) + x0.reshape(-1, 1)
    xt = np.array([-1.0*np.sin(_t), -3.0*np.sin(_t), np.sin(_t) + 1.0]) + xt0.reshape(-1, 1)
    ypr = np.array([np.deg2rad(90*np.sin(_t)), np.deg2rad(20.0*np.sin(_t/3.0)), np.deg2rad(20.0*np.sin(_t/2.0))]) + ypr0.reshape(-1, 1)
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
        [ang1_y, ang2_y, ang3_y, ang4_y] = sim.CalculateCriticalAngles(v_y, all=True)  # right, left, top, bottom
        angs_y = sub_vstack2([ang1_y, ang2_y, ang3_y, ang4_y])
        [ang1_rg, ang2_rg, ang3_rg, ang4_rg] = sim.CalculateCriticalAngles(v_rg, all=all_constraints)  # right, left, top, bottom
        angs_rg = sub_vstack([ang1_rg, ang2_rg, ang3_rg, ang4_rg])
        [ang1_pg, ang2_pg, ang3_pg, ang4_pg] = sim.CalculateCriticalAngles(v_pg, all=all_constraints)  # right, left, top, bottom
        angs_pg = sub_vstack([ang1_pg, ang2_pg, ang3_pg, ang4_pg])
        [ang1_yg, ang2_yg, ang3_yg, ang4_yg] = sim.CalculateCriticalAngles(v_yg, all=True)  # right, left, top, bottom
        angs_yg = sub_vstack2([ang1_yg, ang2_yg, ang3_yg, ang4_yg])

        _in_sight = np.hstack((_in_sight, sim.in_sight))
        _pi = np.hstack((_pi, sim.p_i))

        _a1 = np.hstack((_a1, angs_r))
        _a1[_a1 == 0] = np.inf
        _a1 = np.sort(_a1, axis=0)

        _a2 = np.hstack((_a2, angs_p))
        _a2[_a2 == 0] = np.inf
        _a2 = np.sort(_a2, axis=0)

        _a3 = np.hstack((_a3, angs_y))
        _a3[_a3 == 0] = np.inf
        _a3 = np.sort(_a3, axis=0)

        _a4 = np.hstack((_a4, angs_rg))
        _a4[_a4 == 0] = np.inf
        _a4 = np.sort(_a4, axis=0)

        _a5 = np.hstack((_a5, angs_pg))
        _a5[_a5 == 0] = np.inf
        _a5 = np.sort(_a5, axis=0)

        _a6 = np.hstack((_a6, angs_yg))
        _a6[_a6 == 0] = np.inf
        _a6 = np.sort(_a6, axis=0)

    # convert angle limits to degrees
    _a1 *= 180.0/np.pi
    _a2 *= 180.0/np.pi
    _a3 *= 180.0/np.pi
    _a4 *= 180.0/np.pi
    _a5 *= 180.0/np.pi
    _a6 *= 180.0/np.pi

    ########################### Plot critical boundaries over time #########################


    seperate_plots = True
    plt.figure(3)
    plt.tight_layout()
    if seperate_plots:
        plt.suptitle('Rotational Bounds about UAV Axes of Motion')
    else:
        plt.suptitle('Rotational Bounds about Standard Axes of Motion')

    # -------------------- plot rotation bounds for roll --------------------
    if seperate_plots:
        plt.subplot(311)
    else:
        plt.subplot(321)
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'b')  # dummy for legend
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'r')  # dummy for legend
    for j in range(len(_a1[0])):
        zone_plot(_in_sight[j], _t[j], _a1[:, j].reshape(-1, 1))
    #     if _in_sight[j]:
    #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a1[0, j], _a1[1, j]]), 'b')
    #     else:
    #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a1[0, j], _a1[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Roll Axis Constraint Transform')
    plt.ylabel('Rotation Bounds (deg)')
    plt.legend(("Visibility Region (In Region)", "Visibility Region (Out of Region)"))

    # -------------------- plot rotation bounds for pitch --------------------
    if seperate_plots:
        plt.subplot(312)
    else:
        plt.subplot(323)
    for j in range(len(_a2[0])):
        zone_plot(_in_sight[j], _t[j], _a2[:, j].reshape(-1, 1))
        # if _in_sight[j]:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a2[0, j], _a2[1, j]]), 'b')
        # else:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a2[0, j], _a2[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Pitch Axis Constraint Transform')
    plt.ylabel('Rotation Bounds (deg)')

    # -------------------- plot rotation bounds for yaw --------------------
    if seperate_plots:
        plt.subplot(313)
    else:
        plt.subplot(325)
    for j in range(len(_a3[0])):
        zone_plot(_in_sight[j], _t[j], _a3[:, j].reshape(-1, 1))
        # if _in_sight[j]:
        #     if _a3[0, j] == 0 and _a3[1, j] == 0:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([-180.0, 180.0]), 'b')
        #         continue
        #     if _a3[0, j]/_a3[1, j] > 0:  # wrapping bound angles
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a3[0, j], _a3[1, j]), -180.0]), 'b')
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a3[0, j], _a3[1, j]), 180.0]), 'b')
        #     else:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a3[0, j], _a3[1, j]]), 'b')
        # else:
        #     if np.minimum(_a3[0, j], _a3[1, j]) < 0 and np.maximum(_a3[0, j], _a3[1, j]) > 0:  # wrapping bound angles
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a3[0, j], _a3[1, j]), -180.0]), 'r')
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a3[0, j], _a3[1, j]), 180.0]), 'r')
        #     else:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a3[0, j], _a3[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Yaw Axis Constraint Transform')
    plt.xlabel('Time (s)')
    plt.ylabel('Rotation Bounds (deg)')

    # -------------------- plot rotation bounds for roll gimbal --------------------
    if seperate_plots:
        plt.figure(4)
        plt.tight_layout()
        plt.suptitle('Rotational Bounds about Gimbal Axes of Motion')
        plt.subplot(311)
        plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'b')  # dummy for legend
        plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'r')  # dummy for legend
        plt.ylabel('Rotation Bounds (deg)')
        plt.legend(("Visibility Region (In Region)", "Visibility Region (Out of Region)"))
    else:
        plt.subplot(322)
    for j in range(len(_a4[0])):
        zone_plot(_in_sight[j], _t[j], _a4[:, j].reshape(-1, 1))
        # if _in_sight[j]:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a4[0, j], _a4[1, j]]), 'b')
        # else:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a4[0, j], _a4[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Roll Gimbal Axis Constraint Transform')

    # -------------------- plot rotation bounds for pitch gimbal --------------------
    if seperate_plots:
        plt.subplot(312)
        plt.ylabel('Rotation Bounds (deg)')
    else:
        plt.subplot(324)
    for j in range(len(_a5[0])):
        zone_plot(_in_sight[j], _t[j], _a5[:, j].reshape(-1, 1))
        # if _in_sight[j]:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a5[0, j], _a5[1, j]]), 'b')
        # else:
        #     plt.plot(np.array([_t[j], _t[j]]), np.array([_a5[0, j], _a5[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Pitch Gimbal Axis Constraint Transform')

    # -------------------- plot rotation bounds for yaw gimbal --------------------
    if seperate_plots:
        plt.subplot(313)
        plt.ylabel('Rotation Bounds (deg)')
    else:
        plt.subplot(326)
    for j in range(len(_a6[0])):
        zone_plot(_in_sight[j], _t[j], _a6[:, j].reshape(-1, 1))
        # if _in_sight[j]:
        #     if _a6[0, j] == 0 and _a6[1, j] == 0:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([-180.0, 180.0]), 'b')
        #         continue
        #     if _a6[0, j]/_a6[1, j] > 0:  # wrapping bound angles
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a6[0, j], _a6[1, j]), -180.0]), 'b')
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a6[0, j], _a6[1, j]), 180.0]), 'b')
        #     else:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a6[0, j], _a6[1, j]]), 'b')
        # else:
        #     if np.minimum(_a6[0, j], _a6[1, j]) < 0 and np.maximum(_a6[0, j], _a6[1, j]) > 0:  # wrapping bound angles
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.minimum(_a6[0, j], _a6[1, j]), -180.0]), 'r')
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([np.maximum(_a6[0, j], _a6[1, j]), 180.0]), 'r')
        #     else:
        #         plt.plot(np.array([_t[j], _t[j]]), np.array([_a6[0, j], _a6[1, j]]), 'r')
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Yaw Gimbal Axis Constraint Transform')
    plt.xlabel('Time (s)')

    plt.pause(0.1)

    ########################### Plot Target projetion path #########################

    plt.figure(5)
    if _in_sight[0]:
        pp1 = plt.plot(_pi[1, 0], _pi[0, 0], markersize=15.0, markerfacecolor='b', marker='X', linestyle='None', markeredgecolor='k')
    else:
        pp1 = plt.plot(_pi[1, 0], _pi[0, 0], markersize=15.0, markerfacecolor='r', marker='X', linestyle='None', markeredgecolor='k')
    pp2 = plt.plot(_pi[1, 1:][_in_sight[1:]], _pi[0, 1:][_in_sight[1:]], markersize=5.0, markerfacecolor='b', marker='o', linestyle='None', markeredgecolor='k')
    pp3 = plt.plot(_pi[1, 1:][~_in_sight[1:]], _pi[0, 1:][~_in_sight[1:]], markersize=5.0, markerfacecolor='r', marker='o', linestyle='None', markeredgecolor='k')
    pp4 = plt.plot(_pi[1], _pi[0], 'k', LineWidth=0.25)
    pp5 = plt.plot(np.array([-sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), np.array([sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), 'b')
    plt.plot(np.array([-sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), np.array([-sim.cam_lims[0, 0], -sim.cam_lims[0, 0]]), 'b')
    plt.plot(np.array([sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), np.array([-sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), 'b')
    plt.plot(np.array([-sim.cam_lims[1, 0], -sim.cam_lims[1, 0]]), np.array([-sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), 'b')
    plt.legend(('Start', 'Projection (FOV)', 'Projection (not FOV)', 'Projection Path', 'Camera FOV Bounds'))
    plt.title('Target Projection Path on Normalized Image Plane')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.axes().set_aspect('equal', 'datalim')
    plt.pause(0.1)

    plt.show()
