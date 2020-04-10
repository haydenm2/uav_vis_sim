#!/usr/bin/env python3

from vis_sim import UAV_simulator
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# stack and resize with zeros to new array
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

# stack and resize with zeros to new array for 2d inputs (used for yaw stacking)
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

# plot different visibility region charts given multiple edge points (resulting in multiple visibility zones)
def zone_plot(in_sight, t, pts, bounds=180):
    f_pts = np.vstack([np.array([-bounds]), pts[~np.isnan(pts)].reshape(-1, 1), np.array([bounds])])
    # f_pts = pts2[pts2 != np.inf]
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
                plt.plot(np.array([t, t]), np.array([f_pts[2 * iiii], f_pts[2 * iiii + 1]]), 'r')

def atan_fit(x, a, b, c, d):
    f = a*np.arctan(b*x+c) + d
    return f

if __name__ == "__main__":

    # plotting vectors
    t_end = 69.0
    dt = 0.25
    _t = np.linspace(0.0, t_end, num=int(t_end / dt))
    z_line = _t*0.0  # zero reference line
    u_line = z_line + 180.0  # upper limit reference line
    l_line = z_line - 180.0  # lower limit reference line

    ##########################################################################
    ########################### Live Plotting Bounds #########################
    ##########################################################################

    init = True                         # data struct initialization flag
    show_sim = False                    # toggle display of simulated course
    all_constraints = True              # toggle to return all critical angle constraint solutions or just the closest on either side of the current orientation
    apply_fit = True                   # toggle whether to apply fit parameters to gimbal control or to simply calculate curve fit with no commands
    if apply_fit:
        atan_a, atan_b, atan_c, atan_d = -1.00000000e+00, 5.00000000e-02, -1.72300000e+00, -1.79407593e-09   # calculated parameters for arctangent curve fit

    # Initialize simulator
    x0 = np.array([[0, 0, 1000]]).T
    xt0 = np.array([[0, 1723, 0]]).T
    ypr0 = np.array([0, 0, 0])
    ypr_g0 = np.array([0, 0, 0])
    h_fov = np.deg2rad(10)
    v_fov = np.deg2rad(10)

    sim = UAV_simulator(x0, xt0, ypr0, ypr_g0, h_fov, v_fov)

    Va = 50.0
    x = np.vstack([0*_t, Va*_t, 0*_t]) + x0.reshape(-1, 1)
    xt = np.vstack([0*_t, 0*_t, 0*_t]) + xt0.reshape(-1, 1)
    ypr = np.vstack([0*_t, 0*_t, 0*_t]) + ypr0.reshape(-1, 1)
    if apply_fit:
        ypr_g = np.vstack([0*_t, atan_fit(_t, atan_a, atan_b, atan_c, atan_d), 0*_t])
    else:
        ypr_g = np.vstack([0*_t, 0*_t, 0*_t]) + ypr_g0.reshape(-1, 1)

    for i in range(len(x[0])):
        sim.UpdateX(x[:, i], visualize=show_sim)
        sim.UpdateTargetX(xt[:, i], visualize=show_sim)
        sim.UpdateYPR(ypr[:, i], visualize=show_sim)
        sim.UpdateGimbalYPR(ypr_g[:, i], visualize=show_sim)

        v_combined = sim.GetAxes()
        v_pg = v_combined[:, 4].reshape(-1, 1)

        angs_pg = sim.CalculateCriticalAngles(v_pg, all=all_constraints)

        if init:
            init = False
            _in_sight = sim.in_sight
            _pi = sim.p_i

            _a5 = angs_pg.reshape(-1, 1)
            _a5 = np.sort(_a5, axis=0)

        else:
            _in_sight = np.hstack((_in_sight, sim.in_sight))
            _pi = np.hstack((_pi, sim.p_i))

            _a5 = np.hstack((_a5, angs_pg.reshape(-1, 1)))
            _a5 = np.sort(_a5, axis=0)

    # convert angle limits to degrees
    _a5 *= 180.0/np.pi

    # return to original UAV state for visualization accuracy
    sim.UpdateX(x0, visualize=show_sim)
    sim.UpdateTargetX(xt0, visualize=show_sim)
    sim.UpdateYPR(ypr0, visualize=show_sim)
    sim.UpdateGimbalYPR(ypr_g0, visualize=show_sim)

    ########################### Plot critical boundaries over time #########################
    if ~apply_fit:
        a0 = np.radians(60)/np.radians(90), 2.0, -34.0, 0
        param, param_cov = curve_fit(atan_fit, _t, np.radians((_a5[0]+_a5[1])/2.0), a0)  # result => 1.00000001e+00  4.66578145e-02 -2.97448872e-01 -1.22585389e-08
        print("Arctan curve fit parameters: ", param)

    plt.figure(3)
    plt.tight_layout()

    # -------------------- plot rotation bounds for pitch gimbal --------------------
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'b')  # dummy for legend
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'r')  # dummy for legend
    for j in range(len(_a5[0])):
        zone_plot(_in_sight[j], _t[j], _a5[:, j].reshape(-1, 1))
    plt.plot(_t, z_line, 'k--')
    plt.plot(_t, u_line, 'k-')
    plt.plot(_t, l_line, 'k-')
    plt.title('Pitch Gimbal Axis Constraint Transform')
    plt.ylabel('Rotation Bounds (deg)')
    plt.xlabel('Time (s)')
    plt.legend(("Visibility Region (In Region)", "Visibility Region (Out of Region)"))

    plt.pause(0.1)

    ########################### Plot Target projetion path #########################

    plt.figure(5)
    if _in_sight[0]:
        pp1 = plt.plot(_pi[0, 0], _pi[1, 0], markersize=15.0, markerfacecolor='b', marker='X', linestyle='None', markeredgecolor='k')
    else:
        pp1 = plt.plot(_pi[0, 0], _pi[1, 0], markersize=15.0, markerfacecolor='r', marker='X', linestyle='None', markeredgecolor='k')
    pp2 = plt.plot(_pi[0, 1:][_in_sight[1:]], _pi[1, 1:][_in_sight[1:]], markersize=5.0, markerfacecolor='b', marker='o', linestyle='None', markeredgecolor='k')
    pp3 = plt.plot(_pi[0, 1:][~_in_sight[1:]], _pi[1, 1:][~_in_sight[1:]], markersize=5.0, markerfacecolor='r', marker='o', linestyle='None', markeredgecolor='k')
    pp4 = plt.plot(_pi[0], _pi[1], 'k', LineWidth=0.25)
    pp5 = plt.plot(np.array([-sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), np.array([sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), 'b')
    plt.plot(np.array([-sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), np.array([-sim.cam_lims[1, 0], -sim.cam_lims[1, 0]]), 'b')
    plt.plot(np.array([sim.cam_lims[0, 0], sim.cam_lims[0, 0]]), np.array([-sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), 'b')
    plt.plot(np.array([-sim.cam_lims[0, 0], -sim.cam_lims[0, 0]]), np.array([-sim.cam_lims[1, 0], sim.cam_lims[1, 0]]), 'b')
    plt.legend(('Start', 'Projection (FOV)', 'Projection (not FOV)', 'Projection Path', 'Camera FOV Bounds'))
    plt.title('Target Projection Path on Normalized Image Plane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axes().set_aspect('equal', 'datalim')
    plt.axes().invert_yaxis()
    plt.pause(0.1)

    plt.show()
