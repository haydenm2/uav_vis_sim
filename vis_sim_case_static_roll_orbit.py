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
    x_end = 1732.0
    dx = 1.0
    _x = np.linspace(1.0, x_end, num=int(x_end / dx))
    z_line = _x*0.0  # zero reference line
    u_line = z_line + 180.0  # upper limit reference line
    l_line = z_line - 180.0  # lower limit reference line

    ##########################################################################
    ########################### Live Plotting Bounds #########################
    ##########################################################################

    init = True                         # data struct initialization flag
    show_sim = False                    # toggle display of simulated course
    all_constraints = True              # toggle to return all critical angle constraint solutions or just the closest on either side of the current orientation

    Va = 50.0
    camera_roll_offset = np.radians(60)

    # Initialize simulator
    x0 = np.array([[-_x.item(0), 0, 1000]]).T
    xt0 = np.array([[0, 0, 0]]).T
    ypr0 = np.array([0, 0, np.arctan(Va**2/(9.81*_x.item(0)))])
    ypr_g0 = np.array([0, 0, camera_roll_offset])
    h_fov = np.deg2rad(10)
    v_fov = np.deg2rad(10)

    sim = UAV_simulator(x0, xt0, ypr0, ypr_g0, h_fov, v_fov)
    x = np.vstack([-_x, 0*_x, 0*_x]) + x0.reshape(-1, 1)
    xt = np.vstack([0*_x, 0*_x, 0*_x]) + xt0.reshape(-1, 1)
    ypr = np.vstack([0*_x, 0*_x, np.arctan(Va**2/(9.81*_x))])
    ypr_g = np.vstack([0*_x, 0*_x, 0*_x]) + ypr_g0.reshape(-1, 1)

    for i in range(len(x[0])):
        sim.UpdateX(x[:, i], visualize=show_sim)
        sim.UpdateTargetX(xt[:, i], visualize=show_sim)
        sim.UpdateYPR(ypr[:, i], visualize=show_sim)
        sim.UpdateGimbalYPR(ypr_g[:, i], visualize=show_sim)

        v_combined = sim.GetAxes()
        v_r = v_combined[:, 2].reshape(-1, 1)

        angs_r = sim.CalculateCriticalAngles(v_r, all=all_constraints)

        if init:
            init = False
            _in_sight = sim.in_sight
            _pi = sim.p_i

            _a1 = angs_r.reshape(-1, 1)
            _a1 = np.sort(_a1, axis=0)
        else:
            _in_sight = np.hstack((_in_sight, sim.in_sight))
            _pi = np.hstack((_pi, sim.p_i))

            _a1 = np.hstack((_a1, angs_r.reshape(-1, 1)))
            _a1 = np.sort(_a1, axis=0)

    # convert angle limits to degrees
    _a1 *= 180.0/np.pi

    # return to original UAV state for visualization accuracy
    sim.UpdateX(x0, visualize=show_sim)
    sim.UpdateTargetX(xt0, visualize=show_sim)
    sim.UpdateYPR(ypr0, visualize=show_sim)
    sim.UpdateGimbalYPR(ypr_g0, visualize=show_sim)

    ########################### Plot critical boundaries over radii #########################

    plt.figure(3)
    plt.tight_layout()

    # -------------------- plot rotation bounds for roll --------------------
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'b')  # dummy for legend
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 'r')  # dummy for legend
    for j in range(len(_a1[0])):
        zone_plot(_in_sight[j], _x[j], _a1[:, j].reshape(-1, 1))
    plt.plot(_x, z_line, 'k--')
    plt.plot(_x, u_line, 'k-')
    plt.plot(_x, l_line, 'k-')
    plt.title('Roll Axis Constraint Transform')
    plt.ylabel('Rotation Bounds (deg)')
    plt.legend(("Visibility Region (In Region)", "Visibility Region (Out of Region)"))
    plt.xlabel('Radius (m)')

    plt.pause(0.1)
    plt.show()
