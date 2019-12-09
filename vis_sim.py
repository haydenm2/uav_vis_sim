import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def rot3dxp(ang):
    R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]).transpose()
    return R


def rot3dyp(ang):
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]).transpose()
    return R


def rot3dzp(ang):
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]).transpose()
    return R


if __name__ == "__main__":
    # UAV and target conditions
    r = 10
    h = 40
    x = np.array([[-r], [0], [h]])
    xt = np.array([[0], [0], [0]])
    rpy = np.array([20, 0, 0])
    rpyg = np.array([20, 0, 0])

    phi_fov = np.deg2rad(45)
    phi_g = np.deg2rad(rpyg[0])
    phir = phi_fov / 2 - phi_g + np.arctan2(np.sign(x[0]) * r, h)  # right critical constraint
    phil = -phi_fov / 2 - phi_g + np.arctan2(np.sign(x[0]) * r, h)  # left critical constraint

    phi = np.deg2rad(rpy[0])

    theta_fov = np.deg2rad(45)
    theta_g = np.deg2rad(rpyg[1])
    thetab = theta_fov / 2 - theta_g  # back critical constraint
    thetaf = -theta_fov / 2 - theta_g  # front critical constraint

    theta = np.deg2rad(rpy[1])

    psi = np.deg2rad(rpy[2])
    psi_g = np.deg2rad(rpyg[2])

    xlos = h * np.tan(-phi - phi_g)
    rx = (xlos + h * np.tan(phi_fov / 2)) / (1 - xlos * np.tan(phi_fov / 2) / h)
    lx = (xlos - h * np.tan(phi_fov / 2)) / (1 + xlos * np.tan(phi_fov / 2) / h)

    ylos = h * np.tan(theta + theta_g)
    by = (ylos + h * np.tan(theta_fov / 2)) / (1 - ylos * np.tan(theta_fov / 2) / h)
    fy = (ylos - h * np.tan(theta_fov / 2)) / (1 + ylos * np.tan(theta_fov / 2) / h)

    # general plotting
    linedist = 100
    uav_length = 15
    endpoint = np.array([[0], [-linedist]])

    #  ######################################################################
    #  #############################3D PLOT##################################
    #  ######################################################################

    # setup background base
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    los = np.hstack([x, xt])
    ax.plot3D(los[0, :], los[1, :], los[2, :])
    pose_point = ax.plot3D(x[0], x[1], x[2], 'ko', MarkerFaceColor='k')
    pose_line = ax.plot3D([x[0], x[0]], [x[1], x[1]], [x[2], 0], 'k--')
    pose_target = ax.plot3D(xt[0], xt[1], xt[2], 'ro', MarkerFaceColor='r')
    lfov = np.array([[0, 0], [0, 0], [0, 200]])
    lquiv = np.array([[-uav_length/2, uav_length/2], [0, 0], [0, 0]])
    lcamx = np.array([[0, 10], [0, 0], [0, 0]])
    lbNorth = np.array([[0, 10], [0, 0], [0, 0]])
    rng = 100
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_zlim(-2, rng)
    ax.set_title('3D Sim')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(True)

    Riv = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    R_ib = rot3dzp(psi) @ rot3dyp(theta) @ rot3dxp(phi) @ Riv
    R_vb = rot3dzp(psi) @ rot3dyp(theta) @ rot3dxp(phi)
    R_bg = rot3dzp(psi_g) @ rot3dyp(theta_g) @ rot3dxp(phi_g)
    R_gc = np.eye(3)
    R_cfov1_x = rot3dyp(phi_fov / 2)
    R_cfov2_x = rot3dyp(-phi_fov / 2)
    R_cfov1_y = rot3dxp(theta_fov / 2)
    R_cfov2_y = rot3dxp(-theta_fov / 2)

    # Rotate animation lines back out to common inertial frame for plotting
    loptax = R_ib.transpose() @ R_bg.transpose() @ lfov + x
    lfov1 = R_ib.transpose() @ R_bg.transpose() @ R_gc.transpose() @ R_cfov1_y.transpose() @ R_cfov1_x.transpose() @ lfov + x
    lfov2 = R_ib.transpose() @ R_bg.transpose() @ R_gc.transpose() @ R_cfov2_y.transpose() @ R_cfov1_x.transpose() @ lfov + x
    lfov3 = R_ib.transpose() @ R_bg.transpose() @ R_gc.transpose() @ R_cfov1_y.transpose() @ R_cfov2_x.transpose() @ lfov + x
    lfov4 = R_ib.transpose() @ R_bg.transpose() @ R_gc.transpose() @ R_cfov2_y.transpose() @ R_cfov2_x.transpose() @ lfov + x
    lquiv = R_ib.transpose() @ lquiv + x
    lcamx = R_ib.transpose() @ R_bg.transpose() @ R_gc.transpose() @ lcamx + x
    lbNorth = Riv.transpose() @ lbNorth + x
    lbz = R_ib.transpose() @ lfov + x

    ploptax = ax.plot3D(loptax[0, :], loptax[1, :], loptax[2, :], 'r--')
    plfov1 = ax.plot3D(lfov1[0, :], lfov1[1, :], lfov1[2, :], 'r-')
    plfov2 = ax.plot3D(lfov2[0, :], lfov2[1, :], lfov2[2, :], 'r-')
    plfov3 = ax.plot3D(lfov3[0, :], lfov3[1, :], lfov3[2, :], 'r-')
    plfov4 = ax.plot3D(lfov4[0, :], lfov4[1, :], lfov4[2, :], 'r-')
    plcamx = ax.plot3D(lcamx[0, :], lcamx[1, :], lcamx[2, :], 'r-')
    plbNorth = ax.plot3D(lbNorth[0, :], lbNorth[1, :], lbNorth[2, :], 'g-')
    plbz = ax.plot3D(lbz[0, :], lbz[1, :], lbz[2, :], 'k-')
    ax.quiver3D(lquiv[0, 0], lquiv[1, 0], lquiv[2, 0], lquiv[0, 1] - lquiv[0, 0], lquiv[1, 1] - lquiv[1, 0], lquiv[2, 1] - lquiv[2, 0], 'b', LineWidth=5)

    plt.show()
