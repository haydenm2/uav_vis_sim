import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tools.tools import Rxp, Ryp, Rzp
import parameters.aerosonde_parameters as UAV
import parameters.target_parameters as TAR

class vision_uav_viewer:
    def __init__(self, x0=np.array([[UAV.pn0], [UAV.pe0], [UAV.pd0]]), xt0=np.array([[TAR.pn0], [TAR.pe0], [TAR.pd0]]),
                 ypr=np.array([UAV.psi0, UAV.theta0, UAV.phi0]),
                 ypr_g=np.array([UAV.psig0, UAV.thetag0, UAV.phig0]), h_fov=UAV.h_fov,
                 v_fov=UAV.v_fov):
        # UAV and target conditions
        self.x = x0  # UAV state
        self.xt = xt0  # target state
        self.ypr = ypr  # UAV attitude (yaw, pitch, roll)
        self.ypr_g = ypr_g  # gimbal attitude (yaw, pitch, roll)
        self.h_fov = h_fov  # horizontal field of view
        self.v_fov = v_fov  # vertical field of view
        self.R_perturb_uav = Rxp(0)
        self.perturb_uav = np.array([0, 0, 0])
        self.perturb_gimb = np.array([0, 0, 0])

        # General vectors, plotting variables, and flags
        self.uav_length = 7
        self.x_line_scale = 40
        self.e1 = np.array([[1], [0], [0]])  # basis vector 1
        self.e2 = np.array([[0], [1], [0]])  # basis vector 2
        self.e3 = np.array([[0], [0], [1]])  # basis vector 3
        plt_range = x0.item(2)
        self.in_sight = False
        self.show_north = False
        self.show_z = False

        # ----------------------- Initialize 3D Simulation Plot -----------------------
        # Setup plotting base for 3D sim
        fig = plt.figure(1, figsize=plt.figaspect(2.))
        self.ax = fig.add_subplot(2, 1, 1, projection='3d')  # plot axis handle
        self.ax.set_xlim3d(-plt_range, plt_range)  # plotting x limit
        self.ax.set_ylim3d(-plt_range, plt_range)  # plotting y limit
        self.ax.set_zlim3d(-plt_range, 2.0)  # plotting z limit
        self.ax.set_title('3D Sim')  # plotting title
        self.ax.set_xlabel('N')  # plotting x axis label
        self.ax.set_ylabel('E')  # plotting y axis label
        self.ax.set_zlabel('D')  # plotting z axis label
        self.ax.invert_yaxis()
        self.ax.invert_zaxis()
        self.ax.grid(True)  # plotting show grid lines

        # Define initial rotation matrices
        self.R_iv = np.eye(3) #np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # rotation matrix from inertial to vehicle frame
        self.R_vb = self.R_perturb_uav @ Rxp(self.ypr[2]) @ Ryp(self.ypr[1]) @ Rzp(
            self.ypr[0])  # rotation matrix from vehicle to body frame
        self.R_bg = Rzp(self.ypr_g[2] + self.perturb_gimb[2]) @ Ryp(
            self.ypr_g[1] + self.perturb_gimb[1]) @ Rxp(self.ypr_g[0] + self.perturb_gimb[0]) @ Ryp(
            -np.pi / 2.0)  # rotation matrix from body to gimbal frame
        self.R_gc = Rzp(np.pi / 2.0) @ Ryp(np.pi / 2.0)  # rotation matrix from gimbal to camera frame
        self.R_cfov1_x = Ryp(self.h_fov / 2)  # rotation matrix from camera to max x field of view frame
        self.R_cfov2_x = Ryp(-self.h_fov / 2)  # rotation matrix from camera to min x field of view frame
        self.R_cfov1_y = Rxp(self.v_fov / 2)  # rotation matrix from camera to max y field of view frame
        self.R_cfov2_y = Rxp(-self.v_fov / 2)  # rotation matrix from camera to min y field of view frame
        self.R_ic = self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv

        # Rotate animation lines and points back out to common inertial frame for plotting
        llos = np.hstack([self.x, self.xt])  # line of sight vector
        lx = np.hstack((np.zeros((3, 1)), self.e1)) * self.x_line_scale
        lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ lx + self.x  # transformed camera x-axis line
        if self.show_north:
            lbNorth = self.R_iv.transpose() @ lx + self.x  # transformed North heading line
        self.L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                            self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))  # field of view boundary vectors in camera frame
        self.F = np.hstack([np.cross(self.L[:, 0], self.L[:, 1]).reshape(-1, 1),
                            np.cross(self.L[:, 1], self.L[:, 2]).reshape(-1, 1),
                            np.cross(self.L[:, 2], self.L[:, 3]).reshape(-1, 1),
                            np.cross(self.L[:, 3], self.L[:, 0]).reshape(-1, 1)])  # field of view frustum normal vectors
        L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.L  # field of view corner lines in vehicle frame
        pts = self.R_iv @ (-self.x[2] * L_v @ np.linalg.inv(np.diag((self.e3.transpose() @ L_v)[0]))) + self.x  # field of view corners projected on observation plane
        lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))  # field of view corner line 1
        lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))  # field of view corner line 2
        lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))  # field of view corner line 3
        lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))  # field of view corner line 4

        l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.e3
        pts_temp = self.R_iv @ (-self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x
        loptax = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # optical axis line

        # l_temp = self.R_vb.transpose() @ self.e3
        # pts_temp = -self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x
        if self.show_z:
            lbz = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # z-axis body frame line

        # Plotting handles
        self.plos = self.ax.plot3D(llos[0, :], llos[1, :], llos[2, :])  # plot line of sight vector
        # self.pose_point = self.ax.plot3D(self.x[0], self.x[1], self.x[2], 'ko', MarkerFaceColor='k')  # plot point at UAV pose
        self.pose_line = self.ax.plot3D([self.x[0], self.x[0]], [self.x[1], self.x[1]], [self.x[2], 0], 'k--')  # plot UAV pose line perpendicular to observation plane
        self.pose_target = self.ax.plot3D(self.xt[0], self.xt[1], self.xt[2], 'ro', MarkerFaceColor='r')  # plot point at target pose
        self.ploptax = self.ax.plot3D(loptax[0, :], loptax[1, :], loptax[2, :], 'r--')  # plot optical axis line
        self.plfov1 = self.ax.plot3D(lfov1[0, :], lfov1[1, :], lfov1[2, :], 'r-')  # plot field of view line 1
        self.plfov2 = self.ax.plot3D(lfov2[0, :], lfov2[1, :], lfov2[2, :], 'r-')  # plot field of view line 2
        self.plfov3 = self.ax.plot3D(lfov3[0, :], lfov3[1, :], lfov3[2, :], 'r-')  # plot field of view line 3
        self.plfov4 = self.ax.plot3D(lfov4[0, :], lfov4[1, :], lfov4[2, :], 'r-')  # plot field of view line 4
        self.plcamx = self.ax.plot3D(lcamx[0, :], lcamx[1, :], lcamx[2, :], 'b--')  # plot camera x axis
        if self.show_north:
            self.plbNorth = self.ax.plot3D(lbNorth[0, :], lbNorth[1, :], lbNorth[2, :],
                                           'g-')  # plot North heading line on body
        if self.show_z:
            self.plbz = self.ax.plot3D(lbz[0, :], lbz[1, :], lbz[2, :], 'k-')  # plot z-axis body frame line
        plpts = np.hstack((pts, pts[:, 0].reshape(-1, 1)))  # field of view extent points (with redundant final point)
        self.pfov = self.ax.plot3D(plpts[0, :], plpts[1, :], plpts[2, :], 'c-')  # plot field of view polygon projection on observation plane

        self.UpdateAirCraftModel(init=True)

        # ----------------------- Initialize Camera View Plot -----------------------
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (
                    self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)

        # Setup plotting base for camera FOV
        self.P_t = self.R_ic @ (self.xt - self.x)  # target line of sight vector in camera frame
        self.p_t = (self.P_t / np.linalg.norm(self.P_t)).reshape(-1, 1)  # selected target vector
        self.p_i = self.P_t / (self.e3.transpose() @ self.P_t)  # target point on normalized image plane
        self.delta_b = 1e-10  # acceptable border buffer size
        self.axc = fig.add_subplot(2, 1, 2)  # plot axis handle
        self.axc.set_xlim(-self.cam_lims[0], self.cam_lims[0])  # camera frame y limit
        self.axc.set_ylim(-self.cam_lims[1], self.cam_lims[1])  # camera frame x limit
        self.axc.set_title('Camera Field of View')
        self.axc.set_xlabel('x')  # plotting x axis label (camera x axis)
        self.axc.set_ylabel('y')  # plotting y axis label (camera y axis)
        self.axc.grid(True)  # plotting show grid lines
        self.axc.set_aspect(1)
        self.axc.invert_yaxis()
        self.camera_target_pose = self.axc.plot(self.p_i[0], self.p_i[1], 'ro', MarkerFaceColor='r')  # plot point at target pose on camera normalized image plane

        # Update
        self.UpdateSim()

    # update UAV position
    def UpdateUAV(self, x, ypr, ypr_g, visualize=True):
        self.x = x.reshape(-1, 1)
        self.ypr = ypr
        self.R_vb = self.R_perturb_uav @ Rxp(self.ypr[2]) @ Ryp(self.ypr[1]) @ Rzp(self.ypr[0])
        self.ypr_g = ypr_g
        self.R_bg = Rzp(self.ypr_g[2] + self.perturb_gimb[2]) \
                    @ Ryp(self.ypr_g[1] + self.perturb_gimb[1]) \
                    @ Rxp(self.ypr_g[0] + self.perturb_gimb[0]) \
                    @ Ryp(-np.pi / 2.0)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update target position
    def UpdateTarget(self, xt, visualize=True):
        self.xt = xt.reshape(-1, 1)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # applies perturbation to UAV orientation in body frame
    def UpdatePert_ypr_UAV(self, pert, visualize=True):
        self.R_perturb_uav = Rxp(pert[2]) @ Ryp(pert[1]) @ Rzp(pert[0])
        self.R_vb = self.R_perturb_uav @ Rxp(self.ypr[2]) @ Ryp(self.ypr[1]) @ Rzp(self.ypr[0])
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # applies perturbation to gimbal camera view orientation
    def UpdatePert_ypr_Gimbal(self, pert, visualize=True):
        self.perturb_gimb = pert
        self.R_bg = Rzp(self.ypr_g[2] + self.perturb_gimb[2]) @ Ryp(
            self.ypr_g[1] + self.perturb_gimb[1]) @ Rxp(self.ypr_g[0] + self.perturb_gimb[0]) @ Ryp(
            -np.pi / 2.0)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV camera horizontal field of view limits
    def UpdateHFOV(self, h_fov, visualize=True):
        self.h_fov = h_fov
        self.R_cfov1_x = Ryp(self.h_fov / 2)
        self.R_cfov2_x = Ryp(-self.h_fov / 2)
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (
                    self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)
        self.L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                            self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))  # field of view boundary vectors
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV camera vertical field of view limits
    def UpdateVFOV(self, v_fov, visualize=True):
        self.v_fov = v_fov
        self.R_cfov1_y = Rxp(self.v_fov / 2)
        self.R_cfov2_y = Rxp(-self.v_fov / 2)
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (
                    self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)
        self.L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                            self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                            self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))  # field of view boundary vectors
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # determines if target line of sight projection is within camera limits
    def CheckInFOV(self, visualize=True):
        temp = self.F.T @ self.p_t
        # are all target projection point components within all image borders? -> within field of view
        if np.all(temp >= self.delta_b):
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('x \n Target in sight: True', color="green", fontweight='bold')
        # are all target projection point components within a very small threshold distance from a border and within all other image borders? -> within field of view AND on a border
        elif np.all(temp >= -self.delta_b):
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('x \n Target in sight: Border', color="orange", fontweight='bold')
        # are any target vector component distances from image view borders NaN? -> invalid solution = out of sight
        elif np.any(np.isnan(temp)):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('x \n Target in sight: False (NaN)', color="red", fontweight='bold')
        # is z-element of target vector in camera frame negative? -> behind camera = out of sight
        elif np.any(temp < -self.delta_b):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('x \n Target in sight: False', color="red", fontweight='bold')
        # other cause
        else:
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('x \n Target in sight: False (Other)', color="red", fontweight='bold')

    # updates states and simulation
    def UpdateSim(self, visualize=True):
        # ----------------------- State Updates -----------------------
        self.R_ic = self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv
        self.P_t = self.R_ic @ (self.xt - self.x)
        self.p_t = (self.P_t / np.linalg.norm(self.P_t)).reshape(-1, 1)
        self.p_i = self.P_t / (self.e3.transpose() @ self.P_t)
        self.F = np.hstack([np.cross(self.L[:, 0], self.L[:, 1]).reshape(-1, 1),
                            np.cross(self.L[:, 1], self.L[:, 2]).reshape(-1, 1),
                            np.cross(self.L[:, 2], self.L[:, 3]).reshape(-1, 1),
                            np.cross(self.L[:, 3], self.L[:, 0]).reshape(-1, 1)])
        self.CheckInFOV()

        if visualize:
            # ----------------------- 3D Simulation Update -----------------------

            # Rotate animation lines and points back out to common inertial frame for plotting
            llos = np.hstack([self.x, self.xt])  # line of sight vector
            lx = np.hstack((np.zeros((3, 1)), self.e1)) * self.x_line_scale
            lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ lx + self.x  # transformed camera x-axis line
            if self.show_north:
                lbNorth = self.R_iv.transpose() @ lx + self.x  # transformed North heading line
            L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.L  # field of view corner lines in vehicle frame
            pts = self.R_iv @ (-self.x[2] * L_v @ np.linalg.inv(np.diag(
                (self.e3.transpose() @ L_v)[0]))) + self.x  # field of view corners projected on observation plane
            plpts = np.hstack(
                (pts, pts[:, 0].reshape(-1, 1)))  # field of view extent points (with redundant final point)
            lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))  # field of view corner line 1
            lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))  # field of view corner line 2
            lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))  # field of view corner line 3
            lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))  # field of view corner line 4

            l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.e3
            pts_temp = self.R_iv @ (-self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x  # optical axis line points on observation plane
            loptax = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # optical axis line

            # l_temp = self.R_vb.transpose() @ self.e3
            # pts_temp = self.R_iv @ (-self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x  # body z-axis line points on observation plane
            if self.show_z:
                lbz = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # z-axis body frame line

            # Plotting handles
            self.plos[0].set_data_3d(llos[0, :], llos[1, :], llos[2, :])  # plot line of sight vector
            # self.pose_point[0].set_data_3d(self.x[0], self.x[1], self.x[2])  # plot point at UAV pose
            self.pose_line[0].set_data_3d([self.x[0].item(), self.x[0].item()], [self.x[1].item(), self.x[1].item()],
                                          [self.x[2].item(),
                                           0])  # plot UAV pose line perpendicular to observation plane
            self.pose_target[0].set_data_3d(self.xt[0], self.xt[1], self.xt[2])  # plot point at target pose
            self.ploptax[0].set_data_3d(loptax[0, :], loptax[1, :], loptax[2, :])  # plot optical axis line
            self.plfov1[0].set_data_3d(lfov1[0, :], lfov1[1, :], lfov1[2, :])  # plot field of view line 1
            self.plfov2[0].set_data_3d(lfov2[0, :], lfov2[1, :], lfov2[2, :])  # plot field of view line 2
            self.plfov3[0].set_data_3d(lfov3[0, :], lfov3[1, :], lfov3[2, :])  # plot field of view line 3
            self.plfov4[0].set_data_3d(lfov4[0, :], lfov4[1, :], lfov4[2, :])  # plot field of view line 4
            self.plcamx[0].set_data_3d(lcamx[0, :], lcamx[1, :], lcamx[2, :])  # plot camera x axis
            if self.show_north:
                self.plbNorth[0].set_data_3d(lbNorth[0, :], lbNorth[1, :],
                                             lbNorth[2, :])  # plot North heading line on body
            if self.show_z:
                self.plbz[0].set_data_3d(lbz[0, :], lbz[1, :], lbz[2, :])  # plot z-axis body frame line
            self.pfov[0].set_data_3d(plpts[0, :], plpts[1, :],
                                     plpts[2, :])  # plot field of view polygon projection on observation plane

            self.UpdateAirCraftModel()  # update UAV model plot

            # resize plot
            plot_buff = 10.0
            plot_size = np.max(np.vstack((self.x, self.xt))) + plot_buff
            self.ax.set_xlim3d(-plot_size, plot_size)  # plotting x limit
            self.ax.set_ylim3d(-plot_size, plot_size)  # plotting y limit
            self.ax.set_zlim3d(-plot_size, 0.0)  # plotting z limit
            self.ax.invert_yaxis()
            self.ax.invert_zaxis()
            # self.ax.autoscale_view()

            # ----------------------- Camera View Update -----------------------

            if self.P_t[2] >= 0 and not np.any(
                    np.isnan(self.p_i)):  # prevent plotting projection when target is behind camera or NaNs exist
                self.camera_target_pose[0].set_xdata(self.p_i[0])
                self.camera_target_pose[0].set_ydata(
                    self.p_i[1])  # update target point on camera normalized image plane plot
            else:
                self.camera_target_pose[0].set_xdata(None)
                self.camera_target_pose[0].set_ydata(None)

            self.axc.set_xlim(-self.cam_lims[0], self.cam_lims[0])  # update camera frame y limit
            self.axc.set_ylim(-self.cam_lims[1], self.cam_lims[1])  # update camera frame x limit
            self.axc.invert_yaxis()
            self.axc.set_aspect(1)

            plt.pause(0.01)

    # updates simulation model surfaces of aircraft system and refreshes simulation
    def UpdateAirCraftModel(self, init=False):
        # feature scale
        fuse_l1 = 2
        fuse_l2 = 1
        fuse_l3 = 4
        fuse_h = 1
        fuse_w = 1
        wing_l = 1
        wing_w = 5
        tail_h = 1
        tailwing_l = 1
        tailwing_w = 3

        # points are in NED coordinates
        points = np.array([[fuse_l1, 0, 0],  # point 1
                           [fuse_l2, fuse_w / 2, -fuse_h / 2],  # point 2
                           [fuse_l2, -fuse_w / 2, -fuse_h / 2],  # point 3
                           [fuse_l2, -fuse_w / 2, fuse_h / 2],  # point 4
                           [fuse_l2, fuse_w / 2, fuse_h / 2],  # point 5
                           [-fuse_l3, 0, 0],  # point 6
                           [0, wing_w / 2, 0],  # point 7
                           [-wing_l, wing_w / 2, 0],  # point 8
                           [-wing_l, -wing_w / 2, 0],  # point 9
                           [0, -wing_w / 2, 0],  # point 10
                           [-fuse_l3 + tailwing_l, tailwing_w / 2, 0],  # point 11
                           [-fuse_l3, tailwing_w / 2, 0],  # point 12
                           [-fuse_l3, -tailwing_w / 2, 0],  # point 13
                           [-fuse_l3 + tailwing_l, -tailwing_w / 2, 0],  # point 14
                           [-fuse_l3 + tailwing_l, 0, 0],  # point 15
                           [-fuse_l3, 0, -tail_h],  # point 16
                           ]).T
        # scale points for better rendering
        scale = 10
        points = scale * points

        # transform points
        points = self.R_iv.transpose() @ self.R_vb.transpose() @ points + self.x

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])

        points = points.T
        mesh = np.array([[points[0], points[1], points[2]],  # nose 1
                         [points[0], points[2], points[3]],  # nose 2
                         [points[0], points[3], points[4]],  # nose 3
                         [points[0], points[4], points[1]],  # nose 4
                         [points[5], points[1], points[2]],  # body 1
                         [points[5], points[2], points[3]],  # body 2
                         [points[5], points[3], points[4]],  # body 3
                         [points[5], points[4], points[1]],  # body 4
                         [points[6], points[7], points[8]],  # wing 1
                         [points[8], points[9], points[6]],  # wing 2
                         [points[10], points[11], points[12]],  # tailwing 1
                         [points[12], points[13], points[10]],  # tailwing 2
                         [points[5], points[14], points[15]],  # rudder
                         ])

        MeshColor = []
        MeshColor.append("C3")  # nose 1
        MeshColor.append("C3")  # nose 2
        MeshColor.append("C3")  # nose 3
        MeshColor.append("C3")  # nose 4
        MeshColor.append("C0")  # body 1
        MeshColor.append("C0")  # body 2
        MeshColor.append("C0")  # body 3
        MeshColor.append("C0")  # body 4
        MeshColor.append("C1")  # wing 1
        MeshColor.append("C1")  # wing 2
        MeshColor.append("C2")  # tailwing 1
        MeshColor.append("C2")  # tailwing 2
        MeshColor.append("C1")  # rudder

        # update mesh for UAV
        if init:
            self.collection = []
            for i in range(len(mesh)):
                self.collection.append(Poly3DCollection([list(zip(mesh[i, :, 0], mesh[i, :, 1], mesh[i, :, 2]))]))
                self.collection[i].set_facecolor(MeshColor[i])
                self.ax.add_collection3d(self.collection[i])
        else:
            for i in range(len(mesh)):
                self.collection[i].set_verts([list(zip(mesh[i, :, 0], mesh[i, :, 1], mesh[i, :, 2]))])
