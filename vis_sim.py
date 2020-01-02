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

class UAV_simulator:
    def __init__(self, x0=np.array([[10], [10], [40]]), xt0=np.array([[0], [0], [0]]), rpy=np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]), ypr_g=np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]), h_fov=np.deg2rad(45), v_fov=np.deg2rad(45)):
        # UAV and target conditions
        self.x = x0     # UAV state
        self.xt = xt0     # target state
        self.rpy = rpy   # UAV attitude (roll, pitch, yaw)
        self.ypr_g = ypr_g   # gimbal attitude (yaw, pitch, roll)
        self.h_fov = h_fov     # horizontal field of view
        self.v_fov = v_fov     # vertical field of view
        self.R_perturb_b = rot3dzp(0)
        self.R_perturb_c = rot3dxp(0)

        # General vectors and plotting variables
        self.uav_length = 7
        self.x_line_scale = 20
        self.e1 = np.array([[1], [0], [0]])  # basis vector 1
        self.e2 = np.array([[0], [1], [0]])  # basis vector 2
        self.e3 = np.array([[0], [0], [1]])  # basis vector 3
        plt_range = 100
        self.in_sight = False

        # ----------------------- Initialize 3D Simulation Plot -----------------------

        # Setup plotting base for 3D sim
        fig = plt.figure(1, figsize=plt.figaspect(2.))
        self.ax = fig.add_subplot(2, 1, 1, projection='3d')    # plot axis handle
        self.ax.set_xlim(-plt_range, plt_range)      # plotting x limit
        self.ax.set_ylim(-plt_range, plt_range)      # plotting y limit
        self.ax.set_zlim(-2, plt_range)      # plotting z limit
        self.ax.set_title('3D Sim')      # plotting title
        self.ax.set_xlabel('x')      # plotting x axis label
        self.ax.set_ylabel('y')      # plotting y axis label
        self.ax.set_zlabel('z')      # plotting z axis label
        self.ax.grid(True)      # plotting show grid lines

        # Define initial rotation matrices
        self.R_iv = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])      # rotation matrix from inertial to vehicle frame
        self.R_vb = self.R_perturb_b @ rot3dzp(self.rpy[2]) @ rot3dyp(self.rpy[1]) @ rot3dxp(self.rpy[0])      # rotation matrix from vehicle to body frame
        self.R_bg = self.R_perturb_c @ rot3dxp(self.ypr_g[2]) @ rot3dyp(self.ypr_g[1]) @ rot3dzp(self.ypr_g[0])      # rotation matrix from body to gimbal frame
        self.R_gc = np.eye(3)      # rotation matrix from gimbal to camera frame
        self.R_cfov1_x = rot3dyp(self.h_fov / 2)      # rotation matrix from camera to max x field of view frame
        self.R_cfov2_x = rot3dyp(-self.h_fov / 2)      # rotation matrix from camera to min x field of view frame
        self.R_cfov1_y = rot3dxp(self.v_fov / 2)      # rotation matrix from camera to max y field of view frame
        self.R_cfov2_y = rot3dxp(-self.v_fov / 2)      # rotation matrix from camera to min y field of view frame
        self.R_ic = self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv

        # Rotate animation lines and points back out to common inertial frame for plotting
        llos = np.hstack([self.x, self.xt])     # line of sight vector
        lx = np.hstack((np.zeros((3, 1)), self.e1))*self.x_line_scale
        lquiv = np.hstack((np.zeros((3, 1)), self.e1)) * self.uav_length
        lquiv = self.R_iv.transpose() @ self.R_vb.transpose() @ lquiv + self.x      # transformed UAV heading arrow
        lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ lx + self.x      # transformed camera x-axis line
        lbNorth = self.R_iv.transpose() @ lx + self.x      # transformed North heading line
        L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                       self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                       self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                       self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))       # field of view corner lines
        L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ L                # field of view corner lines in vehicle frame
        pts = self.R_iv @ (self.x[2] * L_v @ np.linalg.inv(np.diag((self.e3.transpose() @ L_v)[0]))) + self.x     # field of view corners projected on observation plane
        lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))       # field of view corner line 1
        lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))       # field of view corner line 2
        lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))       # field of view corner line 3
        lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))       # field of view corner line 4

        l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.e3
        pts_temp = self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x
        loptax = np.hstack((self.x, pts_temp.reshape(-1, 1)))       # optical axis line

        l_temp = self.R_vb.transpose() @ self.e3
        pts_temp = self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x
        lbz = np.hstack((self.x, pts_temp.reshape(-1, 1)))       # z-axis body frame line

        # Plotting handles
        self.plos = self.ax.plot3D(llos[0, :], llos[1, :], llos[2, :])  # plot line of sight vector
        self.pose_point = self.ax.plot3D(self.x[0], self.x[1], self.x[2], 'ko', MarkerFaceColor='k')  # plot point at UAV pose
        self.pose_line = self.ax.plot3D([self.x[0], self.x[0]], [self.x[1], self.x[1]], [self.x[2], 0], 'k--')  # plot UAV pose line perpendicular to observation plane
        self.pose_target = self.ax.plot3D(self.xt[0], self.xt[1], self.xt[2], 'ro', MarkerFaceColor='r')  # plot point at target pose
        self.ploptax = self.ax.plot3D(loptax[0, :], loptax[1, :], loptax[2, :], 'r--')       # plot optical axis line
        self.plfov1 = self.ax.plot3D(lfov1[0, :], lfov1[1, :], lfov1[2, :], 'r-')       # plot field of view line 1
        self.plfov2 = self.ax.plot3D(lfov2[0, :], lfov2[1, :], lfov2[2, :], 'r-')       # plot field of view line 2
        self.plfov3 = self.ax.plot3D(lfov3[0, :], lfov3[1, :], lfov3[2, :], 'r-')       # plot field of view line 3
        self.plfov4 = self.ax.plot3D(lfov4[0, :], lfov4[1, :], lfov4[2, :], 'r-')       # plot field of view line 4
        self.plcamx = self.ax.plot3D(lcamx[0, :], lcamx[1, :], lcamx[2, :], 'r-')       # plot camera x axis
        self.plbNorth = self.ax.plot3D(lbNorth[0, :], lbNorth[1, :], lbNorth[2, :], 'g-')       # plot North heading line on body
        self.plbz = self.ax.plot3D(lbz[0, :], lbz[1, :], lbz[2, :], 'k-')       # plot z-axis body frame line
        plpts = np.hstack((pts, pts[:, 0].reshape(-1, 1)))       # field of view extent points (with redundant final point)
        self.pfov = self.ax.plot3D(plpts[0, :], plpts[1, :], plpts[2, :], 'c-')       # plot field of view polygon projection on observation plane
        self.quiv = self.ax.plot3D(lquiv[0, :], lquiv[1, :], lquiv[2, :], 'b-', LineWidth=5)       # plot UAV arrow

        # ----------------------- Initialize Camera View Plot -----------------------

        self.cam_lims = (self.R_cfov1_y @ self.R_cfov2_x @ self.e3) / (self.e3.transpose() @ self.R_cfov1_y @ self.R_cfov2_x @ self.e3)

        # Setup plotting base for camera FOV
        self.axc = fig.add_subplot(2, 1, 2)  # plot axis handle
        self.axc.set_xlim(-self.cam_lims[1], self.cam_lims[1])  # camera frame y limit
        self.axc.set_ylim(-self.cam_lims[0], self.cam_lims[0])  # camera frame x limit
        self.axc.set_title('Camera Field of View')
        self.axc.set_xlabel('y')  # plotting x axis label (camera x axis)
        self.axc.set_ylabel('x')  # plotting y axis label (camera y axis)
        self.axc.grid(True)  # plotting show grid lines
        self.axc.set_aspect(1)

        self.P_i = self.R_ic @ (self.xt - self.x)  # target line of sight vector in camera frame
        self.p_i = self.P_i / (self.e3.transpose() @ self.P_i)  # target point on normalized image plane
        self.camera_target_pose = self.axc.plot(self.p_i[0], self.p_i[1], 'ro', MarkerFaceColor='r')  # plot point at target pose on camera normalized image plane

        self.UpdateSim()

    # update UAV position
    def UpdateX(self, x):
        self.x = x.reshape(-1, 1)
        self.UpdateSim()

    # update target position
    def UpdateTargetX(self, xt):
        self.xt = xt.reshape(-1, 1)
        self.UpdateSim()

    # update UAV orientation
    def UpdateRPY(self, rpy):
        self.rpy = rpy
        self.R_vb = self.R_perturb_b @ rot3dzp(self.rpy[2]) @ rot3dyp(self.rpy[1]) @ rot3dxp(self.rpy[0])
        self.UpdateSim()

    # update UAV gimbal orientation
    def UpdateGimbalYPR(self, ypr_g):
        self.ypr_g = ypr_g
        self.R_bg = self.R_perturb_c @ rot3dxp(self.ypr_g[2]) @ rot3dyp(self.ypr_g[1]) @ rot3dzp(self.ypr_g[0])
        self.UpdateSim()

    # applies general perturbation to UAV camera view orientation
    def UpdatePertR(self, R, visualize=True):
        self.R_perturb_c = R
        self.R_vb = rot3dzp(self.rpy[2]) @ rot3dyp(self.rpy[1]) @ rot3dxp(self.rpy[0])
        self.R_bg = self.R_perturb_c @ rot3dxp(self.ypr_g[2]) @ rot3dyp(self.ypr_g[1]) @ rot3dzp(self.ypr_g[0])
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV camera horizontal field of view limits
    def UpdateHFOV(self, h_fov):
        self.h_fov = h_fov
        self.R_cfov1_x = rot3dyp(self.h_fov / 2)
        self.R_cfov2_x = rot3dyp(-self.h_fov / 2)
        self.UpdateSim()

    # update UAV camera vertical field of view limits
    def UpdateVFOV(self, v_fov):
        self.v_fov = v_fov
        self.R_cfov1_y = rot3dxp(self.v_fov / 2)
        self.R_cfov2_y = rot3dxp(-self.v_fov / 2)
        self.UpdateSim()

    # updates states and simulation
    def UpdateSim(self, visualize=True):
        # ----------------------- State Updates -----------------------
        self.R_ic = self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv
        self.P_i = self.R_ic @ (self.xt - self.x)
        self.p_i = self.P_i / (self.e3.transpose() @ self.P_i)  # target point on normalized image plane
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)
        self.CheckInFOV()

        if visualize:
            # ----------------------- 3D Simulation Update -----------------------

            # Rotate animation lines and points back out to common inertial frame for plotting
            llos = np.hstack([self.x, self.xt])  # line of sight vector
            lx = np.hstack((np.zeros((3, 1)), self.e1))*self.x_line_scale
            lquiv = np.hstack((np.zeros((3, 1)), self.e1)) * self.uav_length
            lquiv = self.R_iv.transpose() @ self.R_vb.transpose() @ lquiv + self.x  # transformed UAV heading arrow
            lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ lx + self.x  # transformed camera x-axis line
            lbNorth = self.R_iv.transpose() @ lx + self.x  # transformed North heading line
            L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                           self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                           self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                           self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))  # field of view corner lines
            L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ L  # field of view corner lines in vehicle frame
            pts = self.R_iv @ (self.x[2] * L_v @ np.linalg.inv(np.diag((self.e3.transpose() @ L_v)[0]))) + self.x  # field of view corners projected on observation plane
            plpts = np.hstack((pts, pts[:, 0].reshape(-1, 1)))  # field of view extent points (with redundant final point)
            lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))  # field of view corner line 1
            lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))  # field of view corner line 2
            lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))  # field of view corner line 3
            lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))  # field of view corner line 4

            l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.e3
            pts_temp = self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x  # optical axis line points on observation plane
            loptax = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # optical axis line

            l_temp = self.R_vb.transpose() @ self.e3
            pts_temp = self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x  # body z-axis line points on observation plane
            lbz = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # z-axis body frame line

            # Plotting handles
            self.plos[0].set_data_3d(llos[0, :], llos[1, :], llos[2, :])  # plot line of sight vector
            self.pose_point[0].set_data_3d(self.x[0], self.x[1], self.x[2])  # plot point at UAV pose
            self.pose_line[0].set_data_3d([self.x[0].item(), self.x[0].item()], [self.x[1].item(), self.x[1].item()], [self.x[2].item(), 0])  # plot UAV pose line perpendicular to observation plane
            self.pose_target[0].set_data_3d(self.xt[0], self.xt[1], self.xt[2])  # plot point at target pose
            self.ploptax[0].set_data_3d(loptax[0, :], loptax[1, :], loptax[2, :])  # plot optical axis line
            self.plfov1[0].set_data_3d(lfov1[0, :], lfov1[1, :], lfov1[2, :])  # plot field of view line 1
            self.plfov2[0].set_data_3d(lfov2[0, :], lfov2[1, :], lfov2[2, :])  # plot field of view line 2
            self.plfov3[0].set_data_3d(lfov3[0, :], lfov3[1, :], lfov3[2, :])  # plot field of view line 3
            self.plfov4[0].set_data_3d(lfov4[0, :], lfov4[1, :], lfov4[2, :])  # plot field of view line 4
            self.plcamx[0].set_data_3d(lcamx[0, :], lcamx[1, :], lcamx[2, :])  # plot camera x axis
            self.plbNorth[0].set_data_3d(lbNorth[0, :], lbNorth[1, :], lbNorth[2, :])  # plot North heading line on body
            self.plbz[0].set_data_3d(lbz[0, :], lbz[1, :], lbz[2, :])  # plot z-axis body frame line
            self.pfov[0].set_data_3d(plpts[0, :], plpts[1, :], plpts[2, :])  # plot field of view polygon projection on observation plane
            self.quiv[0].set_data_3d(lquiv[0, :], lquiv[1, :], lquiv[2, :])  # plot UAV arrow

            self.ax.autoscale_view()

            # ----------------------- Camera View Update -----------------------

            if self.P_i[2] >= 0 and not np.any(np.isnan(self.p_i)):  # prevent plotting projection when target is behind camera or NaNs exist
                self.camera_target_pose[0].set_xdata(self.p_i[1])
                self.camera_target_pose[0].set_ydata(self.p_i[0])  # update target point on camera normalized image plane plot
            else:
                self.camera_target_pose[0].set_xdata(None)
                self.camera_target_pose[0].set_ydata(None)

            self.axc.set_xlim(-self.cam_lims[1], self.cam_lims[1])  # update camera frame y limit
            self.axc.set_ylim(-self.cam_lims[0], self.cam_lims[0])  # update camera frame x limit
            self.axc.set_aspect(1)

            plt.pause(0.01)

    # determines if target line of sight projection is within camera limits
    def CheckInFOV(self, visualize=True):
        test = self.cam_lims - np.abs(self.p_i)  # test if target point magnitude is smaller than limits
        border_threshold = 1e-10
        if self.P_i[2] < 0:
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False (Target Behind Camera)', color="red", fontweight='bold')
        elif np.any(np.isnan(test)):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False (NaN)', color="red", fontweight='bold')
        elif np.any(test < -border_threshold):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False', color="red", fontweight='bold')
        elif np.all(test >= -border_threshold) and np.any(np.abs(test[0:2]) < border_threshold):
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: Border', color="orange", fontweight='bold')
        else:
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: True', color="green", fontweight='bold')

    # solves rodriguez axis angle rotation equation for angle given a desired axis and point location (i.e. solution to general form a*cos(ang) + b*sin(ang) + c = 0)
    def CalculateCriticalAngles(self, v, test=False, all=False):
        ax1 = np.tensordot(self.p_i - np.tensordot(v, self.p_i) * v, self.e1) - np.tensordot(self.cam_lims[0]*(self.p_i - np.tensordot(v, self.p_i) * v), self.e3)
        ax2 = np.tensordot(self.p_i - np.tensordot(v, self.p_i) * v, self.e1) - np.tensordot(-self.cam_lims[0]*(self.p_i - np.tensordot(v, self.p_i) * v), self.e3)
        ay1 = np.tensordot(self.p_i - np.tensordot(v, self.p_i) * v, self.e2) - np.tensordot(self.cam_lims[1]*(self.p_i - np.tensordot(v, self.p_i) * v), self.e3)
        ay2 = np.tensordot(self.p_i - np.tensordot(v, self.p_i) * v, self.e2) - np.tensordot(-self.cam_lims[1]*(self.p_i - np.tensordot(v, self.p_i) * v), self.e3)

        bx1 = np.tensordot(np.cross(v, self.p_i, axis=0), self.e1) - np.tensordot(self.cam_lims[0]*np.cross(v, self.p_i, axis=0), self.e3)
        bx2 = np.tensordot(np.cross(v, self.p_i, axis=0), self.e1) - np.tensordot(-self.cam_lims[0]*np.cross(v, self.p_i, axis=0), self.e3)
        by1 = np.tensordot(np.cross(v, self.p_i, axis=0), self.e2) - np.tensordot(self.cam_lims[1]*np.cross(v, self.p_i, axis=0), self.e3)
        by2 = np.tensordot(np.cross(v, self.p_i, axis=0), self.e2) - np.tensordot(-self.cam_lims[1]*np.cross(v, self.p_i, axis=0), self.e3)

        cx1 = np.tensordot(np.tensordot(v, self.p_i) * v, self.e1) - np.tensordot(self.cam_lims[0]*np.tensordot(v, self.p_i) * v, self.e3)
        cx2 = np.tensordot(np.tensordot(v, self.p_i) * v, self.e1) - np.tensordot(-self.cam_lims[0]*np.tensordot(v, self.p_i) * v, self.e3)
        cy1 = np.tensordot(np.tensordot(v, self.p_i) * v, self.e2) - np.tensordot(self.cam_lims[1]*np.tensordot(v, self.p_i) * v, self.e3)
        cy2 = np.tensordot(np.tensordot(v, self.p_i) * v, self.e2) - np.tensordot(-self.cam_lims[1]*np.tensordot(v, self.p_i) * v, self.e3)

        # arcsin approach:
        angx1 = np.array([])
        angx2 = np.array([])
        angy1 = np.array([])
        angy2 = np.array([])

        angx1 = np.hstack((angx1, np.arcsin(-cx1/np.sqrt(ax1**2 + bx1**2)) - np.arcsin(ax1/np.sqrt(ax1**2 + bx1**2))))
        angx2 = np.hstack((angx2, np.arcsin(-cx2/np.sqrt(ax2**2 + bx2**2)) - np.arcsin(ax2/np.sqrt(ax2**2 + bx2**2))))
        angy1 = np.hstack((angy1, np.arcsin(-cy1/np.sqrt(ay1**2 + by1**2)) - np.arcsin(ay1/np.sqrt(ay1**2 + by1**2))))
        angy2 = np.hstack((angy2, np.arcsin(-cy2/np.sqrt(ay2**2 + by2**2)) - np.arcsin(ay2/np.sqrt(ay2**2 + by2**2))))

        angx1 = np.hstack((angx1, -np.arcsin(-cx1 / np.sqrt(ax1 ** 2 + bx1 ** 2)) - np.arcsin(ax1 / np.sqrt(ax1 ** 2 + bx1 ** 2))))
        angx2 = np.hstack((angx2, -np.arcsin(-cx2 / np.sqrt(ax2 ** 2 + bx2 ** 2)) - np.arcsin(ax2 / np.sqrt(ax2 ** 2 + bx2 ** 2))))
        angy1 = np.hstack((angy1, -np.arcsin(-cy1 / np.sqrt(ay1 ** 2 + by1 ** 2)) - np.arcsin(ay1 / np.sqrt(ay1 ** 2 + by1 ** 2))))
        angy2 = np.hstack((angy2, -np.arcsin(-cy2 / np.sqrt(ay2 ** 2 + by2 ** 2)) - np.arcsin(ay2 / np.sqrt(ay2 ** 2 + by2 ** 2))))

        angx1 = np.hstack((angx1, -angx1[0]))
        angx2 = np.hstack((angx2, -angx2[0]))
        angy1 = np.hstack((angy1, -angy1[0]))
        angy2 = np.hstack((angy2, -angy2[0]))

        angx1 = np.hstack((angx1, -angx1[1]))
        angx2 = np.hstack((angx2, -angx2[1]))
        angy1 = np.hstack((angy1, -angy1[1]))
        angy2 = np.hstack((angy2, -angy2[1]))

        angx1pp = angx1 + np.pi
        angx2pp = angx2 + np.pi
        angy1pp = angy1 + np.pi
        angy2pp = angy2 + np.pi

        angx1mp = angx1 - np.pi
        angx2mp = angx2 - np.pi
        angy1mp = angy1 - np.pi
        angy2mp = angy2 - np.pi

        angx1 = np.hstack((angx1, angx1pp))
        angx2 = np.hstack((angx2, angx2pp))
        angy1 = np.hstack((angy1, angy1pp))
        angy2 = np.hstack((angy2, angy2pp))

        angx1 = np.hstack((angx1, angx1mp))
        angx2 = np.hstack((angx2, angx2mp))
        angy1 = np.hstack((angy1, angy1mp))
        angy2 = np.hstack((angy2, angy2mp))

        # # arctan approach:
        # angx1 = np.arcsin(-cx1 / np.sqrt(ax1 ** 2 + bx1 ** 2)) - np.arctan(ax1 / bx1)
        # angx2 = np.arcsin(-cx2 / np.sqrt(ax2 ** 2 + bx2 ** 2)) - np.arctan(ax2 / bx2)
        # angy1 = np.arcsin(-cy1 / np.sqrt(ay1 ** 2 + by1 ** 2)) - np.arctan(ay1 / by1)
        # angy2 = np.arcsin(-cy2 / np.sqrt(ay2 ** 2 + by2 ** 2)) - np.arctan(ay2 / by2)

        # # arctan2 approach:
        # angx1 = np.arcsin(-cx1 / np.sqrt(ax1 ** 2 + bx1 ** 2)) - np.arctan2(ax1, bx1)
        # angx2 = np.arcsin(-cx2 / np.sqrt(ax2 ** 2 + bx2 ** 2)) - np.arctan2(ax2, bx2)
        # angy1 = np.arcsin(-cy1 / np.sqrt(ay1 ** 2 + by1 ** 2)) - np.arctan2(ay1, by1)
        # angy2 = np.arcsin(-cy2 / np.sqrt(ay2 ** 2 + by2 ** 2)) - np.arctan2(ay2, by2)

        if test:  # used for returning values of positive angles without assessment of negative angles
            if self.in_sight:
                ang1 = angy1[0]
                ang2 = angy2[0]
                ang3 = angx1[0]
                ang4 = angx2[0]
            else:
                ang1 = np.nan
                ang2 = np.nan
                ang3 = np.nan
                ang4 = np.nan
        else:  # assess result of positive and negative angles to see which results in a rotation constraint of 0 when applied
            cangx1 = np.zeros(len(angx1))
            cangx2 = np.zeros(len(angx1))
            cangy1 = np.zeros(len(angx1))
            cangy2 = np.zeros(len(angx1))
            for i in range(len(angx1)):
                self.UpdatePertR(self.axis_angle_to_R(v, angy1[i]), visualize=False)
                cangy1[i], _, _, _ = self.CalculateCriticalAngles(v, test=True)  # right edge constraint
                self.UpdatePertR(self.axis_angle_to_R(v, angy2[i]), visualize=False)
                _, cangy2[i], _, _ = self.CalculateCriticalAngles(v, test=True)  # left edge constraint
                self.UpdatePertR(self.axis_angle_to_R(v, angx1[i]), visualize=False)
                _, _, cangx1[i], _ = self.CalculateCriticalAngles(v, test=True)  # top edge constraint
                self.UpdatePertR(self.axis_angle_to_R(v, angx2[i]), visualize=False)
                _, _, _, cangx2[i] = self.CalculateCriticalAngles(v, test=True)  # bottom edge constraint
                self.UpdatePertR(self.axis_angle_to_R(v, 0), visualize=False)

            ang1 = np.unique(angy1[(np.abs(cangy1) < 1e-14) * (np.abs(angy1) < np.pi)])
            ang2 = np.unique(angy2[(np.abs(cangy2) < 1e-14) * (np.abs(angy2) < np.pi)])
            ang3 = np.unique(angx1[(np.abs(cangx1) < 1e-14) * (np.abs(angx1) < np.pi)])
            ang4 = np.unique(angx2[(np.abs(cangx2) < 1e-14) * (np.abs(angx2) < np.pi)])
            if not all:
                angs = np.hstack((ang1, ang2, ang3, ang4))
                try:
                    ang1 = np.array([angs[np.where(angs < 0, angs, -np.inf).argmax()]])
                except:
                    ang1 = np.array([])
                try:
                    ang2 = np.array([angs[np.where(angs > 0, angs, np.inf).argmin()]])
                except:
                    ang2 = np.array([])
                ang3 = np.array([])
                ang4 = np.array([])
            else:
                pass

            # if np.abs(cangy1) < 1e-14:
            #     ang1 = angy1[0]
            # else:
            #     ang1 = -angy1[0]
            #
            # if np.abs(cangy2) < 1e-14:
            #     ang2 = angy2[0]
            # else:
            #     ang2 = -angy2[0]
            #
            # if np.abs(cangx1) < 1e-14:
            #     ang3 = angx1[0]
            # else:
            #     ang3 = -angx1[0]
            #
            # if np.abs(cangx2) < 1e-14:
            #     ang4 = angx2[0]
            # else:
            #     ang4 = -angx2[0]

        return [ang1, ang2, ang3, ang4]

    # calculates passive rotation matrix from axis angle rotation
    def axis_angle_to_R(self, ax, ang):
        R = np.array([[np.cos(ang) + ax[0, 0] ** 2 * (1 - np.cos(ang)),
                       ax[0, 0] * ax[1, 0] * (1 - np.cos(ang)) - ax[2, 0] * np.sin(ang),
                       ax[0, 0] * ax[2, 0] * (1 - np.cos(ang)) + ax[1, 0] * np.sin(ang)],
                      [ax[0, 0] * ax[1, 0] * (1 - np.cos(ang)) + ax[2, 0] * np.sin(ang),
                       np.cos(ang) + ax[1, 0] ** 2 * (1 - np.cos(ang)),
                       ax[1, 0] * ax[2, 0] * (1 - np.cos(ang)) - ax[0, 0] * np.sin(ang)],
                      [ax[0, 0] * ax[2, 0] * (1 - np.cos(ang)) - ax[1, 0] * np.sin(ang),
                       ax[1, 0] * ax[2, 0] * (1 - np.cos(ang)) + ax[0, 0] * np.sin(ang),
                       np.cos(ang) + ax[2, 0] ** 2 * (1 - np.cos(ang))]])
        return R.transpose()  # returns passive transform
