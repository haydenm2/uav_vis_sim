import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    def __init__(self, x0=np.array([[10], [10], [40]]), xt0=np.array([[0], [0], [0]]), ypr=np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]), ypr_g=np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]), h_fov=np.deg2rad(45), v_fov=np.deg2rad(45)):
        # UAV and target conditions
        self.x = x0     # UAV state
        self.xt = xt0     # target state
        self.ypr = ypr   # UAV attitude (yaw, pitch, roll)
        self.ypr_g = ypr_g   # gimbal attitude (yaw, pitch, roll)
        self.h_fov = h_fov     # horizontal field of view
        self.v_fov = v_fov     # vertical field of view
        self.R_perturb = rot3dxp(0)

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
        self.R_vb = rot3dxp(self.ypr[2]) @ rot3dyp(self.ypr[1]) @ rot3dzp(self.ypr[0])      # rotation matrix from vehicle to body frame
        self.R_bg = self.R_perturb @ rot3dxp(self.ypr_g[2]) @ rot3dyp(self.ypr_g[1]) @ rot3dzp(self.ypr_g[0])      # rotation matrix from body to gimbal frame
        self.R_gc = np.eye(3)      # rotation matrix from gimbal to camera frame
        self.R_cfov1_x = rot3dyp(self.h_fov / 2)      # rotation matrix from camera to max x field of view frame
        self.R_cfov2_x = rot3dyp(-self.h_fov / 2)      # rotation matrix from camera to min x field of view frame
        self.R_cfov1_y = rot3dxp(self.v_fov / 2)      # rotation matrix from camera to max y field of view frame
        self.R_cfov2_y = rot3dxp(-self.v_fov / 2)      # rotation matrix from camera to min y field of view frame
        self.R_ic = self.R_perturb @ self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv

        # Rotate animation lines and points back out to common inertial frame for plotting
        llos = np.hstack([self.x, self.xt])     # line of sight vector
        lx = np.hstack((np.zeros((3, 1)), self.e1))*self.x_line_scale
        lquiv = np.hstack((np.zeros((3, 1)), self.e1)) * self.uav_length
        lquiv = self.R_iv.transpose() @ self.R_vb.transpose() @ lquiv + self.x      # transformed UAV heading arrow
        lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ lx + self.x      # transformed camera x-axis line
        lbNorth = self.R_iv.transpose() @ lx + self.x      # transformed North heading line
        L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                       self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                       self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                       self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))       # field of view corner lines
        L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ L                # field of view corner lines in vehicle frame
        pts = self.R_iv @ (self.x[2] * L_v @ np.linalg.inv(np.diag((self.e3.transpose() @ L_v)[0]))) + self.x     # field of view corners projected on observation plane
        lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))       # field of view corner line 1
        lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))       # field of view corner line 2
        lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))       # field of view corner line 3
        lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))       # field of view corner line 4

        l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ self.e3
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
        # self.quiv = self.ax.plot3D(lquiv[0, :], lquiv[1, :], lquiv[2, :], 'b-', LineWidth=5)       # plot UAV arrow

        self.UpdateAirCraftModel(init=True)

        # ----------------------- Initialize Camera View Plot -----------------------

        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)

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
    def UpdateX(self, x, visualize=True):
        self.x = x.reshape(-1, 1)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update target position
    def UpdateTargetX(self, xt, visualize=True):
        self.xt = xt.reshape(-1, 1)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV orientation
    def UpdateYPR(self, ypr, visualize=True):
        self.ypr = ypr
        self.R_vb = rot3dxp(self.ypr[2]) @ rot3dyp(self.ypr[1]) @ rot3dzp(self.ypr[0])
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV gimbal orientation
    def UpdateGimbalYPR(self, ypr_g, visualize=True):
        self.ypr_g = ypr_g
        self.R_bg = self.R_perturb @ rot3dxp(self.ypr_g[2]) @ rot3dyp(self.ypr_g[1]) @ rot3dzp(self.ypr_g[0])
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # applies general perturbation to UAV camera view orientation
    def UpdatePertR(self, R, visualize=True):
        self.R_perturb = R
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV camera horizontal field of view limits
    def UpdateHFOV(self, h_fov, visualize=True):
        self.h_fov = h_fov
        self.R_cfov1_x = rot3dyp(self.h_fov / 2)
        self.R_cfov2_x = rot3dyp(-self.h_fov / 2)
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # update UAV camera vertical field of view limits
    def UpdateVFOV(self, v_fov, visualize=True):
        self.v_fov = v_fov
        self.R_cfov1_y = rot3dxp(self.v_fov / 2)
        self.R_cfov2_y = rot3dxp(-self.v_fov / 2)
        self.cam_lims = (self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3) / (self.e3.transpose() @ self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3)
        if visualize:
            self.UpdateSim()
        else:
            self.UpdateSim(visualize=False)

    # updates states and simulation
    def UpdateSim(self, visualize=True):
        # ----------------------- State Updates -----------------------
        self.R_ic = self.R_perturb @ self.R_gc @ self.R_bg @ self.R_vb @ self.R_iv
        self.P_i = self.R_ic @ (self.xt - self.x)
        self.p_i = self.P_i / (self.e3.transpose() @ self.P_i)  # target point on normalized image plane
        self.CheckInFOV()

        if visualize:
            # ----------------------- 3D Simulation Update -----------------------

            # Rotate animation lines and points back out to common inertial frame for plotting
            llos = np.hstack([self.x, self.xt])  # line of sight vector
            lx = np.hstack((np.zeros((3, 1)), self.e1))*self.x_line_scale
            lquiv = np.hstack((np.zeros((3, 1)), self.e1)) * self.uav_length
            lquiv = self.R_iv.transpose() @ self.R_vb.transpose() @ lquiv + self.x  # transformed UAV heading arrow
            lcamx = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ lx + self.x  # transformed camera x-axis line
            lbNorth = self.R_iv.transpose() @ lx + self.x  # transformed North heading line
            L = np.hstack((self.R_cfov1_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                           self.R_cfov2_y.transpose() @ self.R_cfov1_x.transpose() @ self.e3,
                           self.R_cfov2_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3,
                           self.R_cfov1_y.transpose() @ self.R_cfov2_x.transpose() @ self.e3))  # field of view corner lines
            L_v = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ L  # field of view corner lines in vehicle frame
            pts = self.R_iv @ (self.x[2] * L_v @ np.linalg.inv(np.diag((self.e3.transpose() @ L_v)[0]))) + self.x  # field of view corners projected on observation plane
            plpts = np.hstack((pts, pts[:, 0].reshape(-1, 1)))  # field of view extent points (with redundant final point)
            lfov1 = np.hstack((self.x, pts[:, 0].reshape(-1, 1)))  # field of view corner line 1
            lfov2 = np.hstack((self.x, pts[:, 1].reshape(-1, 1)))  # field of view corner line 2
            lfov3 = np.hstack((self.x, pts[:, 2].reshape(-1, 1)))  # field of view corner line 3
            lfov4 = np.hstack((self.x, pts[:, 3].reshape(-1, 1)))  # field of view corner line 4

            l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ self.e3
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
            # self.quiv[0].set_data_3d(lquiv[0, :], lquiv[1, :], lquiv[2, :])  # plot UAV arrow

            self.UpdateAirCraftModel()  # update UAV model plot

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
        points = self.R_iv.transpose() @ self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ self.R_gc @ self.R_bg @ points + self.x

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

        # update body frame z-axis vector plot
        l_temp = self.R_vb.transpose() @ self.R_bg.transpose() @ self.R_gc.transpose() @ self.R_perturb.transpose() @ self.R_gc @ self.R_bg @ self.e3
        pts_temp = self.R_iv @ (self.x[2] * l_temp @ np.linalg.inv(np.diag((self.e3.transpose() @ l_temp)[0]))) + self.x  # body z-axis line points on observation plane
        lbz = np.hstack((self.x, pts_temp.reshape(-1, 1)))  # z-axis body frame line
        self.plbz[0].set_data_3d(lbz[0, :], lbz[1, :], lbz[2, :])  # plot z-axis body frame line

    # determines if target line of sight projection is within camera limits
    def CheckInFOV(self, visualize=True):
        border_dist = self.cam_lims - np.abs(self.p_i)  # how much closer target projection point elements are to center of image than border limits
        border_threshold = 1e-10
        # is z-element of target vector in camera frame negative? -> behind camera = out of sight
        if self.P_i[2] < 0:
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False (Target Behind Camera)', color="red", fontweight='bold')
        # are any target vector component distances from image view borders NaN? -> invalid solution = out of sight
        elif np.any(np.isnan(border_dist)):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False (NaN)', color="red", fontweight='bold')
        # are any target projection point components beyond a given threshold distance of any image borders? -> outside of field of view
        elif np.any(border_dist < -border_threshold):
            self.in_sight = False
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: False', color="red", fontweight='bold')
        # are all target projection point components within a very small threshold distance from a border and within all other image borders? -> within field of view AND on a border
        elif np.all(border_dist >= -border_threshold) and np.any(np.abs(border_dist[0:2]) < border_threshold):
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: Border', color="orange", fontweight='bold')
        # are all target projection point components within all image borders? -> within field of view
        else:
            self.in_sight = True
            if visualize:
                self.axc.set_xlabel('y \n Target in sight: True', color="green", fontweight='bold')

    # determines distance of target projection on normalized image plane from critical boundaries
    def CheckBorderDistances(self):
        if self.in_sight:
            dr = self.cam_lims[1] - self.p_i[1]
            dl = -self.cam_lims[1] - self.p_i[1]
            dt = self.cam_lims[0] - self.p_i[0]
            db = -self.cam_lims[0] - self.p_i[0]
        else:
            dr = np.nan
            dl = np.nan
            dt = np.nan
            db = np.nan
        return [dr, dl, dt, db]

    # solves rodriguez axis angle rotation equation for angle given a desired axis and point location (i.e. solution to general form a*cos(ang) + b*sin(ang) + c = 0)
    def CalculateCriticalAngles(self, v, no_check=False, all=False):
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

        # stack possible solutions for critical angles on each image edge
        angy1 = -self.Wrap(np.array((2*np.arctan2((-by1 + np.sqrt(ay1**2 + by1**2 - cy1**2)), (cy1 - ay1)), 2*np.arctan2((-by1 - np.sqrt(ay1**2 + by1**2 - cy1**2)), (cy1 - ay1)))))
        angy2 = -self.Wrap(np.array((2*np.arctan2((-by2 + np.sqrt(ay2**2 + by2**2 - cy2**2)), (cy2 - ay2)), 2*np.arctan2((-by2 - np.sqrt(ay2**2 + by2**2 - cy2**2)), (cy2 - ay2)))))
        angx1 = -self.Wrap(np.array((2*np.arctan2((-bx1 + np.sqrt(ax1**2 + bx1**2 - cx1**2)), (cx1 - ax1)), 2*np.arctan2((-bx1 - np.sqrt(ax1**2 + bx1**2 - cx1**2)), (cx1 - ax1)))))
        angx2 = -self.Wrap(np.array((2*np.arctan2((-bx2 + np.sqrt(ax2**2 + bx2**2 - cx2**2)), (cx2 - ax2)), 2*np.arctan2((-bx2 - np.sqrt(ax2**2 + bx2**2 - cx2**2)), (cx2 - ax2)))))

        cangx1 = np.zeros(len(angx1))
        cangx2 = np.zeros(len(angx1))
        cangy1 = np.zeros(len(angx1))
        cangy2 = np.zeros(len(angx1))
        for i in range(len(angx1)):
            self.UpdatePertR(self.axis_angle_to_R(v, angy1[i]), visualize=False)
            cangy1[i], _, _, _ = self.CheckBorderDistances()  # right edge constraint
            self.UpdatePertR(self.axis_angle_to_R(v, angy2[i]), visualize=False)
            _, cangy2[i], _, _ = self.CheckBorderDistances()  # left edge constraint
            self.UpdatePertR(self.axis_angle_to_R(v, angx1[i]), visualize=False)
            _, _, cangx1[i], _ = self.CheckBorderDistances()  # top edge constraint
            self.UpdatePertR(self.axis_angle_to_R(v, angx2[i]), visualize=False)
            _, _, _, cangx2[i] = self.CheckBorderDistances()  # bottom edge constraint
            self.UpdatePertR(self.axis_angle_to_R(v, 0), visualize=False)

        # save angles whose perturbations result in new perturbation commands close to zero
        ang1 = angy1[(np.abs(cangy1) < 1e-10)]
        ang2 = angy2[(np.abs(cangy2) < 1e-10)]
        ang3 = angx1[(np.abs(cangx1) < 1e-10)]
        ang4 = angx2[(np.abs(cangx2) < 1e-10)]

        # determine the two angles whose solutions are closest to the current position (smallest magnitude angles on either side)
        if not all:
            angs = np.hstack((ang1, ang2, ang3, ang4))
            try:
                # get largest negative angle
                ang1 = np.array([angs[np.where(angs < 0, angs, -np.inf).argmax()]])
            except:
                ang1 = np.array([])
            try:
                # get smallest positive angle
                ang2 = np.array([angs[np.where(angs > 0, angs, np.inf).argmin()]])
            except:
                ang2 = np.array([])
            # if angles all have same sign then return both angles
            if ang1 == ang2:
                ang2 = angs[angs != ang1]
            # only max of two closest angles (one on either side) so others are cleared
            ang3 = np.array([])
            ang4 = np.array([])
        else:
            pass

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

    def Wrap(self, th):
        if type(th) is np.ndarray:
            th_wrap = np.fmod(th + np.pi, 2 * np.pi)
            for i in range(len(th_wrap)):
                if th_wrap[i] < 0:
                    th_wrap[i] += 2 * np.pi
        else:
            th_wrap = np.fmod(th + np.pi, 2 * np.pi)
            if th_wrap < 0:
                th_wrap += 2 * np.pi
        return th_wrap - np.pi



