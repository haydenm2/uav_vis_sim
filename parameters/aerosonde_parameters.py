import sys
sys.path.append('..')
import numpy as np
from tools.tools import Euler2Quaternion, Quaternion2Euler

######################################################################################
                #   Initial Conditions
######################################################################################
#   Initial conditions for MAV
pn0 = 0.  # initial north position
pe0 = 0.  # initial east position
pd0 = -50.0  # initial down position
u0 = 25.  # initial velocity along body x-axis
v0 = 0.  # initial velocity along body y-axis
w0 = 0.  # initial velocity along body z-axis
phi0 = 0.  # initial roll angle
theta0 = 0.  # initial pitch angle
psi0 = 0.0  # initial yaw angle
phig0 = 0.  # initial roll angle
thetag0 = 0.  # initial pitch angle
psig0 = 0.0  # initial yaw angle
p0 = 0  # initial roll rate
q0 = 0  # initial pitch rate
r0 = 0  # initial yaw rate
Va0 = np.sqrt(u0**2+v0**2+w0**2)
#   Quaternion State
e = Euler2Quaternion(phi0, theta0, psi0)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)

######################################################################################
                #   Camera/gimbal Parameters
######################################################################################
h_fov = np.deg2rad(45)
v_fov = np.deg2rad(45)
Jg = np.array((1.0, 1.0, 1.0))

######################################################################################
                #   Physical Parameters
######################################################################################
mass = 11.0 #kg
Jx = 0.8244 #kg m^2
Jy = 1.135
Jz = 1.759
Jxz = 0.1204
S_wing = 0.55
b = 2.8956
c = 0.18994
S_prop = 0.2027
rho = 1.2682
k_motor = 80
kTp = 0.0
kOmega = 0.0
e = 0.9
AR = (b**2) / S_wing
gravity = 9.8

######################################################################################
                #   Longitudinal Coefficients
######################################################################################
C_L_0 = 0.23
C_L_alpha = 5.61
C_L_q = 7.95
C_L_delta_e = 0.13
C_D_0 = 0.043
C_D_alpha = 0.03
C_D_p = 0.0
C_D_q = 0.0
C_D_delta_e = 0.0135
C_m_0 = 0.0135
C_m_alpha = -2.74
C_m_q = -38.21
C_m_delta_e = -0.99
C_prop = 1.0
M = 50.0
alpha0 = 0.4712
epsilon = 0.1592

######################################################################################
                #   Lateral Coefficients
######################################################################################
C_Y_0 = 0.0
C_Y_beta = -0.83
C_Y_p = 0.0
C_Y_r = 0.0
C_Y_delta_a = 0.075
C_Y_delta_r = 0.19
C_ell_0 = 0.0
C_ell_beta = -0.13
C_ell_p = -0.51
C_ell_r = 0.25
C_ell_delta_a = 0.17
C_ell_delta_r = 0.0024
C_n_0 = 0.0
C_n_beta = 0.073
C_n_p = -0.069
C_n_r = -0.095
C_n_delta_a = -0.011
C_n_delta_r = -0.069

######################################################################################
                #   Propeller thrust / torque parameters (see addendum by McLain)
######################################################################################
# Prop parameters
D_prop = 20*(0.0254)     # prop diameter in m

# Motor parameters
K_V = 145.                   # from datasheet RPM/V
KQ = (1. / K_V) * 60. / (2. * np.pi)  # KQ in N-m/A, V-s/rad
R_motor = 0.042              # ohms
i0 = 1.5                     # no-load (zero-torque) current (A)


# Inputs
ncells = 12.
V_max = 3.7 * ncells  # max voltage for specified number of battery cells

# Coeffiecients from prop_data fit
C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357

######################################################################################
                #   Calculation Variables
######################################################################################
#   gamma parameters pulled from page 36 (dynamics)
gamma = Jx * Jz - (Jxz**2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz**2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz**2)) / gamma
gamma8 = Jx / gamma

#   C values defines on pag 62
C_p_0         = gamma3 * C_ell_0      + gamma4 * C_n_0
C_p_beta      = gamma3 * C_ell_beta   + gamma4 * C_n_beta
C_p_p         = gamma3 * C_ell_p      + gamma4 * C_n_p
C_p_r         = gamma3 * C_ell_r      + gamma4 * C_n_r
C_p_delta_a    = gamma3 * C_ell_delta_a + gamma4 * C_n_delta_a
C_p_delta_r    = gamma3 * C_ell_delta_r + gamma4 * C_n_delta_r
C_r_0         = gamma4 * C_ell_0      + gamma8 * C_n_0
C_r_beta      = gamma4 * C_ell_beta   + gamma8 * C_n_beta
C_r_p         = gamma4 * C_ell_p      + gamma8 * C_n_p
C_r_r         = gamma4 * C_ell_r      + gamma8 * C_n_r
C_r_delta_a    = gamma4 * C_ell_delta_a + gamma8 * C_n_delta_a
C_r_delta_r    = gamma4 * C_ell_delta_r + gamma8 * C_n_delta_r

######################################################################################
                #   Trim States
######################################################################################
gamma_star = 0.0
Va_star = 25.0
alpha_star = 0.0
delta_a_star = -0.00388821
delta_e_star = -0.10919994
delta_r_star = -0.00716267
delta_t_star = 0.78144714

x_trim = np.array( [[-2.98038726e-14],
 [ 2.55050479e-15],
 [-2.00000000e+01],
 [ 2.49156741e+01],
 [ 0.00000000e+00],
 [ 2.05163008e+00],
 [ 9.99156499e-01],
 [ 0.00000000e+00],
 [ 4.10644759e-02],
 [ 0.00000000e+00],
 [ 0.00000000e+00],
 [ 0.00000000e+00],
 [ 0.00000000e+00]])

[phi_star, theta_star, psi_star] = Quaternion2Euler(np.array([x_trim.item(6), x_trim.item(7), x_trim.item(8), x_trim.item(9)]))