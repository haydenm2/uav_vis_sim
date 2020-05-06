"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:
        12/27/2018 - RWB
        1/17/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM
from message_types.msg_state import msg_state

from viewer.viewer import vision_uav_viewer
from dynamics.uav_dynamics import uav_dynamics
from dynamics.wind_simulation import wind_simulation
import parameters.aerosonde_parameters as UAV
import parameters.target_parameters as TAR

# initialize the visualization
uav_view = vision_uav_viewer(x0=np.array([[UAV.pn0], [UAV.pe0], [UAV.pd0]]),
                             xt0=np.array([[TAR.pn0], [TAR.pe0], [TAR.pd0]]),
                             ypr=np.array([UAV.psi0, UAV.theta0, UAV.phi0]),
                             ypr_g=np.array([UAV.psig0, UAV.thetag0, UAV.phig0]),
                             h_fov=UAV.h_fov,
                             v_fov=UAV.v_fov)  # initialize the mav viewer

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
uav = uav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# crash elements
crash_flag = False
crash_state = msg_state()

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    if sim_time < SIM.end_time/6:  #increase altitude
        delta_a = 0.018  # 0.0
        delta_e = -0.1  # -0.2
        delta_r = 0.0  # 0.005
        delta_t = 1.0  # 0.5
    elif sim_time < 2 * SIM.end_time/8:
        delta_a = -0.018  # 0.0
        delta_e = -0.2  # -0.2
        delta_r = 0.00  # 0.005
        delta_t = 1.0  # 0.5
    elif sim_time < 3 * SIM.end_time/8:
        delta_a = -0.03  # 0.0
        delta_e = -0.08  # -0.2
        delta_r = -0.01  # 0.005
        delta_t = 1.0  # 0.5
    else:
        delta_a = -0.018  # 0.0
        delta_e = -0.08  # -0.2
        delta_r = -0.01  # 0.005
        delta_t = 1.0  # 0.5
    # -------set control surfaces-------------
    delta = np.array([[delta_a, delta_e, delta_r, delta_t]]).T  # transpose to make it a column vector

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    uav.update_state(delta, current_wind)  # propagate the MAV dynamics
    if not crash_flag:
        uav_state = uav.msg_true_state
        if uav_state.h <= 0:
            uav_state.h = 0
            crash_flag = True
            crash_state.pn = uav_state.pn
            crash_state.pe = uav_state.pe
            crash_state.h = uav_state.h
            crash_state.phi = uav_state.phi
            crash_state.theta = uav_state.theta
            crash_state.psi = uav_state.psi
            crash_state.phi_g = uav_state.phi_g
            crash_state.theta_g = uav_state.theta_g
            crash_state.psi_g = uav_state.psi_g
    else:
        uav_state = crash_state

    #-------update viewer-------------
    if sim_time % SIM.ts_plotting < SIM.ts_simulation:
        uav_view.UpdateUAV(np.array((uav_state.pn, uav_state.pe, -uav_state.h)), np.array((uav_state.psi, uav_state.theta, uav_state.phi)), np.array((uav_state.psi_g, uav_state.theta_g, uav_state.phi_g)))  # update UAV viewer
        uav_view.UpdateTarget(np.array((0.0, 0.0, 0.0)))  # update target viewer

    #-------increment time-------------
    sim_time += SIM.ts_simulation



