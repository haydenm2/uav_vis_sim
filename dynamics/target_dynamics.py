"""
uav_dynamics
    - this file implements the dynamic equations of motion for UAV
    - use unit quaternion for the attitude state

"""
import sys
sys.path.append('..')
import numpy as np
from message_types.msg_state import msg_state
import parameters.target_parameters as TAR
from tools.tools import RotationVehicle2Body


class target_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 8x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, psi, r]
        self._state = np.array([[TAR.pn0],  # (0)
                                [TAR.pe0],  # (1)
                                [TAR.pd0],  # (2)
                                [TAR.u0],  # (3)
                                [TAR.v0],  # (4)
                                [TAR.w0],  # (5)
                                [TAR.psi0],  # (6)
                                [TAR.r0]])  # (7)
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = TAR.u0

        # initialize true_state message
        self.msg_true_state = msg_state()
        self._update_msg_true_state()

    ###################################
    # public functions
    def update_state(self):
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments()

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data()

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)

        u = state.item(3)
        v = state.item(4)
        w = state.item(5)

        psi = state.item(6)
        r = state.item(7)

        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        n = forces_moments.item(3)


        # position kinematics
        pn_dot = np.cos(psi)*u - np.sin(psi)*v
        pe_dot = np.sin(psi)*u + np.cos(psi)*v
        pd_dot = w

        # position dynamics
        u_dot = r*v + fx / TAR.mass
        v_dot = - r*u + fy / TAR.mass
        w_dot = fz / TAR.mass

        # rotatonal dynamics
        r_dot = 1.0/TAR.Jz * n

        # rotational kinematics
        psi_dot = r

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot, psi_dot, r_dot]]).T

        return x_dot

    def _update_velocity_data(self):
        Rvb = RotationVehicle2Body(0.0, 0.0, self._state.item(6))
        self.Vg = Rvb.transpose() @ self._state[3:6]  # In vehicle frame
        self._Vg = np.linalg.norm(self.Vg)

    def _forces_moments(self):
        mass = TAR.mass
        g = TAR.gravity
        r = self._state.item(7)
        Vg = self._Vg
        f_total = np.array((0.0, 0.0, 0.0))
        m_total = 0.0

        return np.array(
            [[f_total.item(0), f_total.item(1), f_total.item(2), 0.0]]).T

    def _update_msg_true_state(self):
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.u = -self._state.item(3)
        self.msg_true_state.v = -self._state.item(4)
        self.msg_true_state.w = -self._state.item(5)
        self.msg_true_state.psi = self._state.item(6)
        self.msg_true_state.r = self._state.item(7)
        self.msg_true_state.Vg = np.sqrt(self.Vg.item(0) ** 2 + self.Vg.item(1) ** 2 + self.Vg.item(2) ** 2)
