import sys
sys.path.append('..')
import numpy as np

######################################################################################
                #   Initial Conditions
######################################################################################
#   Initial conditions for MAV
pn0 = 0.  # initial north position
pe0 = 0.  # initial east position
pd0 = 0.0  # initial down position
u0 = 25.  # initial velocity along body x-axis
v0 = 0.  # initial velocity along body y-axis
w0 = 0.  # initial velocity along body z-axis
psi0 = 0.0  # initial heading angle
r0 = 0  # initial yaw rate
V0 = np.sqrt(u0**2+v0**2+w0**2)  # initial target velocity