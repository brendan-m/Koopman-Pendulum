# Dynamics function for the pendulum, under control.
#
# in: 
#     x     - state 
#     u     - command
#     model - model struct
# 
# out:
#     xdot  - state change and derivatives
#
# Original author = Matthew J. Howard
# Converted to python = Brendan Michael
import numpy as np
def fn_dynamics_pendulum(x,u,model):
    l   = model['L1']
    m   = model['M1']
    mu  = model['mu']
    g   = model['G']
    qdot  = x[1] # Velocity
    qddot = (g/l)*np.sin(x[0]) + (u-mu*x[1])/((m*l)**2) # Accelleration
    xdot  = [qdot, qddot]
    return xdot
