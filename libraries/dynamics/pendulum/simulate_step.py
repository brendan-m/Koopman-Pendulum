# Make one simulation step
# 
#     xn = simulate_step ( f, x, u, p )
# 
# in:
#    f  - dynamics (function handle)
#    x  - initial state
#    u  - command
#    p  - parameter struct containing:
#         p.solver - numberical solver, chosen from {'euler','rk4'}
#         p.dt     - time step
# 
# out: 
#    xn - next state
#
# Original author = Matthew J. Howard
# Converted to python = Brendan Michael
import numpy as np

def simulate_step (f,x,u,model,dt,p):
    if p == 'euler':
        return x + np.multiply(dt,f(x,u,model)) # euler step       
    elif p == 'rk4'  : 
        a = np.multiply(dt,f(x,u,model))
        b = np.multiply(dt,f(x+.5*a,u,model))
        c = np.multiply(dt,f(x+.5*b,u,model))
        d = np.multiply(dt,f(x+   c,u,model))
        return x + (1/6)*(a + 2*b + 2*c + d)
