import numpy as np
import math

def quadratic_attracting_manifold_dynamics(x, u, model):
    """
    Compute the dynamics of a quadratic attracting manifold

    Parameters
    ----------
    x : array_like, shape (2,)
        Current state of the system
    u : float
        Control input (torque) applied to the pendulum.

    Returns
    -------
    xdot : array_like, shape (2,)
        Time derivative of the state x, computed using the given control input u.
    """
    if not isinstance(x, np.ndarray) or x.shape != (2,):
        raise ValueError("x must be a numpy array of shape (2,)")
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a scalar")
        
    mu    = model['mu']
    lmbda = model['lambda']

    xdot = [mu*x[0] + u , lmbda*(x[1]-x[0]**2)]
    return np.array(xdot)

def pendulum_dynamics(x, u, model):
    """
    Compute the dynamics of a non-linear pendulum.

    Parameters
    ----------
    x : array_like, shape (2,)
        Current state of the pendulum, where x[0] is the angle (in radians)
        and x[1] is the angular velocity (in radians per second).
    u : float
        Control input (torque) applied to the pendulum.

    Returns
    -------
    xdot : array_like, shape (2,)
        Time derivative of the state x, computed using the given control input u.
    """
#     if not isinstance(x, np.ndarray) or x.shape != (2,):
#         raise ValueError("x must be a numpy array of shape (2,)")
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a scalar")
    
    g  = model['G']
    l  = model['L']
    m  = model['M']
    mu = model['mu']
    
    qdot = x[...,1] 
    qddot = (-g/l) * np.sin(x[...,0]) + (u-mu*x[...,1])/((m*l)**2)  # Acceleration
    xdot = np.stack((qdot,qddot),axis=-1)
    return xdot

def duffing_oscillator(x, t, u, model):
    """
    Compute the dynamics of a duffing oscillator

    Parameters
    ----------
    x : array_like, shape (2,)
        Current state of the system
    t : float
        Current time
    u : float
        Control input (torque) applied to the oscillator.
    model : dict
        A dictionary containing the parameters alpha, beta, delta, gamma, and omega.

    Returns
    -------
    xdot : array_like, shape (2,)
        Time derivative of the state x, computed using the given control input u and model parameters.
    """
    if not isinstance(x, np.ndarray) or x.shape != (2,):
        raise ValueError("x must be a numpy array of shape (2,)")
    if not isinstance(t, (int, float)):
        raise ValueError("t must be a scalar")
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a scalar")
        
    alpha = model['alpha']
    beta  = model['beta']
    delta = model['delta']
    gamma = model['gamma']
    omega = model['omega']
    
    acceleration = alpha * x[0] - beta * x[0]**3 - delta * x[1] + gamma * np.cos(omega * t) + u
    return np.array([x[1], acceleration])