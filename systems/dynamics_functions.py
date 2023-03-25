import numpy as np

# Define the function that computes the dynamics of the system
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
    if not isinstance(x, np.ndarray) or x.shape != (2,):
        raise ValueError("x must be a numpy array of shape (2,)")
    if not isinstance(u, (int, float)):
        raise ValueError("u must be a scalar")
    
    qdot = x[1] 
    qddot = (-model['G']/model['L1']) * np.sin(x[0]) + (u-model['mu']*x[1])/((model['M1']*model['L1'])**2)  # Acceleration
    xdot = [qdot, qddot]
    return xdot

# Define the function that computes the dynamics of the system
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

    xdot = [model['mu']*x[0] + u , model['lambda']*(x[1]-x[0]**2)]
    return xdot