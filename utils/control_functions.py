import numpy as np

def compute_control_input(x, target_state, control_type='none', gains=None):
    """
    Compute the control input for a non-linear system.

    Parameters
    ----------
    x : array_like, shape (n,)
        Current state of the system.
    target_state : array_like, shape (n,)
        Target state for the system to track.
    control_type : str, optional
        Type of control to use. Possible values are 'none', 'pd', or 'lqr'.
        Defaults to 'none'.
    gains : dict, optional
        Dictionary of controller gains. If None, default gains will be used.
        Defaults to None.

    Returns
    -------
    u : float or array_like
        Control input to be applied to the system.
    """
    n = len(x)  # Number of variables in the state
    if not isinstance(x, np.ndarray) or x.shape != (n,):
        raise ValueError("x must be a numpy array of shape ({},)".format(n))
    if not isinstance(target_state, np.ndarray) or target_state.shape != (n,):
        raise ValueError("target_state must be a numpy array of shape ({},)".format(n))
    if not isinstance(control_type, str):
        raise ValueError("control_type must be a string")

    if control_type == 'pd':
        # PD control
        if gains is None:
            # Default gains
            Kp = 10
            Kd = 3
        else:
            # User-specified gains
            Kp = gains['Kp']
            Kd = gains['Kd']
        u = -Kp * (x[0] - target_state[0]) - Kd * (x[1] - target_state[1])
    elif control_type == 'none':
        # No control
        u = 0
    else:
        raise ValueError("Invalid control_type: {}".format(control_type))

    return u
