def model_pendulum(G=1, L=1, M=1, mu=0):
    """
    Returns a model of a non-linear pendulum.
    
    Parameters
    ----------
    G : float, optional
        The gravitational constant. Default value is 1.
    L : float, optional
        The length of the pendulum. Default value is 1.
    M : float, optional
        The mass of the pendulum. Default value is 1.
    mu : float, optional
        The damping coefficient. Default value is 0.
    
    Returns
    -------
    model : dict
        A dictionary containing the input parameters `G`, `L`, `M`, and `mu`.
    """
    model = {
        'G' : G,
        'L' : L,
        'M' : M,
        'mu': mu,
    }
    return model

def model_quadratic_attracting_manifold(lmbda=-1, mu=-0.05):
    """
    Returns a model of a quadratic attracting manifold.

    Parameters
    ----------
    lmbda : float, optional
        The constant that governs the nonlinear dynamics of the system. Default value is -1.
    mu : float, optional
        The coefficient that governs the linear dynamics of the system. Default value is -0.05.

    Returns
    -------
    model : dict
        A dictionary containing the input parameters `lmbda` and `mu`.
    """
    model = {
        'lambda': lmbda,
        'mu': mu,
    }
    return model

def model_duffing_oscillator(alpha=1, beta=1, delta=0, gamma=0, omega=0):
    """
    Returns a model of a Duffing oscillator.
    
    Can be used to describe a linear system by setting linear stiffness alpha=1, and rest of variables to 0
    Can be used to describe a simple non-linear system by setting alpha=1, beta=1, and rest of variables to 0
    
    Parameters
    ----------
    alpha : float
        The linear stiffness parameter of the oscillator.
    beta : float
        The nonlinearity parameter of the oscillator.
    delta : float
        The damping parameter of the oscillator.
    gamma : float
        The amplitude of the external forcing term.
    omega : float
        The frequency of the external forcing term.
    
    Returns
    -------
    model : dict
        A dictionary containing the input parameters `alpha`, `beta`, `delta`, `gamma`, and `omega`.
    """
    model = {
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'gamma': gamma,
        'omega': omega,
    }
    return model
    