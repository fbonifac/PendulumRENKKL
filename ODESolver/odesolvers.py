#Packages:
from scipy.integrate import odeint
import numpy as np

def odesolver(derivative_function, x0, timevector, inputvector):
    """Function for integrating ODEs with varying inputs.

    Args:
        -derivative_function (function) - function that calculates the derivative of the state.
            function arguments:
                -x : numpy array (nx x ) - state vector
                -t : float - time
                -u : numpy array (nu x ) - input vector
        -x0 : numpy array (nx x ) - initial state vector
        -timevector : numpy array (N x ) - time vector, the state is evaluated at each of these times
        -inputvector : numpy array (nu x N) - input vector for each time in the time vector
    """

    N = timevector.shape[0]
    n = x0.shape[0]
    x = x0
    trajectory = np.zeros((n, N))
    trajectory[:,0] = x

    for t in range(1,N):
        timestep = [timevector[t-1], timevector[t]]
        x = odeint(derivative_function, x, timestep, args=(inputvector[:,t-1],))
        x = x[1, :]
        trajectory[:,t] = x

    return trajectory