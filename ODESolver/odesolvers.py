#Packages:
from scipy.integrate import odeint
import numpy as np

def odesolver(derivative_function, x0, timevector, inputvector):
    """Function for integrating ODEs with varying inputs.

    Args:
        -derivative_function (function) - function that calculates the derivative of the state.
            function arguments:
                -x : numpy array (nx x ) - N state vectors
                -t : float - time
                -u : numpy array (nu x ) - N input vectors
        -x0 : numpy array (nx x ) - N initial state vectors
        -timevector : numpy array (N x ) - time vector, the state is evaluated at each of these times
        -inputvector : numpy array (nu x N) - N input vectors for each time in the time vector
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

def N_odesolver(derivative_function, x0, timevector, inputvector):
    """Function for integrating multiple(N) ODEs with varying inputs.

    Args:
        -derivative_function (function) - function that calculates the derivative of the state.
            function arguments:
                -x : numpy array (N x nx) - N state vectors
                -t : float - time
                -u : numpy array (N x nu) - N input vectors
        -x0 : numpy array (N x nx) - N initial state vectors
        -timevector : numpy array (T x ) - time vector, the state is evaluated at each of these times
        -inputvector : numpy array (N x nu x T) - N input vectors for each time in the time vector
    """

    T = timevector.shape[0]
    N = x0.shape[0]
    n = x0.shape[1]
    x = x0
    trajectory = np.zeros((N, n, T))
    trajectory[:,:,0] = x

    for i in range(N):
        for t in range(1, T):
            timestep = [timevector[t-1], timevector[t]]
            tempx = odeint(derivative_function, x[i,:], timestep, args=(inputvector[i,:,t-1],))
            x[i,:] = tempx[1, :]
            trajectory[i,:, t] = tempx[1, :]

    return trajectory