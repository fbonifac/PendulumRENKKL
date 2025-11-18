#Packages:
import numpy as np

#Own code:
from ODESolver.odesolvers import odesolver, N_odesolver

class PendulumSystem():
    def __init__(self, x0, beta, l, g=9.81, C = np.array([[1, 0]])):
        """
        Class of the Pendulum system.

        Args:
            -x0 : numpy array (2 x ) - initial state of the system
            -beta : float - damping parameter of the system
            -l : float - length of the pendulum
            -g : float - gravitational constant
        """

        super().__init__()

        self.beta = beta
        self.l = l
        self.g = g
        self.x = x0
        self.C = C
        self.nx = 2
        self.ny = 1
        self.nu = 1

    def calculate_dx(self, x, t, u):
        """Function that calculates the derivative of the state given:
            -x : numpy array (2 x ) - state of the system
            -t : float - time
            -u : numpy array (1 x ) - input to the system"""
        x1, x2 = x
        dx1 = x2
        dx2 = u[0]/self.l - self.beta/self.l*x2 - self.g/self.l*np.sin(x1)
        dx = np.array([dx1, dx2])
        return dx

    def simulate(self, timevector, uvector):
        """Function that integrates the system over a time interval given by:
        -timevector : numpy array (N x ) - time intervals
        -uvector : numpy array (1 x N) - inputs to the system at the time intervals"""
        trajectory = odesolver(self.calculate_dx, self.x, timevector, uvector)
        self.x = trajectory[:,-1]
        return trajectory

    def output(self, x):
        """Function that returns the output of the system given by state"""
        y = np.matmul(self.C, x)
        return y



class N_PendulumSystem():
    def __init__(self, x0, beta, l, g=9.81):
        """
        Class of the Pendulum system.

        Args:
            -x0 : numpy array (N x 2) - initial state of the system
            -beta : float - damping parameter of the system
            -l : float - length of the pendulum
            -g : float - gravitational constant
            -C : numpy array (1 x 2) - output matrix of the system
        """

        super().__init__()

        self.beta = beta
        self.l = l
        self.g = g
        self.x = x0
        self.nx = 2
        self.ny = 1
        self.nu = 1

    def calculate_dx(self, x, t, u):
        """Function that calculates the derivative of the state given:
            -x : numpy array (2 x ) - state of the system
            -t : float - time
            -u : numpy array (1 x ) - input to the system"""
        dx1 = x[1]
        dx2 = u[0]/self.l - self.beta/self.l*x[1] - self.g/self.l*np.sin(x[0])
        dx = np.array([dx1, dx2])
        return dx

    def simulate(self, timevector, uvector):
        """Function that integrates the system over a time interval given by:
        -timevector : numpy array (T x ) - time intervals
        -uvector : numpy array (N x 1 x T) - inputs to the system at the time intervals"""
        trajectory = N_odesolver(self.calculate_dx, self.x, timevector, uvector)
        self.x = trajectory[:,:,-1]
        return trajectory

    def output(self, x):
        """Function that returns the output of the system given by state"""
        y = x[:,0:1,:]
        return y