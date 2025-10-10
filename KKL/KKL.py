#Packages:
import numpy as np

#Own code:
from ODESolver.odesolvers import odesolver

class KKLSystem():
    def __init__(self, z0, A, B):
        """
        Class of the KKL observer system.

        Args:
            -z0 : numpy array (nz x ) - initial state of the system
            -A : numpy array (nz x nz) - state matrix of the linear system
            -B : numpy array (nz x ny) - input matrix of the linear system
        """

        super().__init__()

        self.z = z0
        self.A = A
        self.B = B

    def calculate_dz(self, z, t, y):
        """Function that calculates the derivative of the state given:
            -z : numpy array (nz x ) - state of the system
            -t : float - time
            -y : numpy array (ny x ) - input to the system"""
        dz = np.matmul(self.A, z) + np.matmul(self.B, y)
        return dz

    def simulate(self, timevector, yvector):
        """Function that integrates the system over a time interval given by:
        -timevector : numpy array (N x ) - time intervals
        -uvector : numpy array (ny x N) - inputs to the system at the time intervals"""
        trajectory = odesolver(self.calculate_dz, self.z, timevector, yvector)
        self.z = trajectory[:,-1]
        return trajectory