#Packages:
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

class REN(nn.Module):
    def __init__(self, nx, nq, nu, ny, sigma = 'relu', epsilon = 1.0e-2, device = 'cuda', bias = False):
        """
        Class of Reccurent Equilibrium Networks.

        Args:
            -nx : int - number of internal-states
            -nq : int - number of non-linear states
            -nu : int - number of inputs
            -ny : int - number of outputs
            -sigma : string - activation function sigma(), default: tanh
            -epsilon : float - small positive number to ensure the matrices are positve-definite, default: 1e-2
            -device : string - device to be used for the computations using Pytorch
            -bias : bool - whether to use bias or not, default: False
        """

        super().__init__()

        self.nx = nx
        self.nq = nq
        self.nu = nu
        self.ny = ny
        self.epsilon = epsilon
        self.device = device

        std = 0.5
        # Initialization of the Free Matrices:
        self.X = nn.Parameter(torch.randn(nx + nq, nx + nq, requires_grad=True, device=device) * std)
        self.U = nn.Parameter(torch.randn(nx, nq, requires_grad=True, device=device) * std)
        self.Y1 = nn.Parameter(torch.randn(nx, nx, requires_grad=True, device=device) * std)
        self.XP = nn.Parameter(torch.randn(nx, nx, requires_grad=True, device=device) * std)
        # Initialization of the Weights:
        self.B2 = nn.Parameter(torch.randn(nx, nu, requires_grad=True, device=device) * std)
        self.C2 = nn.Parameter(torch.randn(ny, nx, requires_grad=True, device=device) * std)
        self.D12 = nn.Parameter(torch.randn(nq, nu, requires_grad=True, device=device) * std)
        self.D21 = nn.Parameter(torch.randn(ny, nq, requires_grad=True, device=device) * std)
        D22yKLL = nn.Parameter(torch.randn(ny, nu-1, requires_grad=True, device=device) * std)
        D22u = torch.zeros(ny, 1, device=device)
        self.D22 = torch.cat((D22u, D22yKLL), dim=1)

        # Initialization of the Biases:
        if bias:
            self.bx = nn.Parameter(torch.randn(nx, requires_grad=True, device=device)*std)
            self.bv = nn.Parameter(torch.randn(nq, requires_grad=True, device=device)*std)
            self.by = nn.Parameter(torch.randn(ny, requires_grad=True, device=device)*std)
        else:
            self.bx = torch.zeros(nx, device=device)
            self.bv = torch.zeros(nq, device=device)
            self.by = torch.zeros(ny, device=device)

        #Initialization of the system matrices:
        self.A = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.P = torch.zeros(nx, nx, device=device)

        self.updateParameters()             #Update of: A, B1, C1, D11, D12, D22

        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Updates the value of A, D11, C1, B1, P. These are constrained matrices so when
        learning with the unconstrained matrices these need to be updated, since the forward calculations
        are based on these.
        This function is called after each batch."""
        P = F.linear(self.XP, self.XP) + self.epsilon*torch.eye(self.nx, self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon*torch.eye(self.nx + self.nq, self.nx + self.nq, device=self.device)

        #Partition H into H11, H12, H21, H22
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H11, H12 = torch.split(h1, (self.nx, self.nq), dim=1)
        H21, H22 = torch.split(h2, (self.nx, self.nq), dim=1)

        #Calculate A, D11, C1, B1
        Y = -0.5*(H11 + self.Y1 - self.Y1.T)
        self.A = F.linear(torch.inverse(P), Y.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H22))
        self.D11 = -F.linear(torch.inverse(Lambda), torch.tril(H22,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda), self.U)
        Z = -H12 - self.U
        self.B1 = F.linear(torch.inverse(P), Z.T)

        #Check if the LMI holds:
        """with torch.no_grad():
            temp1 = torch.cat((-F.linear(self.A.T, P.T) - F.linear(P.T, self.A.T), -F.linear(self.C1.T, Lambda.T) - F.linear(P, self.B1.T)), 1)
            temp2 = torch.cat(((-F.linear(self.C1.T, Lambda.T) - F.linear(P, self.B1.T)).T, 2*Lambda - F.linear(Lambda, self.D11.T) - F.linear(self.D11.T, Lambda.T)), 1)
            check_matrix = torch.cat((temp1, temp2), 0)
            if not bool((check_matrix == check_matrix.T).all() and (torch.linalg.eigvals(check_matrix).real > 0).all()):
                print("PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!")"""

    def calculate_w(self, t, x, u):
        """Given x(t), u(t) calculates w(t). (Solution of w(t) = sigma(C1*x(t) + D11*w(t) + D12*u(t) + b_v))
        This calculation exploits the fact that the matrix D11 is a lower triangular matrix."""

        w = torch.zeros(self.nq, device=self.device)
        # Calculate the rest of w(t)
        for i in range(self.nq):
            select_w = torch.zeros(self.nq, device=self.device)
            select_w[i] = 1.
            v = F.linear(x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + self.bv[i]
            w = w + torch.mul(self.act(v), select_w)
        return w

    def forward(self, t, x, u):
        """Given x(t), u(t) calculates x_dot(t). (x_dot(t) = A*x(t) + B1*w(t) + B2*u(t) + b_x)"""
        w  = self.calculate_w(t, x, u)
        x_dot = F.linear(x, self.A) + F.linear(w, self.B1) + F.linear(u, self.B2)
        return x_dot

    def output(self, t, x, u):
        """Given x(t), u(t) calculates y(t). (y(t) = C2*x(t) + D21*w(t) + D22*u(t) + b_y)"""
        w = self.calculate_w(t, x, u)
        y = F.linear(x, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22)
        return y

class RENSystem(nn.Module):
    def __init__(self, nx, nq, nu, ny, sigma='relu', epsilon=1.0e-2, device='cuda', bias=False):

        super().__init__()
        self.sys = REN(nx, nq, nu, ny, sigma, epsilon, device, bias)
        self.nfe = 0
        self.x = torch.zeros(nx, device=device)
        self.u = torch.zeros(nu, device=device)
        self.input_output_matrix = torch.eye(nu, device=device)
        self.input_output_matrix[0,0] = 0

    def updateParameters(self):
        self.sys.updateParameters()

    def forward(self, t, x):
        self.nfe += 1
        x_dot = self.sys(t, x, self.u)
        return x_dot

    def output(self, t, x):
        self.nfe += 1
        y = self.sys.output(t, x, self.u)
        return y

    def simulate(self, timevector, inputvector, initial_condition=None):
        """Given inputs over times in the time vector the function simulates the system and returns the output
        over the given timevector."""

        if initial_condition is not None:
            self.x = initial_condition

        inputvector_output = torch.matmul(inputvector.T, self.input_output_matrix).T

        N = timevector.shape[0]
        output = torch.zeros(self.sys.ny, N, device=self.sys.device)
        self.u = inputvector_output[:, 0]
        output[:, 0] = self.output(timevector[0], self.x)

        for t in range(1, N):
            timestep = timevector[t-1:t+1]
            self.u = inputvector[:, t] #Zero-order hold?
            temp_x = odeint(self, self.x, timestep, method='dopri5', rtol=1.0e-3, atol=1.0e-5, adjoint_rtol=1.0e-3, adjoint_atol=1.0e-5)
            self.x = temp_x[1, :]
            self.u = inputvector_output[:, t]
            output[:, t] = self.output(timevector[t], self.x)
        return output

    def nfe(self):
        """Number of forward evaluations"""
        return self.nfe


class N_REN(nn.Module):
    def __init__(self, nx, nq, nu, ny, sigma = 'relu', epsilon = 1.0e-2, device = 'cuda', bias = False):
        """
        Class of Reccurent Equilibrium Networks.

        Args:
            -nx : int - number of internal-states
            -nq : int - number of non-linear states
            -nu : int - number of inputs
            -ny : int - number of outputs
            -sigma : string - activation function sigma(), default: tanh
            -epsilon : float - small positive number to ensure the matrices are positve-definite, default: 1e-2
            -device : string - device to be used for the computations using Pytorch
            -bias : bool - whether to use bias or not, default: False
        """

        super().__init__()

        self.nx = nx
        self.nq = nq
        self.nu = nu
        self.ny = ny
        self.epsilon = epsilon
        self.device = device

        std = 0.5
        # Initialization of the Free Matrices:
        self.X = nn.Parameter(torch.randn(nx + nq, nx + nq, requires_grad=True, device=device) * std)
        self.U = nn.Parameter(torch.randn(nx, nq, requires_grad=True, device=device) * std)
        self.Y1 = nn.Parameter(torch.randn(nx, nx, requires_grad=True, device=device) * std)
        self.XP = nn.Parameter(torch.randn(nx, nx, requires_grad=True, device=device) * std)
        # Initialization of the Weights:
        self.B2 = nn.Parameter(torch.randn(nx, nu, requires_grad=True, device=device) * std)
        self.C2 = nn.Parameter(torch.randn(ny, nx, requires_grad=True, device=device) * std)
        self.D12 = nn.Parameter(torch.randn(nq, nu, requires_grad=True, device=device) * std)
        self.D21 = nn.Parameter(torch.randn(ny, nq, requires_grad=True, device=device) * std)
        D22yKLL = nn.Parameter(torch.randn(ny, nu-1, requires_grad=True, device=device) * std)
        D22u = torch.zeros(ny, 1, device=device)
        self.D22 = torch.cat((D22u, D22yKLL), dim=1)

        # Initialization of the Biases:
        if bias:
            self.bx = nn.Parameter(torch.zeros(nx, requires_grad=True, device=device))
            self.bv = nn.Parameter(torch.zeros(nq, requires_grad=True, device=device))
            self.by = nn.Parameter(torch.zeros(ny, requires_grad=True, device=device))
            #self.bx = nn.Parameter(torch.randn(nx, requires_grad=True, device=device)*std)
            #self.bv = nn.Parameter(torch.randn(nq, requires_grad=True, device=device)*std)
            #self.by = nn.Parameter(torch.randn(ny, requires_grad=True, device=device)*std)
        else:
            self.bx = torch.zeros(nx, device=device)
            self.bv = torch.zeros(nq, device=device)
            self.by = torch.zeros(ny, device=device)

        #Initialization of the system matrices:
        self.A = torch.zeros(nx, nx, device=device)
        self.D11 = torch.zeros(nq, nq, device=device)
        self.C1 = torch.zeros(nq, nx, device=device)
        self.B1 = torch.zeros(nx, nq, device=device)
        self.P = torch.zeros(nx, nx, device=device)

        self.updateParameters()             #Update of: A, B1, C1, D11, D12, D22

        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Updates the value of A, D11, C1, B1, P. These are constrained matrices so when
        learning with the unconstrained matrices these need to be updated, since the forward calculations
        are based on these.
        This function is called after each batch."""
        P = F.linear(self.XP, self.XP) + self.epsilon*torch.eye(self.nx, self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon*torch.eye(self.nx + self.nq, self.nx + self.nq, device=self.device)

        #Partition H into H11, H12, H21, H22
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H11, H12 = torch.split(h1, (self.nx, self.nq), dim=1)
        H21, H22 = torch.split(h2, (self.nx, self.nq), dim=1)

        #Calculate A, D11, C1, B1
        Y = -0.5*(H11 + self.Y1 - self.Y1.T)
        self.A = F.linear(torch.inverse(P), Y.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H22))
        self.D11 = -F.linear(torch.inverse(Lambda), torch.tril(H22,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda), self.U)
        Z = -H12 - self.U
        self.B1 = F.linear(torch.inverse(P), Z.T)

        #Check if the LMI holds:
        """with torch.no_grad():
            temp1 = torch.cat((-F.linear(self.A.T, P.T) - F.linear(P.T, self.A.T), -F.linear(self.C1.T, Lambda.T) - F.linear(P, self.B1.T)), 1)
            temp2 = torch.cat(((-F.linear(self.C1.T, Lambda.T) - F.linear(P, self.B1.T)).T, 2*Lambda - F.linear(Lambda, self.D11.T) - F.linear(self.D11.T, Lambda.T)), 1)
            check_matrix = torch.cat((temp1, temp2), 0)
            if not bool((check_matrix == check_matrix.T).all() and (torch.linalg.eigvals(check_matrix).real > 0).all()):
                print("PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!")"""

    def calculate_w(self, t, x, u):
        """Given x(t), u(t) calculates w(t). (Solution of w(t) = sigma(C1*x(t) + D11*w(t) + D12*u(t) + b_v))
        This calculation exploits the fact that the matrix D11 is a lower triangular matrix."""

        N = x.shape[0]
        w = torch.zeros(N, self.nq, device=self.device)
        # Calculate the rest of w(t)
        for i in range(self.nq):
            select_w = torch.zeros(self.nq, 1, device=self.device)
            select_w[i,0] = 1.
            v = (F.linear(x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + self.bv[i]).unsqueeze(1)
            w = w + F.linear(self.act(v), select_w)
        return w

    def forward(self, t, x, u):
        """Given x(t), u(t) calculates x_dot(t). (x_dot(t) = A*x(t) + B1*w(t) + B2*u(t) + b_x)"""
        w  = self.calculate_w(t, x, u)
        x_dot = F.linear(x, self.A) + F.linear(w, self.B1) + F.linear(u, self.B2)
        return x_dot

    def output(self, t, x, u):
        """Given x(t), u(t) calculates y(t). (y(t) = C2*x(t) + D21*w(t) + D22*u(t) + b_y)"""
        w = self.calculate_w(t, x, u)
        y = F.linear(x, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22)
        return y

class N_RENSystem(nn.Module):
    def __init__(self, nx, nq, nu, ny, N, sigma='relu', epsilon=1.0e-2, device='cuda', bias=False):

        super().__init__()
        self.sys = N_REN(nx, nq, nu, ny, sigma, epsilon, device, bias)
        self.nfe = 0
        self.x = torch.zeros(N, nx, device=device)
        self.u = torch.zeros(N, nu, device=device)

    def updateParameters(self):
        self.sys.updateParameters()

    def forward(self, t, x):
        self.nfe += 1
        x_dot = self.sys(t, x, self.u)
        return x_dot

    def output(self, t, x):
        self.nfe += 1
        y = self.sys.output(t, x, self.u)
        return y

    def simulate(self, timevector, inputvector, initial_condition=None):
        """Given inputs over times in the time vector the function simulates the system and returns the output
        over the given timevector."""

        if initial_condition is not None:
            self.x = initial_condition

        T = timevector.shape[0]
        N = self.x.shape[0]

        inputvector_output = inputvector
        inputvector_output[:, 0, :] = torch.zeros((N, T), device=self.sys.device)

        output = torch.zeros(N, self.sys.ny, T, device=self.sys.device)
        self.u = inputvector_output[:, :, 0]
        output[:, :, 0] = self.output(timevector[0], self.x)

        for t in range(1, T):
            timestep = timevector[t-1:t+1]
            self.u = inputvector[:, :, t] #Zero-order hold?

            #temp_x = odeint(self, self.x, timestep, method='euler', options=dict(stepsize=(timevector[1]-timevector[0])))
            temp_x = odeint(self, self.x, timestep, method='dopri5', atol=1.0e-5, rtol=1.0e-3, adjoint_atol=1.0e-5, adjoint_rtol=1.0e-3)
            self.x = temp_x[1, :, :]
            self.u = inputvector_output[:, :, t]
            output[:, :, t] = self.output(timevector[t], self.x)
        return output

    def nfe(self):
        """Number of forward evaluations"""
        return self.nfe