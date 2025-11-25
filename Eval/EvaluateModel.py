#Packages:
import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt

#Own code:
from Pendulum.pendulum import N_PendulumSystem
from KKL.KKL import N_KKLSystem
from REN.REN import N_RENSystem, N_RENSystem_TORCHODE

def evaluate_model(T_test, test_function, experiment_folder='../Data/Experiment12/', plot=False):

    hyperparameters = np.load(experiment_folder + 'hyperparameters.npy', allow_pickle=True).item()

    nx = hyperparameters['nx']
    nq = hyperparameters['nq']
    nu = hyperparameters['nu']
    ny = hyperparameters['ny']
    nkkl = hyperparameters['nkkl']
    kkl_poles = hyperparameters['KKL_poles']
    bias = hyperparameters['bias']
    sigma = hyperparameters['activation_function']
    device = hyperparameters['device']
    dT = hyperparameters['sampling_time']
    N = hyperparameters['number_of_experiments']
    default_type = hyperparameters['variable_type']
    T_trans = hyperparameters['transient_time']

    integration_method = hyperparameters['integration_method']
    integration_steps = hyperparameters['fixed_number_of_steps']
    integration_tol = hyperparameters['integration_tolerance']
    integration_atol = hyperparameters['integration_absolute_tolerance']

    torch.set_default_dtype(default_type)

    training = torch.load(experiment_folder + 'model_and_optimizer.pt')


    """Initialize system that will be observed:
        -Initial states at: rand([1,1],[-1,-1])
        -Parameters of the system:
            -beta: 1.5
            -l: 0.5
            -g: 9.81
    """
    initial_pendulum_states = np.random.uniform(-1, 1, (N, 2))
    pendulum = N_PendulumSystem(initial_pendulum_states, 1.5, 0.5, 9.81)
    #Vales fixed to the pendulum
    nxsys = pendulum.nx
    nysys = pendulum.ny
    nusys = pendulum.nu

    """Initialize KKL system (z transformed system):
        -Initial states at: zeros((N, nkkl))
        -Parameters of the system:
            -A: diag(kkl_poles)
            -B: ones((nkkl, nusys))
    """
    #Load KKL poles:
    A = np.diag(kkl_poles)
    B = np.ones((nkkl, nysys))
    initial_KKL_states = np.zeros((N, nkkl), dtype=np.float32)
    kkl = N_KKLSystem(initial_KKL_states, A, B)

    """Initialize REN (Tau* transformation from (u, y, z) to (x)):
        -Parameters of the system:
            -nx: nx
            -nq: nq
            -nu: nu
            -ny: ny
    """
    model = N_RENSystem_TORCHODE(nx, nq, nu, ny, bias=bias, sigma=sigma, device=device)
    model.load_state_dict(training['model'])
    model.updateParameters()

    loss = 1.0
    MSE = nn.MSELoss()

    N_trans = int(T_trans / dT) + 1
    N_test = int(T_test / dT) + 1

    # ---------------------------------------TRANSIENT PHASE------------------------------------------------
    # The whole system (Pendulum+KKL+REN) is simulated for T_trans to line up

    # Create time vector and input vector for transient phase
    t_trans = np.linspace(0, T_trans, N_trans)
    u_trans = np.zeros((N, nusys, N_trans))

    # Simulate the pendulum, calculate output, simulate the KKL system, create the input for the REN, simulate the REN and save the last state
    with torch.no_grad():
        x_trans = pendulum.simulate(t_trans, u_trans)
        y_trans = pendulum.output(x_trans)
        z_trans = kkl.simulate(t_trans, y_trans)
        u_y_trans = np.append(u_trans, y_trans, axis=1)
        ren_u_trans = np.append(u_y_trans, z_trans, axis=1)

        t_trans_torch = torch.from_numpy(t_trans).float()
        t_trans_torch = t_trans_torch.to(device)
        ren_u_trans_torch = torch.from_numpy(ren_u_trans).float()
        ren_u_trans_torch = ren_u_trans_torch.to(device)

        ren_x_init = torch.zeros(N, nx, device=device)
        ren_y_trans = model.simulate(t_trans_torch, ren_u_trans_torch, ren_x_init,
                                     integration_method=integration_method, integration_steps=integration_steps,
                                     integration_tol=integration_tol, integration_atol=integration_atol)

    if plot:
        plt.plot(t_trans, x_trans[1, -1, :], linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_trans, ren_y_trans[1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')
        plt.title('Transient trajectories of the last experiment')
        plt.show()

    # ---------------------------------------TESTING PHASE------------------------------------------------
    # The whole system (Pendulum+KKL+REN) is simulated for T_test with input test_function to test the experiment

    # Create time vector and input vector for testing phase
    t_test = np.linspace(0, T_test, N_test)
    u_test = np.zeros((N, nusys, N_test))

    for i in range(N):
        u_test[i,:,:] = test_function(t_test)

    with torch.no_grad():
        x_test = pendulum.simulate(t_test, u_test)
        y_test = pendulum.output(x_test)
        z_test = kkl.simulate(t_test, y_test)
        u_y_test = np.append(u_test, y_test, axis=1)
        ren_u_test = np.append(u_y_test, z_test, axis=1)

        t_test_torch = torch.from_numpy(t_test).float()
        t_test_torch = t_test_torch.to(device)
        ren_u_test_torch = torch.from_numpy(ren_u_test).float()
        ren_u_test_torch = ren_u_test_torch.to(device)

        y_predict_test = x_test[:, nysys:, :]
        y_predict_test_torch = torch.from_numpy(y_predict_test).float()
        y_predict_test_torch = y_predict_test_torch.to(device)

        ren_y_test = model.simulate(t_test_torch, ren_u_test_torch, ren_x_init,
                                     integration_method=integration_method, integration_steps=integration_steps,
                                     integration_tol=integration_tol, integration_atol=integration_atol)

        loss = MSE(ren_y_test, y_predict_test_torch)
        print(loss)
        nrmse_base = MSE(torch.zeros_like(y_predict_test_torch), y_predict_test_torch)
        nrmse = torch.sqrt(loss/nrmse_base)
        print(nrmse)

    if plot:
        M = 5
        rindex = np.random.randint(0, N, M)

        for i in range(M):

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_test, y_predict_test_torch[rindex[i], 0, :].detach().cpu(), linewidth=1.5, label=r'$x_2(t)$')
            plt.plot(t_test, ren_y_test[rindex[i], 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
            plt.legend(loc='best')
            plt.xlabel('Time [s]')
            plt.ylabel(r'$x_2(t) [rad/s]$')

            plt.subplot(2, 1, 2)
            plt.plot(t_test, u_test[rindex[i], 0, :], linewidth=1.5, label=r'$u(t)$')
            plt.xlabel('Time [s]')
            plt.ylabel(r'$u(t) [rad/s]$')
            plt.legend(loc='best')

            plt.show()



def utest(t):
    n = t.shape[0]
    u = np.ones((1,n))
    step = np.random.rand()
    u[:,:int(n*step)] = (np.random.rand()-0.5)*10*u[:,:int(n*step)]
    u[:,int(n*step):] = (np.random.rand()-0.5)*10*u[:,int(n*step):]
    return u

def utest_sin(t):
    N = 10
    n = t.shape[0]
    As = np.random.rand(10,1)
    ws = np.random.rand(10,1)*t[-1]
    u = np.zeros((1,n))
    for i in range(N):
        u = u + As[i]*np.sin(ws[i]*t)

    return u

evaluate_model(8, utest_sin, experiment_folder='../Data/Experiment1/', plot=True)
