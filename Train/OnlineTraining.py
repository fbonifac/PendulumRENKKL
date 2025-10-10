#Packages:
import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt

#Own code:
from Pendulum.pendulum import PendulumSystem
from KKL.KKL import KKLSystem
from REN.REN import RENSystem, RENL2System


def trainKKLREN(T_trans, T_step, T_test, dT, epochs, step_number, u_range, u_test, learning_rate=3.0e-3, device='cpu'):

    torch.set_default_dtype(torch.float32)

    """Initialize system that will be observed:
        -Initial state at: [1; 1]
        -Parameters of the system:
            -beta: 1.5
            -l: 0.5
            -g: 9.81
    """
    pendulum = PendulumSystem(np.array([1, 1]), 1.5, 0.5, 9.81)

    """Initialize KKL system (z transformed system):
        -Initial state at: [0; 0; 0; 0; 0]
        -Parameters of the system:
            -A: 1000*diag(random(0,1))
            -B: ones((6, 1))
    """
    A = np.diag(-5*np.random.rand(6))
    B = np.ones((6,1))
    kkl = KKLSystem(np.zeros((6,)), A, B)


    """Initialize REN (Tau* transformation from (u, y, z) to (x)):
        -Initial state at: [0; 0; 0; 0; 0; 0]
        -Parameters of the system:
            -nx: 6
            -nq: 6
            -nu: 8 (6 from the z_system, 1 from the input u, 1 from the output y)
            -ny: 2 (2 state of the pendulum)
    """
    model = RENSystem(6, 6, 8, 2, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    loss = 1.0
    MSE = nn.MSELoss()

    N_trans = int(T_trans/dT)
    N_step = int(T_step/dT)
    N_test = int(T_test/dT)

    t_trans = np.linspace(0, T_trans, N_trans+1)
    u_trans = np.zeros((1,N_trans+1))

    #t = t_trans
    #u = u_trans

    with torch.no_grad():
        x_trans = pendulum.simulate(t_trans, u_trans)
        y_trans = pendulum.output(x_trans)
        z_trans = kkl.simulate(t_trans, y_trans)
        u_y_trans = np.append(u_trans, y_trans, axis=0)
        ren_u_trans = np.append(u_y_trans, z_trans, axis=0)

        ren_u_trans_torch = torch.from_numpy(ren_u_trans).float()
        ren_u_trans_torch = ren_u_trans_torch.to(device)
        t_trans_torch = torch.from_numpy(t_trans).float()
        t_trans_torch = t_trans_torch.to(device)

        x_hat_trans = model.simulate(t_trans_torch, ren_u_trans_torch)
        ren_x_start = model.x.detach()

    nfe_values = np.zeros((epochs, 2))
    loss_values = np.zeros((epochs,))

    start = time.time()

    u_steps = np.random.uniform(low=u_range[0], high=u_range[1], size=(step_number,))
    t_epoch = np.linspace(0, T_step*step_number, N_step*step_number)
    u_epoch = np.ones((N_step*step_number,1))
    for b in range(step_number):
        u_epoch[b*N_step:(b+1)*N_step] = u_steps[b]*u_epoch[b*N_step:(b+1)*N_step]
    u_epoch = np.transpose(u_epoch)

    x_epoch = pendulum.simulate(t_epoch, u_epoch)
    y_epoch = pendulum.output(x_epoch)
    z_epoch = kkl.simulate(t_epoch, y_epoch)
    u_y_epoch = np.append(u_epoch, y_epoch, axis=0)
    ren_u_epoch = np.append(u_y_epoch, z_epoch, axis=0)

    t_epoch_torch = torch.from_numpy(t_epoch).float()
    t_epoch_torch = t_epoch_torch.to(device)
    ren_u_epoch_torch = torch.from_numpy(ren_u_epoch).float()
    ren_u_epoch_torch = ren_u_epoch_torch.to(device)
    x_epoch_torch = torch.from_numpy(x_epoch).float()
    x_epoch_torch = x_epoch_torch.to(device)

    x_hat_epoch = model.simulate(t_epoch_torch, ren_u_epoch_torch, ren_x_start)

    """x1_min = min(min(x_epoch[0,:]), min(x_hat_epoch[0,:]))
    x2_min = min(min(x_epoch[1,:]), min(x_hat_epoch[1,:]))
    x1_max = max(max(x_epoch[0,:]), max(x_hat_epoch[0,:]))
    x2_max = max(max(x_epoch[1,:]), max(x_hat_epoch[1,:]))"""

    for e in range(epochs):

        model.nfe = 0
        optimizer.zero_grad()

        x_hat_epoch = model.simulate(t_epoch_torch, ren_u_epoch_torch, ren_x_start)

        loss = MSE(x_hat_epoch, x_epoch_torch)
        print(f"Epoch #: {e+1}.\t||\t Local Loss: {loss:.4f}")

        nfe_values[e, 0] = model.nfe
        model.nfe = 0

        loss.backward()
        optimizer.step()

        model.updateParameters()

        nfe_values[e, 1] = model.nfe
        model.nfe = 0

        with torch.no_grad():
            loss_values[e] = loss

            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(t_epoch, x_epoch[0, :], linewidth=1.5, label=r'$x_1(t)$')
            plt.plot(t_epoch, x_hat_epoch[0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_1(t)$')
            #plt.ylim((x1_min, x1_max))
            plt.xlabel(r'$time [s]$')
            plt.ylabel(r'$x_1(t) [rad]$')
            plt.legend(loc='best')

            plt.subplot(3, 1, 2)
            plt.plot(t_epoch, x_epoch[1, :], linewidth=1.5, label=r'$x_2(t)$')
            plt.plot(t_epoch, x_hat_epoch[1, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
            #plt.ylim((x2_min, x2_max))
            plt.xlabel(r'$time [s]$')
            plt.ylabel(r'$x_2(t) [rad/s]$')
            plt.legend(loc='best')

            plt.subplot(3, 1, 3)
            plt.plot(t_epoch, np.transpose(u_epoch), linewidth=1.5, label=r'$u(t)$')
            plt.xlabel(r'$time [s]$')
            plt.ylabel(r'$u(t) [rad]$')
            plt.legend(loc='best')

            plt.show()

        if (abs(loss) < 1.0e-5):
            print(f"The loss has reached a value smaller than the threshold (1.0e-5)")

    total_time = time.time()-start
    print("")
    print("")
    print("")
    print(f"Finished Training Phase. \nTotal time required: {total_time} s")

    with torch.no_grad():
        t_test = np.linspace(0, T_test, N_test+1)
        u_test = np.expand_dims(u_test(t_test), axis=0)

        x_test = pendulum.simulate(t_test, u_test)
        y_test = pendulum.output(x_test)
        z_test = kkl.simulate(t_test, y_test)
        u_y_test = np.append(u_test, y_test, axis=0)
        ren_u_test = np.append(u_y_test, z_test, axis=0)

        t_test_torch = torch.from_numpy(t_test).float()
        t_test_torch = t_test_torch.to(device)
        ren_u_test_torch = torch.from_numpy(ren_u_test).float()
        ren_u_test_torch = ren_u_test_torch.to(device)
        x_test_torch = torch.from_numpy(x_test).float()
        x_test_torch = x_test_torch.to(device)

        x_hat_test = model.simulate(t_test_torch, ren_u_test_torch)

        test_loss = MSE(x_hat_test, x_test_torch)
        test_nfe = model.nfe
        print(f"\nLoss_testing: {test_loss}")
        print(f"Final NFE-F average: {np.mean(nfe_values[:, 0])} \t||\t NFE-B average: {np.mean(nfe_values[:, 1])}")

    plt.figure()
    plt.plot(loss_values)
    plt.ylabel(r'Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_test, x_test[0, :], linewidth=1.5, label=r'$x_1(t)$')
    plt.plot(t_test, x_hat_test[0, :].cpu().detach(), linewidth=1.5, label=r'$\hat{x}_1(t)$')
    plt.xlabel(r'$time [s]$')
    plt.ylabel(r'$x_1(t) [rad]$')
    plt.legend(loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(t_test, x_test[1, :], linewidth=1.5, label=r'$x_2(t)$')
    plt.plot(t_test, x_hat_test[1, :].cpu().detach(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
    plt.xlabel(r'$time [s]$')
    plt.ylabel(r'$x_2(t) [rad/s]$')
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(t_test, np.transpose(u_test), linewidth=1.5, label=r'$u(t)$')
    plt.xlabel(r'$time [s]$')
    plt.ylabel(r'$u(t) [rad]$')
    plt.legend(loc='best')

    plt.show()


def test_input(t):
    u = 2.0*np.sin(1.0*t)+1.0*np.sin(2.0*t)+1.0*np.sin(4.0*t)+1.0*np.sin(8.0*t)
    return u

trainKKLREN(10, 2, 5, 0.01, 1000, 2, [-7, 7], test_input)