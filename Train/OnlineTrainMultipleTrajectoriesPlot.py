#Packages:
import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt

#Own code:
from Pendulum.pendulum import N_PendulumSystem
from KKL.KKL import N_KKLSystem
from REN.REN import N_RENSystem

def train_N_KKLREN(T_trans, dT, N, T_steps, u_range, M, T_test, u_test, learning_rate=5.0e-3, device='cuda'):
    torch.set_default_dtype(torch.float32)

    """Initialize system that will be observed:
        -Initial states at: rand([1,1],[-1,-1])
        -Parameters of the system:
            -beta: 1.5
            -l: 0.5
            -g: 9.81
    """
    initial_pendulum_states = np.random.uniform(-1, 1, (N, 2))
    pendulum = N_PendulumSystem(initial_pendulum_states, 1.5, 0.5, 9.81)

    """Initialize KKL system (z transformed system):
        -Initial states at: [0; 0; 0; 0; 0; 0]
        -Parameters of the system:
            -A: -5*diag(random(0,1))
            -B: ones((6, 1))
    """
    A = np.diag(-5 * np.random.rand(6))
    B = np.ones((6, 1))
    initial_KKL_states = np.zeros((N, 6), dtype=np.float32)
    kkl = N_KKLSystem(initial_KKL_states, A, B)

    """Initialize REN (Tau* transformation from (u, y, z) to (x)):
        -Parameters of the system:
            -nx: 6
            -nq: 6
            -nu: 8 (6 from the z_system, 1 from the input u, 1 from the output y)
            -ny: 1 (1 state of the pendulum)
    """
    model = N_RENSystem(6, 6, 8, 1, N, bias=True, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    loss = 1.0
    MSE = nn.MSELoss()

    N_trans = int(T_trans / dT)
    N_steps = int(sum(T_steps) / dT)
    N_test = int(T_test / dT)

    t_trans = np.linspace(0, T_trans, N_trans + 1)
    u_trans = np.zeros((N, 1, N_trans + 1))

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

        ren_y_trans = model.simulate(t_trans_torch, ren_u_trans_torch)
        ren_x_start = model.x.detach()

        plt.plot(t_trans, y_trans[np.random.randint(0,N-1),0,:])
        plt.plot(t_trans, ren_y_trans[np.random.randint(0,N-1),0,:].detach().cpu())
        plt.show()

    fnfe_values = np.array([])
    bnfe_values = np.array([])
    loss_values = np.array([])
    e = 0

    start = time.time()

    u_steps = np.random.uniform(low=u_range[0], high=u_range[1], size=(N, T_steps.shape[0]))
    t_epoch = np.linspace(0, sum(T_steps), N_steps)
    u_epoch = np.ones((N, 1, N_steps))
    for i in range(N):
        for b in range(T_steps.shape[0]):
            u_epoch[i,0,int(sum(T_steps[0:b])/dT):int(sum(T_steps[0:b+1])/dT)] = u_steps[i,b]*u_epoch[i,0,int(sum(T_steps[0:b])/dT):int(sum(T_steps[0:b+1])/dT)]

    x_epoch = pendulum.simulate(t_epoch, u_epoch)
    y_epoch = pendulum.output(x_epoch)
    z_epoch = kkl.simulate(t_epoch, y_epoch)
    u_y_epoch = np.append(u_epoch, y_epoch, axis=1)
    ren_u_epoch = np.append(u_y_epoch, z_epoch, axis=1)

    t_epoch_torch = torch.from_numpy(t_epoch).float()
    t_epoch_torch = t_epoch_torch.to(device)
    ren_u_epoch_torch = torch.from_numpy(ren_u_epoch).float()
    ren_u_epoch_torch = ren_u_epoch_torch.to(device)

    y_predict_epoch = x_epoch[:, 1:, :]
    y_predict_epoch_torch = torch.from_numpy(y_predict_epoch).float()
    y_predict_epoch_torch = y_predict_epoch_torch.to(device)

    while(abs(loss) >  5.0e-3):

        model.nfe = 0
        optimizer.zero_grad()

        ren_y_epoch = model.simulate(t_epoch_torch, ren_u_epoch_torch, ren_x_start)

        loss = MSE(ren_y_epoch, y_predict_epoch_torch)
        print(f"Epoch #: {e + 1}.\t||\t Local Loss: {loss:.6f}")

        fnfe = model.nfe
        model.nfe = 0

        loss.backward()
        optimizer.step()

        model.updateParameters()

        bnfe = model.nfe
        model.nfe = 0

        with torch.no_grad():

            rindex = np.random.randint(0, N-1)
            print(f"Plot of the {rindex}th trajectory:")

            plt.subplot(2, 1, 1)
            plt.plot(t_epoch, y_predict_epoch[rindex, 0, :], linewidth=1.5, label=r'$x_2(t)$')
            plt.plot(t_epoch, ren_y_epoch[rindex, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
            plt.xlabel(r'$time [s]$')
            plt.ylabel(r'$x_2(t) [rad/s]$')
            plt.legend(loc='best')

            plt.subplot(2, 1, 2)
            plt.plot(t_epoch, u_epoch[rindex, 0, :], linewidth=1.5, label=r'$u(t)$')
            plt.xlabel(r'$time [s]$')
            plt.ylabel(r'$u(t) [rad]$')
            plt.legend(loc='best')

            plt.show()

            if (abs(loss) < 1.0e-2):
                print(f"The loss has reached a value smaller than the threshold (1.0e-2)")

            e = e + 1
            loss_values = np.append(loss_values, loss.detach().cpu().numpy())
            fnfe_values = np.append(fnfe_values, fnfe)
            bnfe_values = np.append(bnfe_values, bnfe)

    total_time = time.time() - start
    print("")
    print("")
    print("")
    print(f"Finished Training Phase. \nTotal time required: {total_time} s")
    print(f"Final NFE-F average: {np.mean(fnfe_values)} \t||\t NFE-B average: {np.mean(bnfe_values)}")

    torch.save(model.state_dict(), "model.pt")

    with torch.no_grad():
        t_test = np.linspace(0, T_test, N_test)
        u_test1_sizes = np.random.uniform(low=u_range[0], high=u_range[1], size=(M,))
        u_test1 = np.ones((M, 1, N_test), dtype=float)
        for i in range(M):
            u_test1[i,:,:] = u_test1_sizes[i]*u_test1[i,:,:]

        x_pendulum = pendulum.x[0:M,:]
        z_kkl = kkl.z[0:M,:]
        x_ren = ren_x_start[0:M,:]

        pendulum.x = x_pendulum
        kkl.z = z_kkl
        model.x = x_ren

        x_test1 = pendulum.simulate(t_test, u_test1)
        y_test1 = pendulum.output(x_test1)
        z_test1 = kkl.simulate(t_test, y_test1)
        u_y_test1 = np.append(u_test1, y_test1, axis=1)
        ren_u_test1 = np.append(u_y_test1, z_test1, axis=1)

        t_test_torch = torch.from_numpy(t_test).float()
        t_test_torch = t_test_torch.to(device)
        ren_u_test1_torch = torch.from_numpy(ren_u_test1).float()
        ren_u_test1_torch = ren_u_test1_torch.to(device)

        y_predict_test1 = x_test1[:, 1:, :]
        y_predict_test1_torch = torch.from_numpy(y_predict_test1).float()
        y_predict_test1_torch = y_predict_test1_torch.to(device)

        ren_y_test1 = model.simulate(t_test_torch, ren_u_test1_torch)

        test_loss1 = MSE(ren_y_test1, y_predict_test1_torch)
        print(f"\nLoss_testing: {test_loss1}")
        
        rindex1 = np.random.randint(0, M-1)
        rindex2 = np.random.randint(0, M-1)
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_test, y_predict_test1[rindex1, 0, :], linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_test, ren_y_test1[rindex1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(t_test, u_test1[rindex1, 0, :], linewidth=1.5, label=r'$u(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$u(t) [rad]$')
        plt.legend(loc='best')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_test, y_predict_test1[rindex2, 0, :], linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_test, ren_y_test1[rindex1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(t_test, u_test1[rindex2, 0, :], linewidth=1.5, label=r'$u(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$u(t) [rad]$')
        plt.legend(loc='best')

        plt.show()

        print(f"Test trajectories with constant input resulted in the following plots:")

        u_test2 = np.zeros_like(u_test1)
        for i in range(M):
            u_test2[i,0,:] = test_input(t_test)

        pendulum.x = x_pendulum
        kkl.z = z_kkl
        model.x = x_ren

        x_test2 = pendulum.simulate(t_test, u_test2)
        y_test2 = pendulum.output(x_test2)
        z_test2 = kkl.simulate(t_test, y_test2)
        u_y_test2 = np.append(u_test2, y_test2, axis=1)
        ren_u_test2 = np.append(u_y_test2, z_test2, axis=1)

        ren_u_test2_torch = torch.from_numpy(ren_u_test2).float()
        ren_u_test2_torch = ren_u_test2_torch.to(device)

        y_predict_test2 = x_test2[:, 1:, :]
        y_predict_test2_torch = torch.from_numpy(y_predict_test2).float()
        y_predict_test2_torch = y_predict_test2_torch.to(device)

        ren_y_test2 = model.simulate(t_test_torch, ren_u_test2_torch)

        test_loss2 = MSE(ren_y_test2, y_predict_test2_torch)
        print(f"\nLoss_testing: {test_loss2}")

        rindex1 = np.random.randint(0, M - 1)
        rindex2 = np.random.randint(0, M - 1)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_test, y_predict_test2[rindex1, 0, :], linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_test, ren_y_test2[rindex1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(t_test, u_test2[rindex1, 0, :], linewidth=1.5, label=r'$u(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$u(t) [rad]$')
        plt.legend(loc='best')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_test, y_predict_test2[rindex2, 0, :], linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_test, ren_y_test2[rindex1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(t_test, u_test2[rindex2, 0, :], linewidth=1.5, label=r'$u(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$u(t) [rad]$')
        plt.legend(loc='best')

        plt.show()

        print(f"Test trajectories with sinusiodal input resulted in the following plots:")

        
def test_input(t):
    N = 5
    w_max = 10
    A = np.random.rand(N)
    w = np.random.rand(N)*w_max+1
    u = np.zeros_like(t, dtype=float)
    for i in range(N):
        u += A[i]*np.sin(w[i]*t)
    return u

train_N_KKLREN(10, 0.05, 100, np.array([2, 3]), [-5, 5], 10, 10, test_input)