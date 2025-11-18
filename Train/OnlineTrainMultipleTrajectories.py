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

def train_N_KKLREN(T_trans, dT, N, T_steps, u_range, maxwaitepoch, learning_rate=5.0e-3, device='cuda'):
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
    model = N_RENSystem(6, 6, 8, 1, N, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    loss = 1.0
    MSE = nn.MSELoss()

    N_trans = int(T_trans / dT)
    N_steps = int(sum(T_steps) / dT)

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
        """
        plt.plot(t_trans, y_trans[np.random.randint(0,N-1),0,:])
        plt.plot(t_trans, ren_y_trans[np.random.randint(0,N-1),0,:].detach().cpu())
        plt.show()"""

    fnfe_values = np.array([])
    bnfe_values = np.array([])
    loss_values = np.array([])
    e = 0
    computing = True
    minlossindex = 0
    minlossmodel = model.state_dict()

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

    while(computing):

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

            loss_values = np.append(loss_values, loss.detach().cpu().numpy())
            fnfe_values = np.append(fnfe_values, fnfe)
            bnfe_values = np.append(bnfe_values, bnfe)

            if np.min(loss_values) != loss_values[minlossindex]:
                minlossindex = e
                minlossmodel = model.state_dict()

            if e > minlossindex + maxwaitepoch:
                computing = False

            e = e + 1

    total_time = time.time() - start
    print("")
    print("")
    print("")
    print(f"Finished Training Phase. \nTotal time required: {total_time} s")
    print(f"Final NFE-F average: {np.mean(fnfe_values)} \t||\t NFE-B average: {np.mean(bnfe_values)}")

    print(f"Values of KKL poles:\n")
    print(kkl.A)
    np.savetxt("../Eval/model_KKL", kkl.A)
    torch.save(minlossmodel, "model.pt")

train_N_KKLREN(30, 0.05, 100, np.array([2, 3]), [-5, 5], 300,1.0e-3)