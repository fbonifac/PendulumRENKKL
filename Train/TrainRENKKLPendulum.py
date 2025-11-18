#Packages:
import torch
from torch import nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

#Own code:
from Pendulum.pendulum import N_PendulumSystem
from KKL.KKL import N_KKLSystem
from REN.REN import N_RENSystem, N_RENSystem_TORCHODE

def train_N_KKLREN(nx, nq, nkkl, number_of_experiments, batch_size, T_trans, dT,  T_steps, u_range, maxepoch, maxtrainingtime, minNRMSE,
                   ESepochs, learning_rate=5.0e-3, device='cuda', activation_function='relu', default_type=torch.float32, bias=False, plot=False, improve_model=None,
                   kkl_poles=-5, destination_folder='../Data/', integration_method='dopri5', integration_steps=20, integration_tol=1.0e-4,
                   integration_atol=1.0e-6):

    N = number_of_experiments
    torch.set_default_dtype(default_type)

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
        -Initial states at: zeros((nkkl, nusys))
        -Parameters of the system:
            -A: -kklpoles*diag(random(0, 1)) or predefined
            -B: ones((nkkl, nusys))
    """
    #Load KKL poles:
    if np.isscalar(kkl_poles):
        A = np.diag(kkl_poles * np.random.rand(nkkl))
    else:
        A = np.diag(kkl_poles)
    B = np.ones((nkkl, nysys))
    initial_KKL_states = np.zeros((N, nkkl), dtype=np.float32)
    kkl = N_KKLSystem(initial_KKL_states, A, B)

    """Initialize REN (Tau* transformation from (u, y, z) to (x)):
        -Parameters of the system:
            -nx: nx
            -nq: nq
            -nu: nusys + nysys + nkkl (6 from the z_system, 1 from the input u, 1 from the output y)
            -ny: nxsys - nusys (1 state of the pendulum)
    """
    ny = nxsys - nysys
    nu = nusys + nysys + nkkl
    model = N_RENSystem_TORCHODE(nx, nq, nu, ny, bias=bias, sigma=activation_function, device=device)
    #Load existing model
    if improve_model is not None:
        weights = torch.load(improve_model, weights_only=True)
        if not bias and 'sys.bx' in weights.keys():
            del weights['sys.bx']
            del weights['sys.bv']
            del weights['sys.by']
        model.load_state_dict(weights)
        model.updateParameters()

    #Create optimizer: ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    #Declare necessary variables:
    MSE = nn.MSELoss()
    fnfe_values = np.array([])
    bnfe_values = np.array([])
    loss_values = np.array([])
    nrmse_values = np.array([])
    ESepoch_values = np.array([])
    time_values = np.array([])
    e = 0
    training = True
    minlossindex = 0
    minlossmodel = model.state_dict()
    N_batches = int(N / batch_size)

    #Number of steps with dT in T_trans and T_steps
    N_trans = int(T_trans / dT) + 1
    N_steps = int(sum(T_steps) / dT) + 1

    #---------------------------------------TRANSIENT PHASE------------------------------------------------
    #The whole system (Pendulum+KKL+REN) is simulated for T_trans to line up

    #Create time vector and input vector for transient phase
    t_trans = np.linspace(0, T_trans, N_trans)
    u_trans = np.zeros((N, nusys, N_trans))

    #Simulate the pendulum, calculate output, simulate the KKL system, create the input for the REN, simulate the REN and save the last state
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

        y_predict_trans = x_trans[:, nysys:, :]
        y_predict_trans_torch = torch.from_numpy(y_predict_trans).float()
        y_predict_trans_torch = y_predict_trans_torch.to(device)

        ren_x_init = torch.zeros(N, nx, device=device)
        ren_y_trans = model.simulate(t_trans_torch, ren_u_trans_torch, ren_x_init, integration_method=integration_method, integration_steps=integration_steps, integration_tol=1.0e-4, integration_atol=1.0e-6)
        ren_x_start = model.x.detach()
        #The last state is saved to initialize the REN every epoch with the same state

    if plot:
        plt.plot(t_trans, y_predict_trans_torch[-1, 0, :].detach().cpu(), linewidth=1.5, label=r'$x_2(t)$')
        plt.plot(t_trans, ren_y_trans[-1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
        plt.xlabel(r'$time [s]$')
        plt.ylabel(r'$x_2(t) [rad/s]$')
        plt.legend(loc='best')
        plt.title('Transient trajectories of the last experiment')
        plt.show()


    #---------------------------------------TRAINING PHASE------------------------------------------------

    #Create time vector and input vector for training
    t_epoch = np.linspace(0, sum(T_steps), N_steps)
    u_steps = np.random.uniform(low=u_range[0], high=u_range[1], size=(N, T_steps.shape[0]))
    u_epoch = np.ones((N, nusys, N_steps))
    for i in range(N):
        for b in range(T_steps.shape[0]):
            u_epoch[i,:,int(sum(T_steps[0:b])/dT):int(sum(T_steps[0:b+1])/dT)] = u_steps[i,b]*u_epoch[i,:,int(sum(T_steps[0:b])/dT):int(sum(T_steps[0:b+1])/dT)]
        u_epoch[i,:,-1] = u_steps[i,-1]*u_epoch[i,:,-1]

    #Simulate the pendulum, calculate output, simulate the KKL system, create the input for the REN, simulate the REN and save the last state
    x_epoch = pendulum.simulate(t_epoch, u_epoch)
    y_epoch = pendulum.output(x_epoch)
    z_epoch = kkl.simulate(t_epoch, y_epoch)
    u_y_epoch = np.append(u_epoch, y_epoch, axis=1)
    ren_u_epoch = np.append(u_y_epoch, z_epoch, axis=1)

    t_epoch_torch = torch.from_numpy(t_epoch).float()
    t_epoch_torch = t_epoch_torch.to(device)
    ren_u_epoch_torch = torch.from_numpy(ren_u_epoch).float()
    ren_u_epoch_torch = ren_u_epoch_torch.to(device)

    y_predict_epoch = x_epoch[:, nysys:, :]
    y_predict_epoch_torch = torch.from_numpy(y_predict_epoch).float()
    y_predict_epoch_torch = y_predict_epoch_torch.to(device)

    #Calculate base for Normalized Root Mean Square Error
    base_y = torch.zeros_like(y_predict_epoch_torch, device=device)
    NRMSE_base = MSE(base_y, y_predict_epoch_torch).detach().cpu().numpy()

    #Start timer
    start_time = time.time()
    prev_time = start_time

    #Train while there was no improvement for ESepochs
    while(training):

        loss_temp = 0.0
        fnfe = 0
        bnfe = 0

        for i_batch in range(N_batches):
            #Zero out the NFE and the gradients
            optimizer.zero_grad()

            #Create input, initial conditions and goal output for current batch
            ren_u_batch = ren_u_epoch_torch[i_batch*batch_size:(i_batch+1)*batch_size, :, :]
            ren_init_batch = ren_x_start[i_batch*batch_size:(i_batch+1)*batch_size,:]
            y_predict_batch = y_predict_epoch_torch[i_batch*batch_size:(i_batch+1)*batch_size, :, :]

            #Simulate model for the N trajectories with ren_u_epoch_torch inputs from ren_x_start using a given integration method
            ren_y_batch = model.simulate(t_epoch_torch, ren_u_batch, ren_init_batch, integration_method=integration_method, integration_steps=integration_steps, integration_tol=1.0e-4, integration_atol=1.0e-6)

            #Calculate MSE for batch
            batch_loss = MSE(ren_y_batch, y_predict_batch)

            #Save FNFE and zero out NFE
            fnfe += model.nfe
            model.nfe = 0

            #Calculate gradients, do an optimizer step and update the model
            batch_loss.backward()
            optimizer.step()
            model.updateParameters()

            #Save BNFE
            bnfe += model.nfe
            model.nfe = 0

            #Save loss value
            loss_temp += batch_loss.detach().cpu().numpy()

        with torch.no_grad():

            # Save loss and nrmse
            loss = loss_temp / N_batches
            nrmse = np.sqrt(loss / NRMSE_base)

            loss_values = np.append(loss_values, loss)
            nrmse_values = np.append(nrmse_values, nrmse)

            #Update best model
            if np.min(loss_values) != loss_values[minlossindex]:
                minlossindex = e
                minlossmodel = model.state_dict()
            else:
                learning_rate = learning_rate / 2
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate

            if plot:
                plt.subplot(2, 1, 1)
                plt.plot(t_epoch, y_predict_epoch[-1, 0, :], linewidth=1.5, label=r'$x_2(t)$')
                plt.plot(t_epoch, ren_y_batch[-1, 0, :].detach().cpu(), linewidth=1.5, label=r'$\hat{x}_2(t)$')
                plt.xlabel(r'$time [s]$')
                plt.ylabel(r'$x_2(t) [rad/s]$')
                plt.title('Output and input trajectories of the last experiment')
                plt.legend(loc='best')

                plt.subplot(2, 1, 2)
                plt.plot(t_epoch, u_epoch[-1, 0, :], linewidth=1.5, label=r'$u(t)$')
                plt.xlabel(r'$time [s]$')
                plt.ylabel(r'$u(t) [rad]$')
                plt.legend(loc='best')

                plt.show()

            epoch_time = time.time() - prev_time
            print(f"Epoch #: {e + 1}.\t||\t Local Loss: {loss:.6f} \t||\t NRMSE: {nrmse:.6f} \t||\t Time needed for this epoch: {epoch_time:.2f}"
                  f"seconds\t||\t Average time per epoch: {(time.time() - start_time)/(e + 1):.2f} seconds")

            #Stop training
            if (e > maxepoch) or (time.time()-start_time > maxtrainingtime) or (nrmse < minNRMSE) or (e - 1 > minlossindex + ESepochs):
                if (e > maxepoch):
                    print(f"Training reached maximum epoch.")
                elif (time.time()-start_time > maxtrainingtime):
                    print(f"Training reached maximum time.")
                elif (nrmse < minNRMSE):
                    print(f"Target NRMSE reached.")
                elif (e - 1 > minlossindex + ESepochs):
                    print(f"Early stopping activated.")
                training = False

            #Save other parameters
            fnfe_values = np.append(fnfe_values, fnfe)
            bnfe_values = np.append(bnfe_values, bnfe)
            ESepoch_values = np.append(ESepoch_values, e-minlossindex)
            time_values = np.append(time_values, epoch_time)

            prev_time = time.time()
            e = e + 1

    print("")
    print("")
    print("")
    print(f"Finished Training Phase. \nTotal time required: {time.time() - start_time} s")
    print(f"Final NFE-F average: {np.mean(fnfe_values)} \t||\t NFE-B average: {np.mean(bnfe_values)}")
    print(f"Best model had loss: {loss_values[minlossindex]:.6f} and NRMSE: {nrmse_values[minlossindex]:.6f}" )

    # Create path to save data
    exp_num = 1
    path = destination_folder + "Experiment" + str(exp_num)
    while os.path.exists(path):
        exp_num += 1
        path = destination_folder + "Experiment" + str(exp_num)
    os.makedirs(path)

    #Save model and optimizer
    optmodel = {'model':minlossmodel, 'optimizer':optimizer.state_dict()}
    torch.save(optmodel, path + "/model_and_optimizer.pt")

    #Save epoch data
    data_epoch = pd.DataFrame({'Loss' : loss_values, 'NRMSE' : nrmse_values, 'FNFE' : fnfe_values, 'BNFE' : bnfe_values, 'Required_time' : time_values})
    data_epoch.to_csv(path + "/epoch_data.csv", index=True)

    #Save hyperparameters of the experiment
    hyperparameters = {"nx":nx, "nq":nq, "nu":nu, "ny":ny, "nkkl":nkkl, "KKL_poles_multiplier":kkl_poles, "KKL_poles":np.diag(A), "transient_time":T_trans,
                       "experiment_times":T_steps, "sampling_time":dT, "range_of_inputs":u_range, "bias":bias, "activation_function":activation_function, "device":device, "learning_rate":learning_rate, "number_of_experiments":N,
                       "batch_size":batch_size, "max_epochs":maxepoch, "max_training_time":maxtrainingtime, "NRMSE_stopping_threshold":minNRMSE, "early_stopping_epochs":ESepochs,
                       "variable_type":default_type, "integration_method":integration_method, "fixed_number_of_steps":integration_steps, "integration_tolerance":integration_tol,
                       "integration_atol":integration_tol, "integration_absolute_tolerance":integration_atol}
    np.save(path + "/hyperparameters.npy", hyperparameters)



    print(f"Model and experiment data is saved in {path}")


train_N_KKLREN(6, 6, 6, 250, 50, 20, 0.05, np.array([2, 5]), np.array([-5, 5]),
               1200, 120000, 0.01, 20, learning_rate=2.0e-3, kkl_poles=-2, plot = True, integration_method='euler',
               integration_steps=20, integration_tol=1.0e-4, integration_atol=1.0e-6)