import numpy as np
from Train.TrainRENKKLPendulum import train_N_KKLREN

train_N_KKLREN(6, 6, 6, 250, 50, 20, 0.05, np.array([2, 5]), np.array([-5, 5]),
               1200, 120000, 0.01, 20, learning_rate=2.0e-3, kkl_poles=-2, plot = True, integration_method='euler',
               integration_steps=20, integration_tol=1.0e-4, integration_atol=1.0e-6)