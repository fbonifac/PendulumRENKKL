import numpy as np
import torch
import matplotlib.pyplot as plt
from Pendulum.pendulum import N_PendulumSystem
from KKL.KKL import N_KKLSystem

x0s = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
pendu = N_PendulumSystem(x0s, 1.5, 0.5, 9.81)
us = np.ones((3, 1, 101), dtype=float)
t = np.linspace(0, 4, 101)
sim = pendu.simulate(t, us)

plt.subplot(2, 1, 1)
plt.plot(t, sim[0,0,:])
plt.plot(t, sim[1,0,:])
plt.plot(t, sim[2,0,:])
plt.subplot(2, 1, 2)
plt.plot(t, sim[0,1,:])
plt.plot(t, sim[1,1,:])
plt.plot(t, sim[2,1,:])
plt.show()

z0s = np.eye(5, dtype=float)
us = np.ones((5, 1, 101), dtype=float)
kkl = N_KKLSystem(z0s, np.diag([-1, -2, -3, -4, -5]), np.ones((5,1), dtype=float))

sim2 = kkl.simulate(t, us)

plt.figure()
plt.plot(t, sim2[0,0,:])
plt.plot(t, sim2[1,1,:])
plt.plot(t, sim2[2,2,:])
plt.plot(t, sim2[3,3,:])
plt.plot(t, sim2[4,4,:])
plt.show()
