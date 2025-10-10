from REN.REN import RENSystem, RENL2System
import numpy as np
import torch
import matplotlib.pyplot as plt




model = RENSystem(4, 3, 1, 2)
u = torch.ones(1, 10001, device = "cuda")
t = torch.tensor(np.linspace(0, 10, 10001), device = "cuda")
y = model.simulate(t, u)
plt.plot(t.detach().cpu(), y[0, :].detach().cpu())
plt.plot(t.detach().cpu(), y[1, :].detach().cpu())
plt.show()