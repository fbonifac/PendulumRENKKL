#Packages:
import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt

#Own code:
from Pendulum.pendulum import PendulumSystem
from KKL.KKL import KKLSystem
from REN.REN import RENSystem

def trainKKLREN():
    