# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:38:16 2020

@author: Keith
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

device = torch.device("cpu")

x_t_1 = 3.8628e-01*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/1.3025e-01)**2)
x_t_2 = 1.1488e+00*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/1.0167e+00)**2)
y_t_1 = 1.0472e+00*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/1.1485e-03)**2)
y_t_2 = -4.6171e-02*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/9.5341e-01)**2)
x_t = x_t_1 - x_t_2
y_t = y_t_1 - y_t_2
base = torch.mul(torch.unsqueeze(x_t,-1),torch.unsqueeze(y_t,0))

sns.heatmap(base.detach().numpy())