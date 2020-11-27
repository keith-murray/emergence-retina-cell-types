# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 03:54:17 2020

@author: Keith
"""
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from support_old import mse
import matplotlib.pyplot as plt
from pipeline_torch_teration_1 import Retinal_NET

ret = Retinal_NET()
ret.load_state_dict(torch.load('Q:\\Documents\\TDS SuperUROP\\run_3_19_am_10_30\\net_obj.pt'))

fig, axs = plt.subplots(2, 2)
axs[1,0].plot(torch.squeeze(ret.temporal_conv.weight).detach().cpu().numpy())
axs[1,1] = sns.heatmap(torch.squeeze(ret.space_conv.weight).detach().cpu().numpy())
axs[0,0].plot(torch.squeeze(ret.amacrine_kernel.weight).detach().cpu().numpy())
axs[0,1].plot(torch.squeeze(ret.ganglion_kernel.weight).detach().cpu().numpy())
axs[1,0].set_title('Bipolar Time Kernel')
axs[1,1].set_title('Bipolar Space Kernel')
axs[0,0].set_title('Amacrine Time Kernel')
axs[0,1].set_title('Ganglion Time Kernel')
axs[0,0].xaxis.set_ticklabels([])
axs[0,1].xaxis.set_ticklabels([])