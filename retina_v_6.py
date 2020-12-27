# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:31 2020

@author: Keith
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from support_discrim import environment, optimize_func, create_summary_apriori, createTest

device = torch.device("cuda:0")

class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        self.space_conv = nn.Conv3d(1,1,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=False)
        self.temporal_conv = nn.Conv3d(1,1,(15,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.last_lay = nn.Conv3d(1,1,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=False)
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = nn.Conv3d(1,1,(15,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = nn.Conv3d(1,1,(15,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.p1d = (0, 0, 0, 0, 14, 0)

    def forward(self, x):
        first = F.relu(self.temporal_conv(F.pad(self.space_conv(x), self.p1d)))
        out = self.last_lay(first) # Make sure that it is positive
        cell_ama = torch.unsqueeze(torch.flatten(out,start_dim=3), -1)
        ama_pre_pad = self.amacrine_create(cell_ama[:,:,:,:-2,:])
        ama_out = self.amacrine_alpha(torch.sigmoid(self.amacrine_kernel(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -25, -1), -1)
        gang_pre_pad = self.ganglions_create(cell_gang[:,:,:,:-2,:])
        gang_out = self.ganglion_kernel(F.pad(gang_pre_pad,self.p1d))
        ganglion_tot = torch.sigmoid(torch.sub(gang_out,torch.abs(ama_out)))
        cols = self.ganglion_col_create(ganglion_tot)
        fin = torch.squeeze(cols)[:,-1]
        return fin
    

if __name__ == "__main__":
    net = Retinal_NET().to(device)
    envi, res = environment(10, 250,30,120,1,0,1)
    # torch.autograd.set_detect_anomaly(True)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 600)
    plt.plot(loss_vals)
    
    for i in range(100):
        del envi, res
        envi, res = environment(10, 250,30,120,1,0,1)
        loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 1000)
        if i%7 == 0:
            plt.plot(loss_vals)

    create_summary_apriori(net)
    testRes = createTest(net, 250,30,120,1,0,1)
    print(testRes)
    
    