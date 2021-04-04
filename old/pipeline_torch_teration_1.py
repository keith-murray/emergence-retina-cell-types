# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:32:21 2020

@author: Keith
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from support_old import environment,optimize_func,make_output_test_video,stimuli,make_test

device = torch.device("cuda:0")


class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        self.space_conv = nn.Conv3d(1,1,(1,9,9), stride=(1, 4, 4), padding=(0, 0, 0), bias=False)
        self.temporal_conv = nn.Conv3d(1,1,(11,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.last_lay = nn.Conv3d(1,1,(1,3,3), stride=(1, 3, 3), padding=(0, 0, 0), bias=False)
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = nn.Conv3d(1,1,(11,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = nn.Conv3d(1,1,(11,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,2,2), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_row_create = nn.Conv3d(1,1, (1,2,2), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.p1d = (0, 0, 0, 0, 10, 0)

    def forward(self, x):
        out = self.last_lay(F.relu(self.temporal_conv(F.pad(self.space_conv(x), self.p1d))))
        ama_out = self.amacrine_alpha(torch.sigmoid(self.amacrine_kernel(F.pad(self.amacrine_create(out), self.p1d))))
        cell = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -1, -1), -1)
        nxt_step = torch.reshape(self.ganglions_create(cell[:,:,:,:-1,:]), (len(cell[:,0,0,0,0]),len(cell[0,:,0,0,0]),len(cell[0,0,:,0,0]),2,2))
        next_step = F.pad(nxt_step,self.p1d)
        gang_out = self.ganglion_kernel(next_step)         
        ganglion_tot = torch.sigmoid(torch.sub(gang_out,torch.abs(ama_out)))
        cell_stack = torch.stack((self.ganglion_row_create(ganglion_tot),self.ganglion_col_create(ganglion_tot)), dim=3)
        fin = torch.squeeze(cell_stack)
        return fin


if __name__ == "__main__":
    net = Retinal_NET().to(device)
    scene, cent_loc = environment(10, 30)
    loss_vals, loss, pred_py, net = optimize_func(scene, cent_loc, net, 20000)
    plt.plot(loss_vals)
    make_output_test_video(pred_py, cent_loc)
    
    scenef, cent_locf = stimuli(20, 3, -1, 1, 2, 4)
    lossf = make_test(net, scenef, cent_locf)
    
    
    # net.temporal_conv.weight = torch.nn.Parameter(torch.cat((net.temporal_conv.weight[:,:,:6,:,:],torch.zeros(1,1,5,1,1,dtype=torch.float,device="cuda")),dim=2))
    # net.amacrine_kernel.weight = torch.nn.Parameter(torch.cat((net.amacrine_kernel.weight[:,:,:6,:,:],torch.zeros(1,1,5,1,1,dtype=torch.float,device="cuda")),dim=2))
    # net.ganglion_kernel.weight = torch.nn.Parameter(torch.cat((net.ganglion_kernel.weight[:,:,:6,:,:],torch.zeros(1,1,5,1,1,dtype=torch.float,device="cuda")),dim=2))
    
    
    