# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 04:30:02 2020

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
        self.temporal_conv = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(15, 0, 0), bias=False)
        self.last_lay = nn.Conv3d(1,1,(1,3,3), stride=(1, 3, 3), padding=(0, 0, 0), bias=False)
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 2), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(15, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(15, 0, 0), bias=True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,2,2), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_row_create = nn.Conv3d(1,1, (1,2,2), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

    def forward(self, x):
        out = self.last_lay(F.relu(self.temporal_conv(self.space_conv(x))))
        ama_out = self.amacrine_alpha(torch.sigmoid(self.amacrine_kernel(self.amacrine_create(out))))
        cell = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -1, -1), -1)
        gang_out = self.ganglion_kernel(torch.reshape(self.ganglions_create(cell[:,:,:,:-1,:]), (len(cell[:,0,0,0,0]),len(cell[0,:,0,0,0]),len(cell[0,0,:,0,0]),2,2)))
        ganglion_tot = torch.sigmoid(torch.add(gang_out,ama_out))
        out = torch.squeeze(torch.stack((self.ganglion_row_create(ganglion_tot),self.ganglion_col_create(ganglion_tot)), dim=3))
        return out


if __name__ == "__main__":
    net = Retinal_NET().to(device)
    scene, cent_loc = environment(4, 15)
    loss_vals, loss, pred_py, net = optimize_func(scene, cent_loc, net, 1000)
    plt.plot(loss_vals)
    make_output_test_video(pred_py, cent_loc)
    
    scenef, cent_locf = stimuli(20, 3, -1, 1, 2, 4)
    lossf = make_test(net, scenef, cent_locf)

    