# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:31 2020

@author: Keith
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
from support_binary import Dataset, evalFunc, createDataTest, reConfigure

device = torch.device("cuda:0")

class Retinal_NET_binary(nn.Module):
    def __init__(self):
        super(Retinal_NET_binary, self).__init__()
        self.bipolar_space = nn.Conv3d(1,5,(1,10,10), stride=(1, 5, 5), padding=(0, 0, 0), bias=True, groups=1) 
        self.bipolar_temporal = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=5)
        self.amacrine_space = nn.Conv3d(5,5,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=True, groups=5)
        self.ganglions_space = nn.Conv3d(5,5,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=True, groups=5)
        self.amacrine_pull = nn.Conv3d(5,5,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False, groups=5) # Visualize the spacial kernel
        self.ganglions_pull =  nn.Conv3d(5,5,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False, groups=5) # Integrate with last_lay
        self.amacrine_temporal = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=5)
        self.amacrine_alpha = nn.Conv3d(5,5,(1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False, groups=5)
        self.ganglion_temporal = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=5)
        self.ganglion_left_create = nn.Conv3d(5,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.ganglion_right_create = nn.Conv3d(5,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.drop_layer = nn.Dropout(p=0.30)
        
        self.p1d = (0, 0, 0, 0, 50, 0)

    def forward(self, x):
        bipolar_out = F.relu(self.drop_layer(self.bipolar_temporal(F.pad(self.drop_layer(self.bipolar_space(x)), self.p1d))))
        ama_space = self.drop_layer(self.amacrine_space(bipolar_out))
        gang_space = self.drop_layer(self.ganglions_space(bipolar_out))
        cell_ama = torch.unsqueeze(torch.flatten(ama_space,start_dim=3), -1)
        ama_pre_pad = self.drop_layer(self.amacrine_pull(cell_ama[:,:,:,:-2,:]))
        ama_out = self.drop_layer(self.amacrine_alpha(F.relu(self.drop_layer(self.amacrine_temporal(F.pad(ama_pre_pad, self.p1d))))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(gang_space,start_dim=3), -25, -1), -1)
        gang_pre_pad = self.drop_layer(self.ganglions_pull(cell_gang[:,:,:,:-2,:]))
        gang_out = self.drop_layer(self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d)))
        ganglion_tot = self.drop_layer(F.relu(torch.sub(gang_out,torch.abs(ama_out))))
        left = self.ganglion_left_create(ganglion_tot)
        right = self.ganglion_right_create(ganglion_tot)
        fin_left = torch.squeeze(left)[:,-1]
        fin_right = torch.squeeze(right)[:,-1]
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin
    
    def outputCell(self, x):
        bipolar_out = F.relu(self.drop_layer(self.bipolar_temporal(F.pad(self.drop_layer(self.bipolar_space(x)), self.p1d))))
        ama_space = self.drop_layer(self.amacrine_space(bipolar_out))
        gang_space = self.drop_layer(self.ganglions_space(bipolar_out))
        cell_ama = torch.unsqueeze(torch.flatten(ama_space,start_dim=3), -1)
        ama_pre_pad = self.drop_layer(self.amacrine_pull(cell_ama[:,:,:,:-2,:]))
        ama_out = self.drop_layer(self.amacrine_alpha(F.relu(self.drop_layer(self.amacrine_temporal(F.pad(ama_pre_pad, self.p1d))))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(gang_space,start_dim=3), -25, -1), -1)
        gang_pre_pad = self.drop_layer(self.ganglions_pull(cell_gang[:,:,:,:-2,:]))
        gang_out = self.drop_layer(self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d)))
        ganglion_tot = self.drop_layer(F.relu(torch.sub(gang_out,torch.abs(ama_out))))
        left = self.ganglion_left_create(ganglion_tot)
        right = self.ganglion_right_create(ganglion_tot)
        fin_left = torch.squeeze(left)
        fin_right = torch.squeeze(right)
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin


if __name__ == "__main__":
    file = 'model_binary_50_2_no_drop'


    # Stationary dots in the background ???
    # Profile cell types
    # Understand dynamics
    
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(rate,perct_rate)
    # ax.set_xlabel('Velocity Multiplier')
    # ax.set_ylabel('Accuracy')
    # plt.show()    