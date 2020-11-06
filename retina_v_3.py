# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 04:45:03 2020

@author: Keith
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from support import environment,optimize_func,make_output_test_video,stimuli,make_test, create_summary_prior

device = torch.device("cuda:0")


class temp_conv(nn.Module):
    def __init__(self, in_features, TFbias, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(temp_conv,self).__init__()
        self.in_features = in_features
        if alpha == None and TFbias:
            self.alpha = Parameter(torch.rand(8, device=device)*2)
            self.TFbias = True
        elif alpha == None and not TFbias:
            self.alpha = Parameter(torch.rand(7, device=device)*2)
            self.TFbias = False
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        ray = torch.arange(1,32,1,device=device).float()*0.1
        ker = self.alpha[0]*torch.cos(self.alpha[1]*torch.log(ray))+\
        self.alpha[2]*torch.cos(self.alpha[3]*torch.log(ray))+\
        self.alpha[4]*torch.cos(self.alpha[5]*torch.log(ray))+self.alpha[6]*torch.ones(31,device=device).float()
        kernel = torch.reshape(ker, (1,1,31,1,1)).to(device)
        if self.TFbias:
            return F.conv3d(x,kernel,bias=self.alpha[7:8])
        else:
            return F.conv3d(x,kernel)


class bipolar_pool(nn.Module):
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(bipolar_pool,self).__init__()
        self.in_features = in_features
        if alpha == None:
            self.alpha = Parameter(torch.rand(4,4).to(device))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        kernel = torch.reshape(F.relu(self.alpha), (1,1,1,4,4)).to(device)
        return F.conv3d(x,kernel,stride=(1,4,4))


class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        self.space_conv = nn.Conv3d(1,1,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=False)
        self.temporal_conv = temp_conv((30,1,299,20,20), False)
        self.last_lay = bipolar_pool((30,1,299,20,20))
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = temp_conv((30,1,299,12,1), True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = temp_conv((30,1,299,12,1), True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,12,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_row_create = nn.Conv3d(1,1, (1,12,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.p1d = (0, 0, 0, 0, 30, 0)

    def forward(self, x):
        first = F.relu(self.temporal_conv(F.pad(self.space_conv(x), self.p1d)))
        out = self.last_lay(first) # Make sure that it is positive
        cell_ama = torch.unsqueeze(torch.flatten(out,start_dim=3), -1)
        ama_pre_pad = self.amacrine_create(cell_ama[:,:,:,:-1,:])
        ama_out = self.amacrine_alpha(torch.sigmoid(self.amacrine_kernel(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -5, -1), -1)
        gang_pre_pad = self.ganglions_create(cell_gang[:,:,:,:-1,:])
        gang_out = self.ganglion_kernel(F.pad(gang_pre_pad,self.p1d))      
        ganglion_tot = torch.sigmoid(torch.sub(gang_out,torch.abs(ama_out)))
        rows = self.ganglion_row_create(ganglion_tot)
        cols = self.ganglion_col_create(ganglion_tot)
        cell_stack = torch.cat((rows,cols), dim=3)
        fin = torch.squeeze(cell_stack)
        return fin


if __name__ == "__main__":
    net = Retinal_NET().to(device)
    scene, cent_loc = environment(20, 4)
    loss_vals, loss, pred_py, net = optimize_func(scene, cent_loc, net, 3500)
    plt.plot(loss_vals)
    make_output_test_video(pred_py, cent_loc)
    
    scenef, cent_locf = stimuli(20, 5, 0, 0, 2, 9)
    lossf = make_test(net, scenef, cent_locf)
    create_summary_prior(net)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    