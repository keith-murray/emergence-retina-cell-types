# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:54:35 2020

@author: Keith
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from support import environment,optimize_func,make_output_test_video,stimuli,make_test, create_summary

device = torch.device("cuda:0")


# class bipolar_convolution(nn.Module):
#     '''
#     Implementation of soft exponential activation.
#     Shape:
#         - Input: (N, *) where * means, any number of additional
#           dimensions
#         - Output: (N, *), same shape as the input
#     Parameters:
#         - alpha - trainable parameter
#     '''
#     def __init__(self, in_features, alpha = None):
#         '''
#         Initialization.
#         INPUT:
#             - in_features: shape of the input
#             - aplha: trainable parameter
#             aplha is initialized with zero value by default
#         '''
#         super(bipolar_convolution,self).__init__()
#         self.in_features = in_features

#         # initialize alpha
#         if alpha == None:
#             self.alpha = Parameter(torch.tensor([])) # create a tensor out of alpha
#         else:
#             self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
#         self.alpha.requiresGrad = True # set requiresGrad to true!

#     def forward(self, x):
#         '''
#         Forward pass of the function.
#         Applies the function to the input elementwise.
#         '''
#         if (self.alpha == 0.0):
#             return x

#         if (self.alpha < 0.0):
#             return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

#         if (self.alpha > 0.0):
#             return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha


class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        self.space_conv = nn.Conv3d(1,1,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=False)
        self.temporal_conv = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.last_lay = nn.Conv3d(1,1,(1,4,4), stride=(1, 4, 4), padding=(0, 0, 0), bias=False)
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,12,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_row_create = nn.Conv3d(1,1, (1,12,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.p1d = (0, 0, 0, 0, 30, 0)

    def forward(self, x):
        first = F.relu(self.temporal_conv(F.pad(self.space_conv(x), self.p1d)))
        out = self.last_lay(first) # Make sure that it is positive
        cell_ama = torch.unsqueeze(torch.flatten(out,start_dim=3), -1)
        ama_pre_pad = self.amacrine_create(cell_ama[:,:,:,:-1,:])
        ama_out = self.amacrine_alpha(torch.sigmoid(self.amacrine_kernel(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -1, -1), -1)
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
    scene, cent_loc = environment(30, 5)
    loss_vals, loss, pred_py, net = optimize_func(scene, cent_loc, net, 10000)
    plt.plot(loss_vals)
    make_output_test_video(pred_py, cent_loc)
    
    scenef, cent_locf = stimuli(20, 5, 0, 0, 2, 9)
    lossf = make_test(net, scenef, cent_locf)
    create_summary(net)
    
    
    
    
    
    
    
    
    
    
    