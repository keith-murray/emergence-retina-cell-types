# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:05:56 2020

@author: Keith
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from support_discrim import environment, optimize_func, create_summary_prior, createTest

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
            self.alpha = Parameter(torch.rand(6, device=device))
            self.TFbias = True
        elif alpha == None and not TFbias:
            self.alpha = Parameter(torch.rand(5, device=device))
            self.TFbias = False
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        ray = torch.arange(1,42,1,device=device).float()
        ker = self.alpha[0]*torch.cos(self.alpha[1]*torch.log(self.alpha[2]*ray+torch.abs(self.alpha[3])))+\
        self.alpha[4]*torch.ones(41,device=device).float()
        kernel = torch.reshape(torch.flip(ker,(0,)), (1,1,41,1,1)).to(device)
        if self.TFbias:
            return F.conv3d(x,kernel,bias=self.alpha[5:6])
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
            self.alpha = Parameter(torch.rand(5,5).to(device))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        kernel = torch.relu(torch.reshape(F.relu(self.alpha), (1,1,1,5,5)).to(device))
        return F.conv3d(x,kernel,stride=[1,5,5])
    
    
class biploar_space(nn.Module):
    def __init__(self, in_features, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(biploar_space,self).__init__()
        self.in_features = in_features
        if alpha == None:
            self.alpha = Parameter(torch.rand(8, device=device))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True
    def forward(self,inpu):
        x_t_1 = self.alpha[0]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/self.alpha[1])**2)
        x_t_2 = self.alpha[2]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/self.alpha[3])**2)
        y_t_1 = self.alpha[4]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/self.alpha[5])**2)
        y_t_2 = self.alpha[6]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float,requires_grad=True).float()/self.alpha[7])**2)
        x_t = x_t_1 - x_t_2
        y_t = y_t_1 - y_t_2
        base = torch.mul(torch.unsqueeze(x_t,-1),torch.unsqueeze(y_t,0))
        base = torch.unsqueeze(base,0)
        base = torch.unsqueeze(base,0)
        base = torch.unsqueeze(base,0)
        return F.conv3d(inpu,base,stride=[1,5,5])
        

class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        # self.space_conv = nn.Conv3d(1,1,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=False)
        self.space_conv = biploar_space((5,1,299,100,100))
        self.temporal_conv = temp_conv((30,1,299,20,20), False)
        # self.temporal_conv = nn.Conv3d(1,1,(41,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        # self.last_lay = nn.Conv3d(1,1,(1,4,4), stride=(1, 4, 4), padding=(0, 0, 0), bias=False)
        self.last_lay = bipolar_pool((5,1,299,20,20))
        self.amacrine_create = nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.ganglions_create =  nn.Conv3d(1,1,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=False)
        self.amacrine_kernel = temp_conv((30,1,299,12,1), True)
        # self.amacrine_kernel = nn.Conv3d(1,1,(41,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(1,1, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.ganglion_kernel = temp_conv((30,1,299,12,1), True)
        # self.ganglion_kernel = nn.Conv3d(1,1,(41,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_col_create = nn.Conv3d(1,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.p1d = (0, 0, 0, 0, 40, 0)

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
    envi, res = environment(5, 250,200,160,0.2,0.05,0.5)
    # torch.autograd.set_detect_anomaly(True)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 500)
    plt.plot(loss_vals)
    
    del envi
    envi, res = environment(4, 250,200,160,0.2,0.05,0.5)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 450)
    plt.plot(loss_vals)

    del envi
    envi, res = environment(4, 250,200,160,0.2,0.05,0.5)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 400)
    plt.plot(loss_vals)
    
    del envi
    envi, res = environment(4, 250,200,160,0.2,0.05,0.5)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 350)
    plt.plot(loss_vals)
    
    del envi
    envi, res = environment(4, 250,200,160,0.2,0.05,0.5)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 300)
    plt.plot(loss_vals)
    
    del envi
    envi, res = environment(4, 250,200,160,0.2,0.05,0.5)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 250)
    plt.plot(loss_vals)

    create_summary_prior(net)
    testRes = createTest(net, 250,200,160,0.2,0.05,0.5)
    print(testRes)
    
    
    
    
    