# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:31 2020

@author: Keith
"""
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from celluloid import Camera
import seaborn as sns

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
        self.drop_layer = nn.Dropout(p=0.00)
        
        self.p1d = (0, 0, 0, 0, 50, 0)

    def forward(self, x):
        bipolar_out = F.relu(self.drop_layer(self.bipolar_temporal(F.pad(self.drop_layer(self.bipolar_space(x)), self.p1d))))
        ama_space = self.drop_layer(self.amacrine_space(bipolar_out)) # Enforce Excitatory
        gang_space = self.drop_layer(self.ganglions_space(bipolar_out)) # Enforce Excitatory
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
        bipolar_out = (self.bipolar_temporal(F.pad(self.bipolar_space(x), self.p1d)))
        ama_space = self.amacrine_space(bipolar_out)
        gang_space = self.ganglions_space(bipolar_out)
        cell_ama = torch.unsqueeze(torch.flatten(ama_space,start_dim=3), -1)
        ama_pre_pad = self.amacrine_pull(cell_ama[:,:,:,:-2,:])
        ama_out = self.amacrine_alpha((self.amacrine_temporal(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(gang_space,start_dim=3), -1, -1), -1)
        gang_pre_pad = self.ganglions_pull(cell_gang[:,:,:,:-2,:])
        gang_out = self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d))
        ganglion_tot = (torch.sub(gang_out,torch.abs(ama_out)))
        return -1*torch.sum(bipolar_out[0,2,-1,:10,:10])

class DreamMap(nn.Module):
    def __init__(self):
        super(DreamMap, self).__init__()
        self.map = nn.parameter.Parameter(torch.rand(1,1,51,255,255))

    def forward(self, x):
        fin = F.relu(torch.nn.functional.dropout(self.map, p=x))
        return fin


def findDreamMap(modelName):
    '''
    modelName = 'binary_75_3_3333'
    testSet = 'binary_75_3_testset'
    '''
    net = Retinal_NET_binary().to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP' + os.sep + modelName + os.sep + 'model.pt'
    weights = torch.load(save_loc)
    # Modifications
    weights['ganglion_right_create.weight'][0,3:,0,:,0] = torch.zeros(2,49).to(device)
    weights['ganglion_right_create.weight'][0,:2,0,:,0] = torch.zeros(2,49).to(device)
    weights['ganglion_right_create.weight'][0,2,0,:,0] = torch.ones(49).to(device)        
    weights['amacrine_pull.weight'][:,0,0,0,0] = torch.zeros(5).to(device)
    weights['ganglions_pull.weight'][:2,0,0,0,0] = torch.zeros(2).to(device)
    weights['ganglions_pull.weight'][3:,0,0,0,0] = torch.zeros(2).to(device)
    weights['ganglions_pull.weight'][2:3,0,0,0,0] = -1*torch.ones(1).to(device)
    # End Modifications
    net.load_state_dict(weights)
    net.eval()

    dreamMap = DreamMap().to(device)
    dreamMap.train()
    
    # Training
    optimizer = torch.optim.Adam(dreamMap.parameters(), weight_decay=0.01)
    res = []

    for i in range(1000):
        optimizer.zero_grad()
        estimation = dreamMap(0.001)
        output = net.outputCell(estimation)
        res.append(output.item())
        output.backward(retain_graph=True)
        optimizer.step()
        
    # optimizer = torch.optim.Adam(dreamMap.parameters(), weight_decay=0.01)
    # for i in range(500):
    #     optimizer.zero_grad()
    #     estimation = dreamMap(0.0)
    #     output = net.outputCell(estimation)
    #     res.append(output.item())
    #     output.backward(retain_graph=True)
    #     optimizer.step()
        
    plt.plot(res)
        
    dreamMap.eval()
    dream = dreamMap(0)
    return dreamMap, dream
# How much readout to decision variable from bipolar vs amacrine vs ganglion (readout other directions (left,right))
# How much time is needed to reach a decision
# PCA!!!

def plotDream(stim):
    fig = plt.figure()
    camera = Camera(fig)
    test = stim.cpu().detach().numpy()[0,0,:,:,:]
    for i in range(test.shape[0]):
        plt.imshow(test[i,:100,:100], cmap='hot', interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/dreamScene.gif', writer = 'pillow', fps=30)
    return


if __name__ == "__main__":
    maap, dreams = findDreamMap('model_binary_50_2_no_drop')
    plotDream(dreams)
    
    
    
    