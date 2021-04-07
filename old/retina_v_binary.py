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
from support_binary import Dataset, optimize_func, evalFunc, reConfigure

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
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(gang_space,start_dim=3), -5, -1), -1)
        gang_pre_pad = self.drop_layer(self.ganglions_pull(cell_gang[:,:,:,:-2,:]))
        gang_out = self.drop_layer(self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d)))
        ganglion_tot = self.drop_layer(F.relu(torch.sub(gang_out,torch.abs(ama_out))))
        left = self.ganglion_left_create(ganglion_tot)
        right = self.ganglion_right_create(ganglion_tot)
        fin_left = torch.squeeze(left)[:,-1]
        fin_right = torch.squeeze(right)[:,-1]
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin


if __name__ == "__main__":
    net = Retinal_NET_binary().to(device)
    save_loc = input('Save file: ')
    save_loc = 'Q:\Documents\TDS SuperUROP\\' + save_loc    
    os.mkdir(save_loc)
    
    # Parameters
    params = {'batch_size': 9,
              'shuffle': True,
              'num_workers': 0}
    max_epochs = 4
    
    test_params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 0}
    
    # Datasets
    labels = range(700)
    test_labels = range(100)
    
    # Generators
    training_set = Dataset('dataset_binary_50_2',labels)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    count = 0
    
    test_set = Dataset('testset_binary_50_2',test_labels)
    test_generator = torch.utils.data.DataLoader(test_set, **test_params)
    
    for epoch in range(max_epochs):
        # Training
        net.train()
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = reConfigure(local_batch, local_labels, device)
            loss_vals, pred_py, net = optimize_func(local_batch, local_labels, net, 200)
            if count%20 == 0:
                plt.plot(loss_vals)
            count += 1
        
        net.drop_layer.p = net.drop_layer.p / 3
        # Testing
        net.eval()
        testStore = []
        testRun = 0
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            local_batch, local_labels = reConfigure(local_batch, local_labels, device)
            loss = evalFunc(local_batch, local_labels, net)
            testStore.append(loss)
            testRun += loss
        
        print(testRun/len(testStore))
    
    torch.save(net.state_dict(), save_loc+'\model.pt')
    
    
    
    
    