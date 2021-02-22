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
from support_binary import Dataset, optimize_func, testFunc

device = torch.device("cuda:0")


class Retinal_NET(nn.Module):
    def __init__(self):
        super(Retinal_NET, self).__init__()
        self.space_conv = nn.Conv3d(5,5,(1,10,10), stride=(1, 5, 5), padding=(0, 0, 0), bias=True) 
        self.temporal_conv = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.last_lay = nn.Conv3d(5,5,(1,5,5), stride=(1, 5, 5), padding=(0, 0, 0), bias=True)
        self.amacrine_create = nn.Conv3d(5,5,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=True)
        self.ganglions_create =  nn.Conv3d(5,5,(1,1,1), stride=(1, 2, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_kernel = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.amacrine_alpha = nn.Conv3d(5,5, (1,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_kernel = nn.Conv3d(5,5,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_left_create = nn.Conv3d(5,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.ganglion_right_create = nn.Conv3d(5,1, (1,49,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.drop_layer = nn.Dropout(p=0.33333)
        
        self.p1d = (0, 0, 0, 0, 50, 0)

    def forward(self, x):
        first = F.relu(self.drop_layer(self.temporal_conv(F.pad(self.drop_layer(self.space_conv(x)), self.p1d))))
        out = self.drop_layer(self.last_lay(first)) # Make sure that it is positive
        cell_ama = torch.unsqueeze(torch.flatten(out,start_dim=3), -1)
        ama_pre_pad = self.drop_layer(self.amacrine_create(cell_ama[:,:,:,:-2,:]))
        ama_out = self.drop_layer(self.amacrine_alpha(F.relu(self.drop_layer(self.amacrine_kernel(F.pad(ama_pre_pad, self.p1d))))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(out,start_dim=3), -25, -1), -1)
        gang_pre_pad = self.drop_layer(self.ganglions_create(cell_gang[:,:,:,:-2,:]))
        gang_out = self.drop_layer(self.ganglion_kernel(F.pad(gang_pre_pad,self.p1d)))
        ganglion_tot = self.drop_layer(F.relu(torch.sub(gang_out,torch.abs(ama_out))))
        left = self.ganglion_left_create(ganglion_tot)
        right = self.ganglion_right_create(ganglion_tot)
        fin_left = torch.squeeze(left)[:,-1]
        fin_right = torch.squeeze(right)[:,-1]
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin
    

if __name__ == "__main__":
    net = Retinal_NET().to(device)
    save_loc = input('Save file: ')
    save_loc = 'Q:\Documents\TDS SuperUROP\\' + save_loc    
    os.mkdir(save_loc)
    
    # Parameters
    params = {'batch_size': 8,
              'shuffle': True,
              'num_workers': 0}
    max_epochs = 2
    
    test_params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 0}
    
    # Datasets
    labels = range(200)
    test_labels = range(50)
    
    # Generators
    training_set = Dataset('binary_75_3_dataset',labels)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    count = 0
    
    test_set = Dataset('binary_75_3_testset',test_labels)
    test_generator = torch.utils.data.DataLoader(test_set, **test_params)
    
    # Train
    net.train()
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = torch.squeeze(local_batch.to(device)), torch.squeeze(local_labels.to(device))
            loss_vals, pred_py, net = optimize_func(local_batch, local_labels, net, 50)
            if count%10 == 0:
                plt.plot(loss_vals)
                print(count)
            count += 1
    
    torch.save(net.state_dict(), save_loc+'\model.pt')
    
    # Testing
    net.eval()
    testStore = []
    testRun = 0
    for local_batch, local_labels in test_generator:
        # Transfer to GPU
        local_batch, local_labels = torch.squeeze(local_batch.to(device)), torch.squeeze(local_labels.to(device))
        loss = testFunc(local_batch, local_labels, net)
        testStore.append(loss)
        testRun += loss
    
    print(testRun/len(testStore))
    
    
    
    