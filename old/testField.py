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
from support_binary import Dataset, optimize_func, reConfigure, tens_np, plotStimulus, graph_model

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
        bipolar_out = F.relu(self.bipolar_temporal(F.pad(self.bipolar_space(x), self.p1d)))
        ama_space = self.amacrine_space(bipolar_out)
        gang_space = self.ganglions_space(bipolar_out)
        cell_ama = torch.unsqueeze(torch.flatten(ama_space,start_dim=3), -1)
        ama_pre_pad = self.amacrine_pull(cell_ama[:,:,:,:-2,:])
        ama_out = self.amacrine_alpha(F.relu(self.amacrine_temporal(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.roll(torch.flatten(gang_space,start_dim=3), -1, -1), -1)
        gang_pre_pad = self.ganglions_pull(cell_gang[:,:,:,:-2,:])
        gang_out = self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d))
        ganglion_tot = F.relu(torch.sub(gang_out,torch.abs(ama_out)))
        right = self.ganglion_right_create(ganglion_tot)
        fin_right = torch.squeeze(right)[:,-1]
        fin_left = -1*fin_right
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin
    
    def newFor(self, x):
        bipolar_out = F.relu(self.bipolar_temporal(F.pad(self.bipolar_space(x), self.p1d)))
        print(F.pad(self.bipolar_space(x), self.p1d).shape)
        # ama_space = self.amacrine_space(bipolar_out)
        gang_space = self.ganglions_space(bipolar_out)
        # cell_ama = torch.unsqueeze(torch.flatten(ama_space,start_dim=3), -1)
        # ama_pre_pad = self.amacrine_pull(cell_ama[:,:,:,:-2,:])
        # ama_out = self.amacrine_alpha(F.relu(self.amacrine_temporal(F.pad(ama_pre_pad, self.p1d))))
        cell_gang = torch.unsqueeze(torch.flatten(gang_space,start_dim=3), -1)
        gang_pre_pad = self.ganglions_pull(cell_gang[:,:,:,:-2,:])
        gang_out = self.ganglion_temporal(F.pad(gang_pre_pad,self.p1d))
        # ganglion_tot = F.relu(torch.sub(gang_out,torch.abs(ama_out)))
        ganglion_tot = F.relu(gang_out)
        
        # plotStimulus(gang_pre_pad[0:1,2:3,:,:,:])
        
        # sns.heatmap(tens_np(gang_pre_pad[0,2,:,:,0]))
        # plt.show()
        # plt.clf()
        
        right = self.ganglion_right_create(ganglion_tot)
        fin_right = torch.squeeze(right)[:,-1]
        fin_left = -1*fin_right
        fin = torch.stack((fin_left, fin_right),dim=1)
        return fin


def evalFunc(scence, cent_loc, net):
    estimation = net(scence)
    accuracy = []
    for i in range(scence.shape[0]):
        if estimation[i][0] < estimation[i][1]:
            if cent_loc[i][0] < cent_loc[i][1]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        else:
            if cent_loc[i][0] > cent_loc[i][1]:
                accuracy.append(1)
            else:
                accuracy.append(0)
    return sum(accuracy)/len(accuracy)


if __name__ == "__main__":
    modelName = 'model_binary_50_2_no_drop'
    results = []
    for x in range(1):
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
        
        # weights['ganglions_space.weight'][:,0,0,:,:] = weights['ganglions_space.weight'][:,0,0,:,:]
        # weights['ganglions_space.bias'] = torch.zeros(5).to(device)
        
        # weights['ganglion_temporal.weight'][:,0,:,0,0] = torch.ones(5,51).to(device)
        # End Modifications
        net.load_state_dict(weights)
        net.eval()
        
        stimulus1 = torch.load('Q:/Documents/TDS SuperUROP/testset_binary_2_sparse_slow/9/stimulus.pt')
        stimulus2 = torch.load('Q:/Documents/TDS SuperUROP/testset_binary_2_sparse_slow/2/stimulus.pt')
        stimulus = torch.cat((stimulus1,stimulus2), dim=0).to(device)
        estimation = net.newFor(stimulus)
        
        # plotStimulus(stimulus1)
        
        # test_labels = range(50)
        # test_params = {'batch_size': 10,
        #           'shuffle': True,
        #           'num_workers': 0}
        # test_set = Dataset('testset_binary_2_sparse',test_labels)
        # test_generator = torch.utils.data.DataLoader(test_set, **test_params)
        # testStore = []
        # testRun = 0
        # for local_batch, local_labels in test_generator:
        #     local_batch, local_labels = reConfigure(local_batch, local_labels, device)
        #     loss = evalFunc(local_batch, local_labels, net)
        #     testStore.append(loss)
        #     testRun += loss
        # results.append(testRun/len(testStore))
        # print(testRun/len(testStore))
    # plt.plot(range(100),results)
    # plt.xlabel('Roll Factor')
    # plt.ylabel('Test accuracy')
    # plt.title('Accuracy tested against Rolling Factor')
    
    
    
    