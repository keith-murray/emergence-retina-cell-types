# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:15:38 2021

@author: Keith
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")


class Bipolar(nn.Module):
    def __init__(self, drop):
        super(Bipolar, self).__init__()
        self.bipolar_space = nn.Conv3d(1,1,(1,10,10), stride=(1, 5, 5), padding=(0, 0, 0), bias=True, groups=1)
        self.bipolar_temporal = nn.Conv3d(1,1,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.drop_layer = nn.Dropout(p=drop) # Initialization will specify dropout rate
        self.p1d = (0, 0, 0, 0, 50, 0) # Manual pad to preference the begining

    def forward(self, stimulus):
        spacial = self.drop_layer(self.bipolar_space(stimulus))
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.bipolar_temporal(spacial_padded))
        return F.leaky_relu(temporal) # RELU to enforce positivity of output
    

class Amacrine(nn.Module):
    def __init__(self, drop, types):
        super(Amacrine, self).__init__()
        self.amacrine_space = nn.Conv3d(types,1,(1,5,5), stride=(1, 4, 4), padding=(0, 0, 0), bias=True, groups=1)
        self.amacrine_temporal = nn.Conv3d(1,1,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1) # Exp decay?
        self.drop_layer = nn.Dropout(p=drop)
        self.p1d = (0, 0, 0, 0, 50, 0)

    def forward(self, bipolar_out):
        spacial = self.drop_layer(self.amacrine_space(bipolar_out))
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.amacrine_temporal(spacial_padded))
        return -1*F.leaky_relu(temporal) # Neg-RELU to enforce negativity of output
    

class Ganglion(nn.Module):
    def __init__(self, drop, types):
        super(Ganglion, self).__init__()
        self.ganglion_bipolar_space = nn.Conv3d(types,1,(1,5,5), stride=(1, 4, 4), padding=(0, 0, 0), bias=True, groups=1)
        self.ganglion_amacrine_space = nn.Conv3d(types,1,(1,3,3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True, groups=1)
        self.ganglion_temporal = nn.Conv3d(1,1,(51,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1) # Exp decay?
        self.drop_layer = nn.Dropout(p=drop)
        self.p1d = (0, 0, 0, 0, 50, 0)

    def forward(self, bipolar_out, amacrine_out):
        spacial_bipolar = self.drop_layer(self.ganglion_bipolar_space(bipolar_out))
        spacial_amacrine = self.drop_layer(self.ganglion_amacrine_space(amacrine_out))
        spacial = spacial_bipolar + spacial_amacrine
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.ganglion_temporal(spacial_padded)) # Less opinions about positivity
        return F.leaky_relu(temporal) # RELU to enforce positivity of output


class Decision(nn.Module):
    def __init__(self, drop, types):
        super(Decision, self).__init__()
        self.decision_space = nn.Conv3d(types,1,(1,12,12), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.drop_layer = nn.Dropout(p=drop)

    def forward(self, ganglion_out):
        spacial = self.drop_layer(self.decision_space(ganglion_out))
        return F.leaky_relu(spacial) # RELU to enforce positivity of output


class RetinaModel(nn.Module):
    def __init__(self, types, drop):
        super(RetinaModel, self).__init__()
        self.bipolar_cells = []
        self.amacrine_cells = []
        self.ganglion_cells = []
        
        for i in range(types): # BEWARE: Very hacky code below, be CAUTIOUS
            exec("""
self.bipolar{0} = Bipolar(drop)
self.bipolar_cells.append(self.bipolar{0})
self.amacrine{0} = Amacrine(drop, types)
self.amacrine_cells.append(self.amacrine{0})
self.ganglion{0} = Ganglion(drop, types)
self.ganglion_cells.append(self.ganglion{0})
                 """.format(i)) # Scary code done now, always be cautious of exec()
        
        self.decisionLeft = Decision(drop, types)
        self.decisionRight = Decision(drop, types)

    def forward(self, stimulus):
        bipolar_outputs = [cell(stimulus) for cell in self.bipolar_cells]
        bipolar = torch.cat(bipolar_outputs, 1)
        
        amacrine_outputs = [cell(bipolar) for cell in self.amacrine_cells]
        amacrine = torch.cat(amacrine_outputs, 1)
        
        ganglion_outputs = [cell(bipolar, amacrine) for cell in self.ganglion_cells]
        ganglion = torch.cat(ganglion_outputs, 1)
        
        dcsnLeft = self.decisionLeft(ganglion)
        dcsnRight = self.decisionRight(ganglion)
        decision = torch.sum(torch.cat((dcsnLeft, dcsnRight), 1)[:,:,:,0,0], dim=2) # Take the last values from the decision cells
        return decision


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_name, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.file_name = file_name

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        dirr = 'Q:\Documents\TDS SuperUROP\\'+self.file_name + os.sep + str(ID)

        # Load data and get label
        X = torch.load(dirr+'\stimulus.pt')
        y = torch.load(dirr+'\label.pt')
        return X, y


class WeightReinforcer(object):
    def __init__(self, lower_bound=0):
        self.bound = lower_bound
        
    def __call__(self, module):
        if isinstance(module, Amacrine):
            w_a_space = module.amacrine_space.weight.data
            w_a_space = torch.clamp(w_a_space, min=self.bound)
            module.amacrine_space.weight.data = w_a_space
            
            w_a_temporal = module.amacrine_temporal.weight.data
            w_a_temporal = torch.clamp(w_a_temporal, min=self.bound)
            module.amacrine_temporal.weight.data = w_a_temporal
            
        elif isinstance(module, Ganglion):
            w_g_b_space = module.ganglion_bipolar_space.weight.data
            w_g_b_space = torch.clamp(w_g_b_space, min=self.bound)
            module.ganglion_bipolar_space.weight.data = w_g_b_space
            
            w_g_a_space = module.ganglion_amacrine_space.weight.data
            w_g_a_space = torch.clamp(w_g_a_space, min=self.bound)
            module.ganglion_amacrine_space.weight.data = w_g_a_space
            
            w_g_temporal = module.ganglion_temporal.weight.data
            w_g_temporal = torch.clamp(w_g_temporal, min=self.bound)
            module.ganglion_temporal.weight.data = w_g_temporal
        
        elif isinstance(module, Decision):
            w_d_space = module.decision_space.weight.data
            w_d_space = torch.clamp(w_d_space, min=self.bound)
            module.decision_space.weight.data = w_d_space


class WeightInitializer(object):
    def __init__(self, init=1):
        self.init = init
        
    def __call__(self, module):
        if isinstance(module, Bipolar):
            w_b_space = module.bipolar_space.weight.data + self.init + np.random.random_sample()
            module.bipolar_space.weight.data = w_b_space
            
            w_b_temporal = module.bipolar_temporal.weight.data + self.init + np.random.random_sample()
            module.bipolar_temporal.weight.data = w_b_temporal
            
        elif isinstance(module, Amacrine):
            w_a_space = module.amacrine_space.weight.data + self.init + np.random.random_sample()
            module.amacrine_space.weight.data = w_a_space
            
            w_a_temporal = module.amacrine_temporal.weight.data + self.init + np.random.random_sample()
            module.amacrine_temporal.weight.data = w_a_temporal
            
        elif isinstance(module, Ganglion):
            w_g_b_space = module.ganglion_bipolar_space.weight.data + self.init + np.random.random_sample()
            module.ganglion_bipolar_space.weight.data = w_g_b_space
            
            w_g_a_space = module.ganglion_amacrine_space.weight.data + self.init + np.random.random_sample()
            module.ganglion_amacrine_space.weight.data = w_g_a_space
            
            w_g_temporal = module.ganglion_temporal.weight.data + self.init + np.random.random_sample()
            module.ganglion_temporal.weight.data = w_g_temporal
        
        elif isinstance(module, Decision):
            w_d_space = module.decision_space.weight.data + self.init + np.random.random_sample()
            module.decision_space.weight.data = w_d_space


def reConfigure(data):
    return torch.unsqueeze(torch.squeeze(data[0].to(device)), dim=1), torch.squeeze(data[1].to(device))


def OptimizeModel(net, data, label, epochs, iterations):    
    # Datasets
    trainfunc = Dataset(data,range(label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=25, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.AdamW(net.parameters())
    lossfunc = torch.nn.CrossEntropyLoss()
    weightClipper = WeightReinforcer(lower_bound=0.0)
    
    # Enable Training
    net.train()
    
    # Train
    for epoch in range(epochs):
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = reConfigure(data)
            for r in range(iterations):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = lossfunc(outputs, labels[:,1].long())
                loss.backward(retain_graph=True)
                optimizer.step()
                net.apply(weightClipper)
                running_loss += loss.item()
                
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/iterations*10))
                running_loss = 0.0
    
    print('Finished Training')


def CompareTensors(outputs, labels):
    right_decisions = []
    for i in range(outputs.shape[0]):
        if outputs[i,1] > outputs[i,0]:
            right_decisions.append(1)
        else:
            right_decisions.append(0)
    right_decisions = torch.tensor(right_decisions).to(device)
    compare = torch.eq(right_decisions, labels[:,1])
    return torch.sum(compare)


def TestModel(net, data, label):    
    # Datasets
    testfunc = Dataset(data,range(label))
    testloader = torch.utils.data.DataLoader(testfunc, batch_size=10, shuffle=True, num_workers=0)
    
    # Enable Testing
    net.eval()
    
    # Test    
    running_loss = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = reConfigure(data)
        outputs = net(inputs)
        running_loss += CompareTensors(outputs, labels)
        
    print('Acuracy: %.2f' % (running_loss/label))
    print('Finished Testing')


if __name__ == "__main__":
    net = RetinaModel(10, 0.35).to(device)
    initialize = WeightInitializer(init=0.5)
    net.apply(initialize)
    OptimizeModel(net, 'dataset_2_5x_speed', 1000, 2, 10)
    TestModel(net, 'testset_2_5x_speed', 100)
    
    
    