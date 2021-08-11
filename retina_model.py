# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:15:38 2021

@author: Keith
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0")


class Bipolar(nn.Module):
    def __init__(self, drop):
        super(Bipolar, self).__init__()
        self.bipolar_space = nn.Conv3d(1,1,(1,10,10), stride=(1, 5, 5), padding=(0, 0, 0), bias=True, groups=1)
        self.bipolar_temporal = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.drop_layer = nn.Dropout(p=drop) # Initialization will specify dropout rate
        self.p1d = (0, 0, 0, 0, 30, 0)

    def forward(self, stimulus):
        spacial = self.drop_layer(self.bipolar_space(stimulus))
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.bipolar_temporal(spacial_padded))
        return F.relu(temporal) # RELU to enforce positivity of output
    

class Amacrine(nn.Module):
    def __init__(self, drop, types):
        super(Amacrine, self).__init__()
        self.amacrine_space = nn.Conv3d(types,1,(1,5,5), stride=(1, 4, 4), padding=(0, 0, 0), bias=True, groups=1)
        self.amacrine_temporal = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1) # Exp decay?
        self.drop_layer = nn.Dropout(p=drop)
        self.p1d = (0, 0, 0, 0, 30, 0)

    def forward(self, bipolar_out):
        spacial = self.drop_layer(self.amacrine_space(bipolar_out))
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.amacrine_temporal(spacial_padded))
        return -1*F.relu(temporal) # Neg-RELU to enforce negativity of output
    

class Ganglion(nn.Module):
    def __init__(self, drop, types):
        super(Ganglion, self).__init__()
        self.ganglion_bipolar_space = nn.Conv3d(types,1,(1,5,5), stride=(1, 4, 4), padding=(0, 0, 0), bias=True, groups=1)
        self.ganglion_amacrine_space = nn.Conv3d(types,1,(1,3,3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True, groups=1)
        self.ganglion_temporal = nn.Conv3d(1,1,(31,1,1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1) # Exp decay?
        self.drop_layer = nn.Dropout(p=drop)
        self.p1d = (0, 0, 0, 0, 30, 0)

    def forward(self, bipolar_out, amacrine_out):
        spacial_bipolar = self.drop_layer(self.ganglion_bipolar_space(bipolar_out))
        spacial_amacrine = self.drop_layer(self.ganglion_amacrine_space(amacrine_out))
        spacial = spacial_bipolar + spacial_amacrine
        spacial_padded = F.pad(spacial, self.p1d)
        temporal = self.drop_layer(self.ganglion_temporal(spacial_padded))
        return F.relu(temporal) # RELU to enforce positivity of output


class Decision(nn.Module):
    def __init__(self, drop, types):
        super(Decision, self).__init__()
        self.decision_space = nn.Conv3d(types,1,(1,12,12), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, groups=1)
        self.drop_layer = nn.Dropout(p=drop)

    def forward(self, ganglion_out):
        spacial = self.drop_layer(self.decision_space(ganglion_out))
        return spacial


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
        decision = torch.cat((dcsnLeft, dcsnRight), 1)[:,:,-1,0,0] # Take the last values from the decision cells
        return decision


class AnalysisModel(nn.Module):
    def __init__(self, types, drop):
        super(AnalysisModel, self).__init__()
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
        decision = torch.cat((dcsnLeft, dcsnRight), 1)[:,:,-1,0,0]
        return decision
    
    def deepdream(self, stimulus):
        bipolar_outputs = [cell(stimulus) for cell in self.bipolar_cells]
        bipolar = torch.cat(bipolar_outputs, 1)
        
        amacrine_outputs = [cell(bipolar) for cell in self.amacrine_cells]
        amacrine = torch.cat(amacrine_outputs, 1)
        
        ganglion_outputs = [cell(bipolar, amacrine) for cell in self.ganglion_cells]
        ganglion = torch.cat(ganglion_outputs, 1)
        
        dcsnLeft = self.decisionLeft(ganglion)
        dcsnRight = self.decisionRight(ganglion)
        decision = torch.cat((dcsnLeft, dcsnRight), 1)[:,:,-1,0,0]
        return -1*(torch.sum(ganglion_outputs[0][0,0,:,8,8])) # + torch.sum(ganglion_outputs[1][0,0,:,4,4]))
        # return (torch.sum(bipolar_outputs[0][0,0,:,10:11,10:11]) + torch.sum(bipolar_outputs[0][0,0,:,30:31,30:31]))
    
    def extractstage(self, stimulus):
        bipolar_outputs = [cell(stimulus) for cell in self.bipolar_cells]
        bipolar = torch.cat(bipolar_outputs, 1)
        
        amacrine_outputs = [cell(bipolar) for cell in self.amacrine_cells]
        amacrine = torch.cat(amacrine_outputs, 1)
        
        ganglion_outputs = [cell(bipolar, amacrine) for cell in self.ganglion_cells]
        ganglion = torch.cat(ganglion_outputs, 1)
        
        dcsnLeft = self.decisionLeft(ganglion)
        dcsnRight = self.decisionRight(ganglion)
        return bipolar_outputs, amacrine_outputs, ganglion_outputs, dcsnLeft, dcsnRight


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
        dirr = 'Q:/Documents/Github'+os.sep+self.file_name + os.sep + str(ID)

        # Load data and get label
        X = torch.load(dirr+os.sep+'stimulus.pt')
        y = torch.load(dirr+os.sep+'label.pt')
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


def reConfigure(data):
    return torch.unsqueeze(torch.squeeze(data[0].to(device)), dim=1), torch.squeeze(data[1].to(device))


def OptimizeModel(net, dataset, label, epochs, iterations):    
    # Datasets
    trainfunc = Dataset('dataset_'+dataset,range(label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=15, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(net.parameters())
    lossfunc = torch.nn.CrossEntropyLoss()
    weightClipper = WeightReinforcer(lower_bound=0.0)
    loss_results = []
    
    # Train
    for epoch in range(epochs):
    
        for i, data in enumerate(trainloader, 0):
            inputs, labels = reConfigure(data)
            net.train()
            for r in range(iterations):
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = lossfunc(outputs, labels[:,1].long())
                # if CompareTensors(outputs, labels).item() == 25:
                #     print(loss.item())
                #     print(r)
                #     break
                loss.backward(retain_graph=True)
                optimizer.step()
                net.apply(weightClipper)
            
            # if TestModel(net, 'testset_'+dataset, 100, printTF=True) == 1.00:
            #     break
        loss_results.append(loss.item())
            
    print('Finished Training')
    return loss_results


def CompareTensors(outputs, labels):
    right_decisions = outputs[:,1] - outputs[:,0]
    compare = torch.eq(right_decisions > 0, labels[:,1] > 0.5)
    return torch.sum(compare)


def TestModel(net, data, label, printTF=False, label_dis=False):
    # Datasets
    testfunc = Dataset(data,range(label))
    testloader = torch.utils.data.DataLoader(testfunc, batch_size=int(label/2), shuffle=True, num_workers=0)
    
    # Enable Testing
    net.eval()
    label_store = torch.zeros(2).to(device)
    
    # Test    
    running_loss = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = reConfigure(data)
        label_store = label_store + torch.sum(labels, 0)
        outputs = net(inputs)
        running_loss += CompareTensors(outputs, labels)
        
    if printTF:
        print('Acuracy: %.2f' % (running_loss/label))
    if label_dis:
        return running_loss/label, label_store/label
    else:
        return running_loss/label


def CompareTensors_LeftRight(outputs, labels):
    right_decisions = outputs[:,1] - outputs[:,0]
    out = right_decisions > 0
    label = labels[:,1] > 0.5
    compare = torch.eq(out, label)
    left = torch.logical_not(torch.logical_or(out, label))
    right = torch.logical_and(out, label)
    return torch.sum(compare), torch.sum(left), torch.sum(right), torch.sum(right)/torch.sum(label)


def tens_np(tens):
    return tens.cpu().detach().numpy()


def TestModel_LeftRight(net, data, label):
    # Datasets
    testfunc = Dataset(data,range(label))
    testloader = torch.utils.data.DataLoader(testfunc, batch_size=int(label/2), shuffle=True, num_workers=0)
    
    # Enable Testing
    net.eval()
    label_store = torch.zeros(2).to(device)
    
    # Test    
    running_loss = 0
    running_left = 0
    running_right = 0
    right_a = []
    for i, data in enumerate(testloader, 0):
        inputs, labels = reConfigure(data)
        label_store = label_store + torch.sum(labels, 0)
        outputs = net(inputs)
        temp_loss, temp_left, temp_right, right_acc = CompareTensors_LeftRight(outputs, labels)
        right_a.append(right_acc)
        running_loss += temp_loss
        running_left += temp_left
        running_right += temp_right
    
    accuracy_right = tens_np(sum(right_a)/len(right_a))
    return tens_np(running_loss/label), tens_np(running_left/running_loss), tens_np(running_right/running_loss), accuracy_right


if __name__ == "__main__":
    net = RetinaModel(8, 0.40).to(device)
    loss = OptimizeModel(net, '2x_speed', 1000, 1000, 1)
    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model_graph.pt')
    plt.plot(loss)
    
    
    