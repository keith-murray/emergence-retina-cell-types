# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:36:09 2021

@author: Keith
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import matplotlib.pyplot as plt
from retina_model import Bipolar, Amacrine, Ganglion, Decision, TestModel
from create_data import createDataTest, plotStimulus

device = torch.device("cuda:0")

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
    
    def dream(self, stimulus):
        bipolar_outputs = [cell(stimulus) for cell in self.bipolar_cells]
        bipolar = torch.cat(bipolar_outputs, 1)
        
        amacrine_outputs = [cell(bipolar) for cell in self.amacrine_cells]
        amacrine = torch.cat(amacrine_outputs, 1)
        
        ganglion_outputs = [cell(bipolar, amacrine) for cell in self.ganglion_cells]
        ganglion = torch.cat(ganglion_outputs, 1)
        
        dcsnLeft = self.decisionLeft(ganglion)
        dcsnRight = self.decisionRight(ganglion)
        decision = torch.cat((dcsnLeft, dcsnRight), 1)[:,:,-1,0,0]
        return -1*dcsnRight[0,0,-1,0,0]


def PsychometricCurveAnalysis(types,model):
    net = AnalysisModel(types, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\'+model+'.pt'
    weights = torch.load(save_loc)
    net.load_state_dict(weights)
    net.eval()
    
    results = []
    
    for x in range(0,31):
        createDataTest('psychometric' , x/10, 0, 200)
        accuracy = TestModel(net, 'testset_psychometric', 200)
        results.append(accuracy)
        shutil.rmtree('Q:/Documents/TDS SuperUROP/testset_psychometric')
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot([x/10 for x in range(0,31)], results)
    
    plt.xlabel('Ratio between Velocities')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy in varied environments')
    plt.show()
    return results
    

def FindRelevantTypes(types, model, testset, threshold):
    net = AnalysisModel(types, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\'+model+'.pt'
    weights = torch.load(save_loc)
    net.load_state_dict(weights)
    net.eval()
    
    base_result = TestModel(net, testset, 100).item()
    print(base_result)
    countB = 0
    resultsB = []
    countA = 0
    resultsA = []
    countG = 0
    resultsG = []
    
    for i in net.children():
        if isinstance(i, Bipolar):
            temp_weight = i.bipolar_temporal.weight.data
            i.bipolar_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsB.append(accuracy)
            if accuracy < threshold*base_result:
                i.bipolar_temporal.weight.data = temp_weight
            countB += 1
        
        elif isinstance(i, Amacrine):
            temp_weight = i.amacrine_temporal.weight.data
            i.amacrine_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsA.append(accuracy)
            if accuracy < threshold*base_result:
                i.amacrine_temporal.weight.data = temp_weight
            countA += 1
            
        elif isinstance(i, Ganglion):
            temp_weight = i.ganglion_temporal.weight.data
            i.ganglion_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsG.append(accuracy)
            if accuracy < threshold*base_result:
                i.ganglion_temporal.weight.data = temp_weight
            countG += 1

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(resultsB, 'y')
    ax.plot(resultsA, 'r')
    ax.plot(resultsG, 'g')
    
    plt.xlabel('Cell Type Leissioned')
    plt.ylabel('Accuracy')
    plt.title('Leissioning Cell Types to Affect Accuracy')
    plt.show()
    return [resultsB, resultsA, resultsG]


class DreamMap(nn.Module):
    def __init__(self):
        super(DreamMap, self).__init__()
        self.map = nn.parameter.Parameter(torch.rand(1,1,51,255,255))

    def forward(self, x):
        fin = F.relu(torch.nn.functional.dropout(self.map, p=x))
        return fin


def TrainDeepDream(types, model, dream, epochs, iterations):
    # Create Model
    net = AnalysisModel(types, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\'+model+'.pt'
    weights = torch.load(save_loc)
    net.load_state_dict(weights)
    net.eval()
    
    # Optimize and Loss
    optimizer = torch.optim.AdamW(net.parameters())
    
    # Train
    for epoch in range(epochs):
        dream.train()
        for r in range(iterations):
            optimizer.zero_grad()
            inputs = dream(0.01)
            outputs = net.dream(inputs)
            outputs.backward(retain_graph=True)
            optimizer.step()
            
    print('Finished Training')


if __name__ == "__main__":
    
    # results = PsychometricCurveAnalysis(8,'model_2x')
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 1.01)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 1.00)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 0.95)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 0.90)

    dream = DreamMap().to(device)
    TrainDeepDream(8,'model\\model',dream,1,1000)
    plotStimulus(dream(0.01),'deep_dream')
    
    