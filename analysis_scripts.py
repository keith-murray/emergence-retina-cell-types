# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:36:09 2021

@author: Keith
"""

import torch
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt
from retina_model import Bipolar, Amacrine, Ganglion, AnalysisModel, TestModel, Dataset, CompareTensors, reConfigure, TestModel_LeftRight
from create_data import createDataTest, plotStimulus, tens_np, createSet_definedDist
from truncated_models import TransplantModel, TransplantModel_TopDist, TransplantModel_TopDist_SanAma
import random
import itertools
import numpy as np
import json
import seaborn as sns

device = torch.device("cuda:0")

def PsychometricCurveAnalysis(types,model,create=False):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    results = []
    
    for x in range(0,41):
        if create:
            createDataTest('psychometric_'+str(x) , x/10, 0, 100)
        accuracy = TestModel(net, 'testset_psychometric_'+str(x), 100)
        results.append(accuracy.item())
        
    for x in range(0,41):
        if x < 10:
            results[x] = 1 - results[x]
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot([x/10 for x in range(0,41)], results)
    
    plt.xlabel('Ratio between Velocities')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy in varied environments')
    plt.show()
    return results
    

def FindRelevantTypes(types, model, testset, threshold, exist=None):
    if exist is None:
        net = AnalysisModel(types, 0.00).to(device)
        weights = torch.load(model)
        net.load_state_dict(weights)
    else:
        net = exist
        state = net.state_dict()
    net.eval()
    
    base_result, label_dists = TestModel(net, testset, 100, label_dis=True)
    base_result = base_result.item()
    print(base_result)
    resultsB = []
    resultsA = []
    resultsG = []
    
    for i in net.children():
        if isinstance(i, Bipolar):
            temp_weight = i.bipolar_temporal.weight.data
            temp_bias = i.bipolar_temporal.bias.data
            i.bipolar_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.bipolar_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsB.append(accuracy)
            if accuracy < threshold*base_result:
                i.bipolar_temporal.weight.data = temp_weight
                i.bipolar_temporal.bias.data = temp_bias
        
        elif isinstance(i, Amacrine):
            temp_weight = i.amacrine_temporal.weight.data
            temp_bias = i.amacrine_temporal.bias.data
            i.amacrine_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.amacrine_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsA.append(accuracy)
            if accuracy < threshold*base_result:
                i.amacrine_temporal.weight.data = temp_weight
                i.amacrine_temporal.bias.data = temp_bias
            
        elif isinstance(i, Ganglion):
            temp_weight = i.ganglion_temporal.weight.data
            temp_bias = i.ganglion_temporal.bias.data
            i.ganglion_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.ganglion_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsG.append(accuracy)
            if accuracy < threshold*base_result:
                i.ganglion_temporal.weight.data = temp_weight
                i.ganglion_temporal.bias.data = temp_bias

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(resultsB, 'y')
    ax.plot(resultsA, 'r')
    ax.plot(resultsG, 'g')
    ax.plot([threshold*base_result for i in range(len(resultsB))], 'p-')
    # ax.plot([label_dists[0].item() for i in range(len(resultsB))], 'm+')
    # ax.plot([label_dists[1].item() for i in range(len(resultsB))], 'k+')

    ax.set_xlabel('Cell Type Leissioned')
    ax.set_xlabel('Accuracy')
    ax.set_title('Leissioning Cell Types to Affect Accuracy')
    plt.show()
    ax.savefig('figures/lesion_full_model.svg')
    
    if exist is not None:
        net.load_state_dict(state)
    return [resultsB, resultsA, resultsG, threshold*base_result]


def scramble(net):
    net_list = list(net.children())
    random.shuffle(net_list)
    return net_list


def PruneNonImportantCells(types, model, testset, threshold, graph=True):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    base_result = TestModel(net, testset, 100, printTF=False).item()
    resultsB = []
    resultsA = []
    resultsG = []
    cells_kept = 0
    
    for i in scramble(net):
        if isinstance(i, Ganglion):
            temp_weight = i.ganglion_temporal.weight.data
            i.ganglion_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsG.append(accuracy)
            if accuracy < threshold*base_result:
                i.ganglion_temporal.weight.data = temp_weight
                cells_kept += 1
            
    for i in scramble(net):
        if isinstance(i, Bipolar):
            temp_weight = i.bipolar_temporal.weight.data
            i.bipolar_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsB.append(accuracy)
            if accuracy < threshold*base_result:
                i.bipolar_temporal.weight.data = temp_weight
                cells_kept += 1
        
    for i in scramble(net):
        if isinstance(i, Amacrine):
            temp_weight = i.amacrine_temporal.weight.data
            i.amacrine_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            accuracy = TestModel(net, testset, 100).item()
            resultsA.append(accuracy)
            if accuracy < threshold*base_result:
                i.amacrine_temporal.weight.data = temp_weight
                cells_kept += 1
    
    print('Cells Kept: '+str(cells_kept))
    
    if graph:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(resultsB, 'y')
        ax.plot(resultsA, 'r')
        ax.plot(resultsG, 'g')
        ax.plot([threshold*base_result for i in range(len(resultsB))], 'p-')
        
        plt.xlabel('Cell Type Leissioned')
        plt.ylabel('Accuracy')
        plt.title('Continual Leissioning Cell Types to Affect Accuracy')
        plt.show()
    return (net, cells_kept)


def PrinciplePrune(types, model, testset):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    high_score = 0.90
    best_set = None
    
    basket1 = ['0','1','2','3','4','5','6','7']
    basket2 = ['0','1','2','3','4','5','6','7']
    basket3 = ['0','1','2','3','4','5','6','7']
    basket4 = ['1','2','4','6','7']
    basket5 = ['0','2','3','6','7']
    basket6 = ['0','2','3']
    
    for i in range(len(basket1)):
        basket1[i] = 'bipolar' + basket1[i]
        basket2[i] = 'amacrine' + basket2[i]
        basket3[i] = 'ganglion' + basket3[i]
    for i in range(len(basket4)):
        basket4[i] = 'bipolar' + basket4[i]
    for i in range(len(basket5)):
        basket5[i] = 'amacrine' + basket5[i]
    for i in range(len(basket6)):
        basket6[i] = 'ganglion' + basket6[i]

    baskets = set(basket1 + basket2 + basket3)
    x = list(itertools.combinations(basket4, 1)) + list(itertools.combinations(basket4, 2))
    y = list(itertools.combinations(basket5, 1)) + list(itertools.combinations(basket5, 2))
    z = list(itertools.combinations(basket6[:2], 2)) + list(itertools.combinations(basket6, 3))
    
    for a in x:
        for b in y:
            for c in z:
                temp_set = set(a + b + c)
                exclude = baskets - temp_set
                for i in exclude:
                    if 'bipolar' in i:
                        exec("""
net.{0}.bipolar_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.bipolar_temporal.bias.data = torch.zeros(1).to(device)
                         """.format(i))                    
                    elif 'amacrine' in i:
                        exec("""
net.{0}.amacrine_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.amacrine_temporal.bias.data = torch.zeros(1).to(device)
                         """.format(i))
                    elif 'ganglion' in i:
                        exec("""
net.{0}.ganglion_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.ganglion_temporal.bias.data = torch.zeros(1).to(device)
                         """.format(i))
                
                accuracy = TestModel(net, testset, 100).item()
                if accuracy > high_score:
                    # high_score = accuracy
                    # best_set = temp_set
                    print(accuracy)
                    print(temp_set)
                elif accuracy > 0.80 and accuracy == high_score:
                    print(accuracy)
                    print(temp_set)
                
                net.load_state_dict(weights)
                net.eval()
    
    return best_set, high_score


def CreateLeisionGraph():
    leision_results = []
    for x in range(75,101):
        temp_results = []
        for i in range(4):
            (net, cells_kept) = PruneNonImportantCells(8, 'models_and_data/model', 'testset_2x_speed', x/100, graph=False)
            temp_results.append(cells_kept)
        leision_results.append(sum(temp_results)/len(temp_results))
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot([x/100 for x in range(75,101)], leision_results)
    plt.xlabel('Accuracy Threshold')
    plt.ylabel('Number of Cells Kept')
    plt.title('The amount of cells necessary to maintain various accuracies')
    plt.show()


class DreamMap(nn.Module):
    def __init__(self):
        super(DreamMap, self).__init__()
        self.map = nn.parameter.Parameter(torch.rand(1,1,31,255,255)*2-1)
        self.drop = nn.Dropout(p=0.10)
        self.clip = nn.Tanh()

    def forward(self):
        fin = self.clip(self.drop(self.map))
        return fin


def TrainDeepDream(types, model, dream, epochs, iterations):
    # Create Model
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    # Optimize and Loss
    dream.train()
    optimizer = torch.optim.Adam(dream.parameters(), weight_decay=0.00001)
    res = []
    
    # Train
    for epoch in range(epochs):
        dream.train()
        for r in range(iterations):
            optimizer.zero_grad()
            inputs = dream()
            outputs = net.deepdream(inputs)
            outputs.backward(retain_graph=True)
            optimizer.step()
            res.append(outputs.item())
            
    plt.plot(res)
    dream.eval()
    print('Finished Training')


def DifferentStages(types, model, testset, num, stim=None, left=True):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    if stim is None:
        stimulus = torch.load(testset +'/'+num+'/stimulus.pt').to(device)
    else:
        stimulus = stim
    
    bipolar_outputs, amacrine_outputs, ganglion_outputs, dcsnLeft, dcsnRight = net.extractstage(stimulus)
    
    for i in range(types):
        plotStimulus(bipolar_outputs[i], 'bipolar'+str(i))
        plotStimulus(amacrine_outputs[i], 'amacrine'+str(i))
        plotStimulus(ganglion_outputs[i], 'ganglion'+str(i))
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(tens_np(torch.flatten(ganglion_outputs[i][0,0],start_dim=1))[:,:10])
        # plt.show()
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(tens_np(torch.squeeze(dcsnLeft)), 'r', label='Left Cell')
    ax.plot(tens_np(torch.squeeze(dcsnRight)), 'g', label='Right Cell')
    plt.xlabel('Time Step')
    plt.ylabel('Response')
    plt.title('Right and Left Cell Responses')
    plt.legend()
    if left:
        plt.savefig('figures/left_stim_res.svg')
    else:
        plt.savefig('figures/right_stim_res.svg')
    plt.show()


def TestFlaws(types, model, data, label, printTF=False):
    # Model
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    # Datasets
    testfunc = Dataset(data,range(label))
    testloader = torch.utils.data.DataLoader(testfunc, batch_size=1, shuffle=False, num_workers=0)
    
    # Test    
    running_loss = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = torch.unsqueeze(torch.unsqueeze(torch.squeeze(data[0].to(device)), dim=0), dim=0), data[1].to(device)
        outputs = net(inputs)
        comparison = CompareTensors(outputs, labels)
        if comparison == 0:
            plotStimulus(inputs, 'wrong_in'+str(i))
        running_loss += comparison
        
    if printTF:
        print('Acuracy: %.2f' % (running_loss/label))
    return running_loss/label


def SlowVelocityTest(types,model):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    results = []
    
    for x in range(0,41):
        createDataTest('slow_velocity', 2, 0, 100, slow_speed=x/10)
        accuracy = TestModel(net, 'testset_slow_velocity', 100)
        results.append(accuracy)
        shutil.rmtree('testset_slow_velocity')
        
    fig = plt.figure()
    ax = plt.axes()
    ax.plot([x/10 for x in range(0,41)], results)
    
    plt.xlabel('Slow Velocity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Model across various Slow Velocities')
    plt.show()
    return results


def ExtractStageActivations(net, data, label):    
    # Datasets
    testfunc = Dataset(data,range(label))
    testloader = torch.utils.data.DataLoader(testfunc, batch_size=label, shuffle=False, num_workers=0)
    
    stage_result = []
    
    # Test
    for i, data in enumerate(testloader, 0):
        inputs, labels = reConfigure(data)
        bipolar_outputs, amacrine_outputs, ganglion_outputs, dcsnLeft, dcsnRight = net.extractstage(inputs)
        
        for j in range(len(bipolar_outputs)):
            stage_result.append(torch.mean(bipolar_outputs[j]).item())
        for j in range(len(bipolar_outputs)):
            stage_result.append(torch.mean(amacrine_outputs[j]).item())
        for j in range(len(bipolar_outputs)):
            stage_result.append(torch.mean(ganglion_outputs[j]).item())
        break
    
    return stage_result


def ResponsesAcrossDifferentStages(types, model):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    all_stage_activations = []
    
    for i in range(0,34):
        testset = 'testset_2x_'+str(i)+'_dist'
        stage_activities = ExtractStageActivations(net, testset, 50)
        all_stage_activations.append(stage_activities)
    
    results = np.array(all_stage_activations).T
    count = 0
    
    for x in ['Bipolar','Amacrine','Ganglion']:
        fig = plt.figure()
        ax = plt.axes()
        for y in range(types):
            ax.plot([x*0.03 + 0.50 for x in range(0,34)],results[count, :], label=str(y))
            count += 1
        plt.xlabel('Slow Velocity')
        plt.ylabel('Average Response Magnitude')
        plt.title(x+' Type Average Response Magnitude across Slow Velocities')
        plt.legend()
        plt.savefig('figures/'+x+'_activations.svg')
        plt.show()
    
    return results


def BottomDist():
    return np.random.random_sample()/8 + 0.55


def LowerDist():
    return np.random.random_sample()/4 + 0.5


def UpperDist():
    return 3*np.random.random_sample()/4 + 0.75


def TopDist():
    return np.random.random_sample()/2 + 1


def MakeVaryingDists():
    createSet_definedDist('testset_bottom_dist',100,2,BottomDist)
    createSet_definedDist('testset_lower_dist',100,2,LowerDist)
    createSet_definedDist('testset_upper_dist',100,2,UpperDist)
    createSet_definedDist('testset_top_dist',100,2,TopDist)


def TruncateModelBarLesion(types, model, testset):
    net = AnalysisModel(types, 0.00).to(device)
    weights = torch.load(model)
    net.load_state_dict(weights)
    net.eval()
    
    base_res, true_left, true_right = TestModel_LeftRight(net, testset, 100)
    base_result = base_res
    print(base_result)
    resultsB = None
    resultsA = None
    resultsG0 = None
    resultsG2 = None
    
    for i in net.children():
        if isinstance(i, Bipolar):
            temp_weight = i.bipolar_temporal.weight.data
            temp_bias = i.bipolar_temporal.bias.data
            i.bipolar_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.bipolar_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy, left, right = TestModel_LeftRight(net, testset, 100)
            if accuracy < base_result:
                resultsB = (accuracy, left, right)
                i.bipolar_temporal.weight.data = temp_weight
                i.bipolar_temporal.bias.data = temp_bias
        
        elif isinstance(i, Amacrine):
            temp_weight = i.amacrine_temporal.weight.data
            temp_bias = i.amacrine_temporal.bias.data
            i.amacrine_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.amacrine_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy, left, right = TestModel_LeftRight(net, testset, 100)
            if accuracy < base_result:
                resultsA = (accuracy, left, right)
                i.amacrine_temporal.weight.data = temp_weight
                i.amacrine_temporal.bias.data = temp_bias
            
        elif isinstance(i, Ganglion):
            temp_weight = i.ganglion_temporal.weight.data
            temp_bias = i.ganglion_temporal.bias.data
            i.ganglion_temporal.weight.data = torch.zeros(temp_weight.shape).to(device)
            i.ganglion_temporal.bias.data = torch.zeros(temp_bias.shape).to(device)
            accuracy, left, right = TestModel_LeftRight(net, testset, 100)
            if accuracy < base_result:
                if resultsG0 is None:
                    resultsG0 = (accuracy, left, right)
                else:
                    resultsG2 = (accuracy, left, right)
                i.ganglion_temporal.weight.data = temp_weight
                i.ganglion_temporal.bias.data = temp_bias
                
    left_accuracy = [base_res*true_left, resultsB[0]*resultsB[1], resultsA[0]*resultsA[1], resultsG0[0]*resultsG0[1], resultsG2[0]*resultsG2[1]]
    right_accuracy = [base_res*true_right, resultsB[0]*resultsB[2], resultsA[0]*resultsA[2], resultsG0[0]*resultsG0[2], resultsG2[0]*resultsG2[2]]
    
    X = ['Full Model','Bipolar Lesion', 'Amacrine Lesion', 'Ganglion1 Lesion', 'Ganglion2 Lesion']
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X, left_accuracy, color = 'c', width = 0.25, label='Left Direction Accuracy')
    ax.bar(X, right_accuracy, bottom=left_accuracy, color = 'm', width = 0.25, label='Right Direction Accuracy')
    plt.ylabel('Model Accuracy Percentage')
    plt.title('Accuracy when lesioning various cells')
    plt.legend()
    plt.savefig('figures/truncated_model_lesion.svg')
    plt.show()


def ExamineAmacrineResponsesinTruncated(types, cell, testset):
    left_accuracy = []
    right_accuracy = []
    
    for i in range(8):
        net = AnalysisModel(types, 0.00).to(device)
        save_loc = 'models_and_data/'+cell+'\\model_top_dist_'+str(i)+'.pt'
        weights = torch.load(save_loc)
        net.load_state_dict(weights)
        net.eval()
        base_res, true_left, true_right = TestModel_LeftRight(net, testset, 100)
        left_accuracy.append(base_res*true_left)
        right_accuracy.append(base_res*true_right)

    X = [x for x in range(8)]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X, left_accuracy, color = 'c', width = 0.25, label='Left Direction Accuracy')
    ax.bar(X, right_accuracy, bottom=left_accuracy, color = 'm', width = 0.25, label='Right Direction Accuracy')
    plt.ylabel('Model Accuracy Percentage')
    plt.xlabel(cell+' Cell')
    plt.title('Accuracy when using different '+cell+' Cells')
    plt.legend()
    plt.savefig('figures/'+cell+'_substitute.svg')
    plt.show()


def BipolarAmacrineSubstitute():
    model_results = {}
    bipolar = [1,2,4,6,7]
    amacrine = [0,2,3,6,7]
    for b in bipolar:
        for a in amacrine:
            left_accuracy = []
            right_accuracy = []
            net = AnalysisModel(2, 0.00).to(device)
            save_loc = 'models_and_data/model_'+str(b)+'_'+str(a)+'.pt'
            weights = torch.load(save_loc)
            net.load_state_dict(weights)
            net.eval()
            
            for i in range(0,34):
                testset = 'testset_2x_'+str(i)+'_dist'
                base_res, left, right = TestModel_LeftRight(net, testset, 100)
                left = base_res*left
                right = base_res*right
                left_accuracy.append(left.item())
                right_accuracy.append(right.item())
                
            model_results[str(b)+'_'+str(a)] = [left_accuracy, right_accuracy]
            current_results = {'Left Accuracy': left_accuracy, 'Right Accuracy': right_accuracy}
            velocity_labels = [x*0.03 + 0.50 for x in range(0,34)]
            
            fig, ax = plt.subplots()
            ax.stackplot(velocity_labels, current_results.values(),
                         labels=current_results.keys())
            ax.legend(loc='upper left')
            ax.set_title('Slow Velocity Curve for Bipolar '+str(b) + ' and Amacrine '+str(a))
            ax.set_xlabel('Slow Velocity')
            ax.set_ylabel('Model Accuracy')
            plt.savefig('figures/graph_'+str(b)+'_'+str(a)+'.svg')
            plt.show()

    # json_dump = json.dumps(model_results)
    # f = open("models_and_data/combinations.json","w")
    # f.write(json_dump)
    # f.close()
    return model_results


def TruncatedModelSlowCurve(types, model, sace):
    left_accuracy = []
    right_accuracy = []
    net = AnalysisModel(types, 0.00).to(device)
    save_loc = 'models_and_data/'+model+'.pt'
    weights = torch.load(save_loc)
    net.load_state_dict(weights)
    net.eval()
    
    for i in range(0,34):
        testset = 'testset_2x_'+str(i)+'_dist'
        base_res, left, right,  = TestModel_LeftRight(net, testset, 100)
        left = base_res*left
        right = base_res*right
        left_accuracy.append(left.item())
        right_accuracy.append(right.item())
        
    current_results = {'Left Accuracy': left_accuracy, 'Right Accuracy': right_accuracy}
    velocity_labels = [x*0.03 + 0.50 for x in range(0,34)]
    
    fig, ax = plt.subplots()
    ax.stackplot(velocity_labels, current_results.values(),
                 labels=current_results.keys())
    ax.legend(loc='upper left')
    ax.set_title('Slow Velocity Curve for Truncated Model')
    ax.set_xlabel('Slow Velocity')
    ax.set_ylabel('Model Accuracy')
    plt.savefig('figures/'+sace+'.svg')
    plt.show()
    return None


def BipolarSubstitute():
    model_results = {}
    # bipolar = [1,2,4,6,7]
    for b in range(8):
        left_accuracy = []
        right_accuracy = []
        only_right_accuracy = []
        net = AnalysisModel(2, 0.00).to(device)
        save_loc = 'models_and_data/model_bipolar_'+str(b)+'.pt'
        weights = torch.load(save_loc)
        net.load_state_dict(weights)
        net.eval()
        
        for i in range(0,51):
            testset = '1x_'+str(i)+'_dist'
            base_res, left, right, only_right = TestModel_LeftRight(net, testset, 30)
            left = base_res*left
            right = base_res*right
            left_accuracy.append(left.item())
            right_accuracy.append(right.item())
            only_right_accuracy.append(only_right.item())
            
        model_results[str(b)] = [left_accuracy, right_accuracy, only_right_accuracy]
    
    fig, ax = plt.subplots()
    for y in range(8):
        ax.plot([x*0.05+0.5 for x in range(0,51)], model_results[str(y)][2], label=str(y))
    # ax.stackplot(velocity_labels, current_results.values(),
    #              labels=current_results.keys())
    ax.legend(loc='upper left')
    ax.set_title('Slow Velocity Curve for Bipolar '+str(b))
    ax.set_xlabel('Fast Velocity')
    ax.set_ylabel('Model Accuracy')
    plt.savefig('figures/graph_fast_bipolar_'+str(b)+'.svg')
    plt.show()

    json_dump = json.dumps(model_results)
    f = open("models_and_data/bipolar_empha.json","w")
    f.write(json_dump)
    f.close()
    return model_results[str(y)][2]


def ControlledDirection(net, sace):
    left_accuracy = []
    right_accuracy = []
    net.eval()
    
    for i in range(0,51):
        testset = '1x_'+str(i)+'_dist'
        base_res, left, right = TestModel_LeftRight(net, testset, 28)
        left = base_res*left
        right = base_res*right
        left_accuracy.append(left.item())
        right_accuracy.append(right.item())
        
    current_results = {'Left Accuracy': left_accuracy, 'Right Accuracy': right_accuracy}
    velocity_labels = [x*0.05 + 0.50 for x in range(0,51)]
    
    fig, ax = plt.subplots()
    ax.stackplot(velocity_labels, current_results.values(),
                 labels=current_results.keys())
    plt.savefig('figures/'+sace+'.svg')
    plt.show()
    return np.array(left_accuracy) + np.array(right_accuracy)


def BipolarPrune(types, model):
    net = AnalysisModel(types, 0.00).to(device)
    save_loc = 'models_and_data/'+model+'.pt'
    weights = torch.load(save_loc)
    net.load_state_dict(weights)
    net.eval()
    
    results = np.zeros((8,51))
    basket1 = ['0','1','2','3','4','5','6','7']
    basket2 = ['0','1','2','3','4','5','6','7']
    basket3 = ['0','1','2','3','4','5','6','7']
    
    for i in range(len(basket1)):
        basket1[i] = 'bipolar' + basket1[i]
        basket2[i] = 'amacrine' + basket2[i]
        basket3[i] = 'ganglion' + basket3[i]
    
    for a in basket1:
        exclude = set(basket1) - set([a,])
        for i in exclude:
            print(i)
            if 'bipolar' in i:
                exec("""
net.{0}.bipolar_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.bipolar_temporal.bias.data = torch.zeros(1).to(device)
                """.format(i))                    
            elif 'amacrine' in i:
                exec("""
net.{0}.amacrine_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.amacrine_temporal.bias.data = torch.zeros(1).to(device)
                """.format(i))
            elif 'ganglion' in i:
                exec("""
net.{0}.ganglion_temporal.weight.data = torch.zeros(1, 1, 31, 1, 1).to(device)
net.{0}.ganglion_temporal.bias.data = torch.zeros(1).to(device)
                """.format(i))
                
        results[int(a[-1]),:] = ControlledDirection(net, 'bipolar_full_model_'+a[-1])
        net.load_state_dict(weights)
        net.eval()
    
    plt.plot([x*0.05 + 0.50 for x in range(0,51)],results.T)
    plt.savefig('figures/overlap_bipolar_accuracy.svg')
    
    return None


if __name__ == "__main__":
    
    # results = PsychometricCurveAnalysis(8,'model_2x')
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 1.01)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 1.00)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 0.95)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 0.90)
    
    # CreateLeisionGraph()
    # SlowVelocityTest(8, 'model\\model')
    # ResponsesAcrossDifferentStages(8, 'model\\model')
    # best_set, high_score = PrinciplePrune(8, 'model\\model', 'testset_2x_speed')
    
    # TransplantModel()
    # results = FindRelevantTypes(2, 'model\\model_2_types', 'testset_2x_speed', 1.00, exist=None)
    # PsychometricCurveAnalysis(2, 'model\\model_2_types')
    
    # plotStimulus(torch.load('Q:/Documents/TDS SuperUROP/testset_2x_speed/0/stimulus.pt'), 'examine_stim_right')
    # DifferentStages(2, 'model\\model_2_types', 'testset_2x_speed', '0')
    # plotStimulus(torch.load('Q:/Documents/TDS SuperUROP/testset_2x_speed/2/stimulus.pt'), 'examine_stim_left')
    # DifferentStages(2, 'model\\model_2_types', 'testset_2x_speed', '2')
    
    # accuracy = TestFlaws(2, 'model\\model_2_types', 'testset_2x_speed', 100, printTF=True)
    # SlowVelocityTest(2, 'model\\model_2_types')
    # ResponsesAcrossDifferentStages(2, 'model\\model_2_types')
    
    # MakeVaryingDists()
    # results = FindRelevantTypes(8, 'model\\model', 'testset_lower_dist', 1.10)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_lower_dist', 1.00)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_lower_dist', 0.95)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_upper_dist', 1.10)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_upper_dist', 1.00)
    # results = FindRelevantTypes(8, 'model\\model', 'testset_upper_dist', 0.95)
    
    # accuracy = TestFlaws(2, 'model\\model_2_types', 'testset_upper_dist', 100, printTF=True)
    # accuracy = TestFlaws(8, 'model\\model', 'testset_bottom_dist', 100, printTF=True)
    # accuracy = TestFlaws(2, 'model\\model_2_types', 'testset_bottom_dist', 100, printTF=True)
    
    # best_set, high_score = PrinciplePrune(8, 'model\\model', 'testset_bottom_dist')
    # print('++++++++++++++++++++')
    # best_set, high_score = PrinciplePrune(8, 'model\\model', 'testset_top_dist')
    
    # TransplantModel_TopDist()
    # results = FindRelevantTypes(2, 'model\\model_top_dist', 'testset_top_dist', 1.00)
    # results = FindRelevantTypes(2, 'model\\model_top_dist', 'testset_upper_dist', 1.00)
    # results = FindRelevantTypes(2, 'model\\model_top_dist', 'testset_lower_dist', 1.00)
    # results = FindRelevantTypes(2, 'model\\model_top_dist', 'testset_bottom_dist', 1.00)
    # accuracy = TestFlaws(2, 'model\\model_top_dist', 'testset_top_dist', 100, printTF=True)
    # plotStimulus(torch.load('Q:/Documents/TDS SuperUROP/testset_top_dist/36/stimulus.pt'), 'examine_stim_right')
    # DifferentStages(2, 'model\\model_top_dist', 'testset_top_dist', '36')
    
    # TransplantModel_TopDist_SanAma()
    # accuracy = TestFlaws(2, 'model\\model_top_dist_san_ama', 'testset_top_dist', 50, printTF=True)
    # results = FindRelevantTypes(2, 'model\\model_top_dist_san_ama', 'testset_top_dist', 1.00)
    # DifferentStages(2, 'model\\model_top_dist_san_ama', 'testset_top_dist', '36')
    
    # dream = DreamMap().to(device)
    # TrainDeepDream(2,'model\\model_top_dist',dream,1,5000)
    # plotStimulus(dream(),'deep_dream')
    # DifferentStages(2,'model\\model_top_dist',None,None,stim=dream())
    # torch.save(dream(), 'Q:\Documents\TDS SuperUROP\\model\\right_cell_stim.pt')
    
    # TruncateModelBarLesion(2, 'model\\model_top_dist', 'testset_top_dist')
    # ExamineAmacrineResponsesinTruncated(2, 'Amacrine', 'testset_top_dist')
    # ExamineAmacrineResponsesinTruncated(2, 'Bipolar', 'testset_top_dist')
    
    # TestsetVelocityDist()
    # BipolarAmacrineSubstitute()
    # TruncatedModelSlowCurve()

    # result_full = PsychometricCurveAnalysis(8,'model\\model',create=False)
    # np.save('Q:/Documents/TDS SuperUROP/model/psycho_alpha_full.npy', result_full)
    # result_ablated = PsychometricCurveAnalysis(2,'model\\model_top_dist',create=False)
    # np.save('Q:/Documents/TDS SuperUROP/model/psycho_alpha_ablated.npy', result_ablated)
    # for i in [0]:
    #     plt.plot([x/10 for x in range(0,41)],result_full, label='Full')
    #     plt.plot([x/10 for x in range(0,41)],result_ablated,label='Ablated')
    #     plt.legend()
    #     plt.xlabel('Ratio between Velocities')
    #     plt.ylabel('Accuracy')
    #     plt.title('Model accuracy in varied environments')
    #     plt.savefig('psycho_alpha.svg')
    #     plt.show()

    # TruncatedModelSlowCurve(8, 'model\\model', 'full_model_stack')
    # TruncatedModelSlowCurve(2, 'model\\model_top_dist', 'ablated_model_stack')
    # [resultsB, resultsA, resultsG, thresh] = FindRelevantTypes(8, 'model\\model', 'testset_2x_speed', 0.95)
    # ResponsesAcrossDifferentStages(8, 'model\\model')
    
    # TruncatedModelSlowCurve(2, 'model\\model_top_dist_san_bipolar', 'ablated_model_stack_san_bipolar')
    # TruncatedModelSlowCurve(2, 'model\\model_top_dist_san_ama', 'ablated_model_stack_san_ama')
    # TruncatedModelSlowCurve(2, 'model\\model_top_dist_ganglion0', 'ablated_model_stack_ganglion0')
    # TruncatedModelSlowCurve(2, 'model\\model_top_dist_ganglion1', 'ablated_model_stack_ganglion1')
    
    # DifferentStages(2,'model\\model_top_dist',None,None,stim=torch.load('Q:/Documents/TDS SuperUROP/model/top dist/left ganglion/left_cell_stim.pt'), left=True)
    # DifferentStages(2,'model\\model_top_dist',None,None,stim=torch.load('Q:/Documents/TDS SuperUROP/model/top dist/right ganglion/right_cell_stim.pt'), left=False)
    
    # save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist.pt'
    # weights = torch.load(save_loc)
    # for i in [1]:
    #     sns.heatmap(tens_np(torch.squeeze(weights['bipolar0.bipolar_space.weight'])))
    #     plt.savefig('bipolar_spacial.svg')
        
    #     plt.plot(tens_np(torch.squeeze(weights['bipolar0.bipolar_temporal.weight'])))
    #     plt.savefig('bipolar_temporal.svg')
        
    #     sns.heatmap(tens_np(torch.squeeze(weights['amacrine0.amacrine_space.weight'])[0,:,:]))
    #     plt.savefig('amacrine_spacial.svg')
        
    #     plt.plot(tens_np(torch.squeeze(weights['amacrine0.amacrine_temporal.weight'])))
    #     plt.savefig('amacrine_temporal.svg')
        
    #     sns.heatmap(tens_np(torch.squeeze(weights['ganglion0.ganglion_amacrine_space.weight'])[0,:,:]))
    #     plt.savefig('ganglion_ama_space_0.svg')

    #     sns.heatmap(tens_np(torch.squeeze(weights['ganglion1.ganglion_amacrine_space.weight'])[0,:,:]))
    #     plt.savefig('ganglion_ama_space_1.svg')
    
    # left_gang = tens_np(torch.load('Q:/Documents/TDS SuperUROP/model/top dist/left ganglion/left_cell_stim.pt'))
    # plt.imshow(left_gang[0,0,0,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('left_deep_1.svg')
    # plt.imshow(left_gang[0,0,30,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('left_deep_2.svg')
    # plt.imshow(left_gang[0,0,50,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('left_deep_3.svg')
    
    # right_gang = tens_np(torch.load('Q:/Documents/TDS SuperUROP/model/top dist/right ganglion/right_cell_stim.pt'))
    # plt.imshow(right_gang[0,0,0,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('right_deep_1.svg')
    # plt.imshow(right_gang[0,0,30,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('right_deep_2.svg')
    # plt.imshow(right_gang[0,0,50,120:220,130:230], cmap='hot', vmin=-2.0, vmax=2.0)
    # plt.savefig('right_deep_3.svg')
    
    # BipolarSubstitute()
    # BipolarPrune(8, 'model\\model')
    res = BipolarSubstitute()
    print('Fin.')
    
    
    
    
    
    
    