# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 22:12:12 2021

@author: murra
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from celluloid import Camera
from create_data import tens_np
from retina_model import Bipolar, Amacrine, Ganglion, AnalysisModel, TestModel, TestModel_LeftRight

device = torch.device('cpu')
plt.rcParams["font.family"] = "arial"
plt.rcParams['font.size'] = 16

def PsycometricFunction():
    model = np.load('models_and_data/psycho_alpha_full.npy')[10:]
    ablated = np.load('models_and_data/psycho_alpha_ablated.npy')[10:]
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(0,len(model))/10+1, model, label='Full Model')
    ax.plot(np.arange(0,len(model))/10+1, ablated, label='Ablated Model')
        
    ax.set_ylabel('Accuracy on Testset')
    ax.set_xlabel(r'Speed Ratio ($\alpha$)')
    ax.legend()
    
    plt.savefig('figures/psychometric_fig.svg')
    plt.show()
    
    return None

def ModelAblation():
    bipolar = np.load('models_and_data/bipolar_ablation.npy')
    amacrine = np.load('models_and_data/amacrine_ablation.npy')
    ganglion = np.load('models_and_data/ganglion_ablation.npy')
    
    width = 0.9
    
    fig, ax = plt.subplots()
    
    ax.bar(np.arange(0,8), bipolar, width, label='Bipolar')
    ax.bar(np.arange(9,17), amacrine, width, label='Amacrine')
    ax.bar(np.arange(18,26), ganglion, width, label='Ganglion')
        
    ax.set_ylabel('Accuracy on Testset')
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel('Cell Type Ablated')
    ax.set_xticks([3.45,12.45,21.5])
    ax.set_xticklabels(['Bipolar','Amacrine','Ganglion'])
    # ax.legend()
    
    plt.savefig('figures/cell_type_ablation_fig.svg')
    plt.show()
    
    return None

def figure3():
    model = np.load('models_and_data/psycho_alpha_full.npy')[10:]
    ablated = np.load('models_and_data/psycho_alpha_ablated.npy')[10:]
    
    bipolar = np.load('models_and_data/bipolar_ablation.npy')
    amacrine = np.load('models_and_data/amacrine_ablation.npy')
    ganglion = np.load('models_and_data/ganglion_ablation.npy')
    
    width = 0.9
    
    f, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 3]})
    
    a0.plot(np.arange(0,len(model))/10+1, model, label='Full Model')
    a0.plot(np.arange(0,len(model))/10+1, ablated, label='Ablated Model')
        
    a0.set_ylabel('Accuracy on Testset')
    a0.set_xlabel(r'Speed Ratio ($\alpha$)')
    a0.set_title('A', loc='left')
    a0.legend()
    
    a1.bar(np.arange(0,8), bipolar, width, label='Bipolar')
    a1.bar(np.arange(9,17), amacrine, width, label='Amacrine')
    a1.bar(np.arange(18,26), ganglion, width, label='Ganglion')
        
    a1.set_ylabel('Accuracy on Testset')
    a1.set_ylim(bottom=0.5)
    a1.set_xlabel('Cell Type Ablated')
    a1.set_xticks([3.45,12.45,21.5])
    a1.set_xticklabels(['Bipolar','Amacrine','Ganglion'])
    a1.set_title('B', loc='left')
    f.tight_layout()
    
    plt.savefig('figures/jcb figs/figure 3.svg', format="svg")
    plt.show()

    return None

def plotStimulus(testSet, name):
    '''
    testSet = 'binary_75_3_testset'
    '''
    fig = plt.figure()
    camera = Camera(fig)
    test = testSet.cpu().detach().numpy()[0,0,:,139:211,139:211]
    for i in range(test.shape[0]):
        plt.imshow(test[i,:,:], cmap='coolwarm', vmin=-1.0, vmax=1.0)
        camera.snap()
    animation = camera.animate()
    animation.save(name+'.gif', writer = 'pillow', fps=15)
    
    return None

def DeepDream():
    right = torch.load('models_and_data/right_cell_stim.pt',
                       map_location=torch.device('cpu')).cpu().detach().numpy()[0,0,:,139:211,139:211]
    left = torch.load('models_and_data/left_cell_stim.pt',
                      map_location=torch.device('cpu')).cpu().detach().numpy()[0,0,:,139:211,139:211]
    
    f, axs = plt.subplots(2, 3, figsize=(7, 4.25))
    
    im = axs[0,0].imshow(right[0], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].set_ylabel('Ganglion cell type 0')
    axs[0,0].set_title('A')
    axs[0,1].imshow(right[30], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,2].imshow(right[45], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,2].set_xticks([])
    axs[0,2].set_yticks([])
    
    axs[1,0].imshow(left[0], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    axs[1,0].set_ylabel('Ganglion cell type 2')
    axs[1,0].set_title('C')
    axs[1,1].imshow(left[30], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])
    axs[1,2].imshow(left[45], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,2].set_xticks([])
    axs[1,2].set_yticks([])

    cax = plt.axes([0.92, 0.125, 0.0225, 0.75])
    plt.colorbar(im, cax=cax)
    plt.show()
    
    return None

def DifferentStages():
    right = torch.load('models_and_data/right_cell_stim.pt',
                       map_location=torch.device('cpu'))
    left = torch.load('models_and_data/left_cell_stim.pt',
                      map_location=torch.device('cpu'))
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'models_and_data/model_top_dist.pt'
    weights = torch.load(save_loc, map_location=torch.device('cpu'))
    net.load_state_dict(weights)
    net.eval()
        
    _, _, _, rightdcsnLeft, rightdcsnRight = net.extractstage(right)
    _, _, _, leftdcsnLeft, leftdcsnRight = net.extractstage(left)
    
    fig, (a0, a1) = plt.subplots(2, 1)
    a0.plot(tens_np(torch.squeeze(rightdcsnRight)), 'g', label='Right Cell')
    a0.plot(tens_np(torch.squeeze(rightdcsnLeft)), 'r', label='Left Cell')
    a0.plot(range(51), np.zeros(51), 'k-', alpha=0.25)
    a0.set_title('B')
    a0.set_xticks([])
    a0.set_ylabel('Activity magnitude')
    a0.legend()
    
    a1.plot(tens_np(torch.squeeze(leftdcsnRight)), 'g', label='Right Cell')
    a1.plot(tens_np(torch.squeeze(leftdcsnLeft)), 'r', label='Left Cell')
    a1.plot(range(51), np.zeros(51), 'k-', alpha=0.25)
    a1.set_title('D')
    a1.set_xlabel('Time Step')
    a1.set_ylabel('Activity magnitude')

    plt.show()

    return None

def figure4():
    right = torch.load('models_and_data/right_cell_stim.pt',
                       map_location=torch.device('cpu'))
    left = torch.load('models_and_data/left_cell_stim.pt',
                      map_location=torch.device('cpu'))
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'models_and_data/model_top_dist.pt'
    weights = torch.load(save_loc, map_location=torch.device('cpu'))
    net.load_state_dict(weights)
    net.eval()
        
    _, _, _, rightdcsnLeft, rightdcsnRight = net.extractstage(right)
    _, _, _, leftdcsnLeft, leftdcsnRight = net.extractstage(left)
    
    right = torch.load('models_and_data/right_cell_stim.pt',
                       map_location=torch.device('cpu')).cpu().detach().numpy()[0,0,:,139:211,139:211]
    left = torch.load('models_and_data/left_cell_stim.pt',
                      map_location=torch.device('cpu')).cpu().detach().numpy()[0,0,:,139:211,139:211]
    
    f, axs = plt.subplots(2, 4, figsize=(18, 7.5))
    
    im = axs[0,0].imshow(right[0], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].set_ylabel('Ganglion cell type 0')
    axs[0,0].set_title('A', loc='left')
    axs[0,1].imshow(right[30], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])
    axs[0,2].imshow(right[45], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[0,2].set_xticks([])
    axs[0,2].set_yticks([])
    
    axs[1,0].imshow(left[0], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    axs[1,0].set_ylabel('Ganglion cell type 2')
    axs[1,0].set_title('C', loc='left')
    axs[1,1].imshow(left[30], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])
    axs[1,2].imshow(left[45], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axs[1,2].set_xticks([])
    axs[1,2].set_yticks([])

    axs[0,3].plot(tens_np(torch.squeeze(rightdcsnRight)), 'g', label='Right Cell')
    axs[0,3].plot(tens_np(torch.squeeze(rightdcsnLeft)), 'r', label='Left Cell')
    axs[0,3].plot(range(51), np.zeros(51), 'k-', alpha=0.25)
    axs[0,3].set_title('B', loc='left')
    axs[0,3].set_xticks([])
    axs[0,3].set_ylabel('Activity magnitude')
    axs[0,3].legend()
    
    axs[1,3].plot(tens_np(torch.squeeze(leftdcsnRight)), 'g', label='Right Cell')
    axs[1,3].plot(tens_np(torch.squeeze(leftdcsnLeft)), 'r', label='Left Cell')
    axs[1,3].plot(range(51), np.zeros(51), 'k-', alpha=0.25)
    axs[1,3].set_title('D', loc='left')
    axs[1,3].set_xlabel('Time Step')
    axs[1,3].set_ylabel('Activity magnitude')

    plt.colorbar(im, ax=axs[0,2])
    plt.colorbar(im, ax=axs[1,2])
    f.tight_layout()
    
    plt.savefig('figures/jcb figs/figure 4.svg', format="svg")
    plt.show()
    
    return None

def BipolarActivations():
    f = open("models_and_data/bipolar_empha.json")
    model_results = json.loads(f.read())
    f.close()
    
    fig, ax = plt.subplots()
    for y in range(8):
        ax.plot([x*0.05+0.5 for x in range(0,51)], model_results[str(y)][2], label=str(y))
    ax.legend(loc='upper left')
    ax.set_title('B',loc='left')
    ax.set_xlabel(r'$f$ speed')
    ax.set_ylabel('Probability of activation')
    
    plt.show()
    
    return None

def ExampleStimulus():
    stim = torch.load('models_and_data/stimulus.pt').cpu().detach().numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(stim[0,0,0,:,:], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('A',loc='left')
    
    plt.show()
    
    return None

def figure5():
    f = open("models_and_data/bipolar_empha.json")
    model_results = json.loads(f.read())
    f.close()
    
    stim = torch.load('models_and_data/stimulus.pt').cpu().detach().numpy()

    f, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
    a0.imshow(stim[0,0,0,:,:], cmap='gray')
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_title('A',loc='left')
    
    for y in range(8):
        a1.plot([x*0.05+0.5 for x in range(0,51)], model_results[str(y)][2], label=str(y))
    a1.legend(loc='upper left')
    a1.set_title('B',loc='left')
    a1.set_xlabel(r'$f$ speed')
    a1.set_ylabel('Probability of activation')
    
    plt.savefig('figures/jcb figs/figure 5.svg', format="svg")
    plt.show()
    
    return None

def LeftAndRight():
    data = torch.load('models_and_data/left_and_right_accuracies.pt', 
                      map_location=torch.device('cpu'))
    
    x = np.arange(len(data['right']))
    width = 0.35
    labels = ['Full', 'Ablated', 'Bipolar Ablated', 'Amacrine Ablated']
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, data['left'], width, label=r'Left $F$')
    ax.bar(x + width/2, data['right'], width, label=r'Right $F$')
    
    ax.set_ylabel('Testset Accuracy')
    ax.set_xlabel('Model')
    ax.set_title('A', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()
    
def SlowCurveAccuracy():
    data = torch.load('models_and_data/ablated_amacrine_results.pt')
    velocity_labels = [x*0.03 + 0.50 for x in range(0,34)]
    
    fig, ax = plt.subplots()
    
    ax.plot(velocity_labels, data['right'], label=r'Right $F$')
    ax.plot(velocity_labels, data['left'], label=r'Left $F$')
    ax.plot(velocity_labels, data['total'], label=r'Both $F$\'s')
    
    ax.set_title('B', loc='left')
    ax.set_xlabel(r'$s$ speed')
    ax.set_ylabel('Testset Accuracy')
    ax.legend(loc=(0.05,0.1))
    plt.show()
        
    return None

def AmacrineDendrites():
    model = torch.load('models_and_data/model.pt', map_location=torch.device('cpu'))
    bipolar_index = [1,2,4,6,7]
    l = len(bipolar_index)
    data = torch.zeros((l*8,5,5))
    count = 0
    
    for x in model:
        if 'amacrine_space' in x and 'ganglion_amacrine' not in x and 'bias' not in x:
            data[count*l:count*l+l] = model[x][0,bipolar_index,0,:,:] + model[x[:-6]+'bias']
            count += 1
    
    data = nn.functional.relu(data).numpy()
    mean = np.nanmean(data, axis=(0))
    
    fig, ax = plt.subplots()
    
    im = ax.imshow(mean, cmap='Reds')
    plt.colorbar(im, ax=ax)
    plt.show()
    
    return None
    
def figure6():
    data1 = torch.load('models_and_data/left_and_right_accuracies.pt', 
                      map_location=torch.device('cpu'))
    
    y = np.arange(len(data1['right']))
    width = 0.35
    labels = ['Full', 'Ablated', 'Bipolar', 'Amacrine']
    
    data2 = torch.load('models_and_data/ablated_amacrine_results.pt')
    velocity_labels = [x*0.03 + 0.50 for x in range(0,34)]
    
    model = torch.load('models_and_data/model.pt', map_location=torch.device('cpu'))
    bipolar_index = [1,2,4,6,7]
    l = len(bipolar_index)
    data3 = torch.zeros((l*8,5,5))
    count = 0
    
    for x in model:
        if 'amacrine_space' in x and 'ganglion_amacrine' not in x and 'bias' not in x:
            data3[count*l:count*l+l] = model[x][0,bipolar_index,0,:,:] + model[x[:-6]+'bias']
            count += 1
    
    data3 = nn.functional.relu(data3).numpy()
    mean = np.nanmean(data3, axis=(0))
    
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(20, 4), gridspec_kw={'width_ratios': [1, 1, 1]})
    a0.bar(y - width/2, data1['left'], width, label=r'Left $F$')
    a0.bar(y + width/2, data1['right'], width, label=r'Right $F$')
    a0.set_ylabel('Testset Accuracy')
    a0.set_xlabel('Model')
    a0.set_title('A', loc='left')
    a0.set_xticks(y)
    a0.set_xticklabels(labels)
    a0.legend()
    
    a1.plot(velocity_labels, data2['right'], label=r'Right $F$')
    a1.plot(velocity_labels, data2['left'], label=r'Left $F$')
    a1.plot(velocity_labels, data2['total'], label=r'Both')
    a1.set_title('B', loc='left')
    a1.set_xlabel(r'$s$ speed')
    a1.set_ylabel('Testset Accuracy')
    a1.legend(loc=(0.03,0.1))
    
    im = a2.imshow(mean, cmap='Reds')
    plt.colorbar(im, ax=a2, ticks=[0.00,0.02], label='Weight strength')
    a2.set_title('C', loc='left')
    a2.set_xticks([])
    a2.set_yticks([])
    a2.set_xlabel('Horrizontal connections')
    a2.set_ylabel('Vertical connections')
    
    plt.savefig('figures/jcb figs/figure 6.svg', format="svg")
    plt.show()

if __name__ == "__main__":
    figure3()
    figure4()
    figure5()
    figure6()

    