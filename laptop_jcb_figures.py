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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams["font.family"] = "arial"

def PsycometricFunction():
    model = np.load('../retina_model_models/psycho_alpha_full.npy')[10:]
    ablated = np.load('../retina_model_models/psycho_alpha_ablated.npy')[10:]
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(0,len(model))/10+1, model, label='Full Model')
    ax.plot(np.arange(0,len(model))/10+1, ablated, label='Ablated Model')
        
    ax.set_ylabel('Accuracy on Testset')
    ax.set_xlabel(r'Speed Ratio ($\alpha$)')
    ax.legend()
    
    plt.savefig('psychometric_fig.svg')
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
    
    plt.savefig('cell_type_ablation_fig.svg')
    plt.show()
    
    return None

def figure4():
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

def figure5():
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
    
    f, axs = plt.subplots(2, 4, figsize=(10, 5))
    
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

    cax = plt.axes([1, 0.125, 0.0225, 0.75])
    plt.colorbar(im, cax=cax)
    f.tight_layout()
    plt.show()
    
    return None

def BipolarSubstitute():
    f = open("models_and_data/bipolar_empha.json")
    model_results = json.loads(f.read())
    
    fig, ax = plt.subplots()
    for y in range(8):
        ax.plot([x*0.05+0.5 for x in range(0,51)], model_results[str(y)][2], label=str(y))
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$f$ speed')
    ax.set_ylabel('Probability of activation')
    plt.show()
    
    return None

if __name__ == "__main__":
    # PsycometricFunction()
    # ModelAblation()
    figure4()
    # DeepDream()
    # DifferentStages()
    figure5()
    BipolarSubstitute()
    
    