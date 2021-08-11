# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 02:37:19 2021

@author: Keith
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from retina_model import Bipolar, Amacrine, Ganglion, AnalysisModel, TestModel

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
    
    plt.savefig('../retina_figs/psychometric_fig.svg')
    plt.show()
    
    return None

def ModelAblation():
    bipolar = np.load('../retina_model_models/bipolar_ablation.npy')
    amacrine = np.load('../retina_model_models/amacrine_ablation.npy')
    ganglion = np.load('../retina_model_models/ganglion_ablation.npy')
    
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
    
    plt.savefig('../retina_figs/cell_type_ablation_fig.svg')
    plt.show()
    
    return None

def figure4():
    model = np.load('../retina_model_models/psycho_alpha_full.npy')[10:]
    ablated = np.load('../retina_model_models/psycho_alpha_ablated.npy')[10:]
    
    bipolar = np.load('../retina_model_models/bipolar_ablation.npy')
    amacrine = np.load('../retina_model_models/amacrine_ablation.npy')
    ganglion = np.load('../retina_model_models/ganglion_ablation.npy')
    
    width = 0.9
    
    f, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 3]})
    
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

    plt.show()

if __name__ == "__main__":
    # PsycometricFunction()
    # ModelAblation()
    figure4()