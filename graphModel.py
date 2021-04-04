# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:31 2020

@author: Keith
"""
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



if __name__ == "__main__":
    modelName = 'model_binary_50_2_no_drop'
    model = graph_model(modelName)
    
    
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(np.ndarray.flatten(perct_mat), np.ndarray.flatten(rate_mat), np.ndarray.flatten(perct_rate), cmap='viridis')
    # ax.view_init(40,20)
    # ax.set_xlabel('Percent Discrimination')
    # ax.set_ylabel('Rate Difference')
    # ax.set_zlabel('Task Accuracy')
    # plt.show()
    
    # plt.plot(results[0,0,:].cpu().detach().numpy())
    # plt.plot(results[0,1,:].cpu().detach().numpy())