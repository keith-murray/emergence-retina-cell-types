# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:31 2020

@author: Keith
"""
from celluloid import Camera
import os
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0")


def plotStimulus(testSet):
    '''
    testSet = 'binary_75_3_testset'
    '''
    # Select sample
    dirr = 'Q:\Documents\TDS SuperUROP\\' + testSet + os.sep + str(1)

    # Load data and get label
    stim = torch.load(dirr+'\stimulus.pt')

    fig = plt.figure()
    camera = Camera(fig)
    test = stim.cpu().detach().numpy()[0,0,:,:,:]
    for i in range(test.shape[0]):
        plt.imshow(test[i,:,:], cmap='hot', interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('testScene.gif', writer = 'pillow', fps=30)
    return

if __name__ == "__main__":
    plotStimulus('binary_67_1p5_testset')
    