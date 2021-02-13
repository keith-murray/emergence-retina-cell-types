# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 03:42:45 2020

@author: Keith
"""

from celluloid import Camera
import matplotlib.pyplot as plt
from support_discrim import scene

scene1, cent_loc1 = scene(250,30,120,1,0,1).createScene()
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene1[0,0,:])):
    plt.imshow(scene1[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('one_dricetion.gif', writer = 'pillow', fps=25)
