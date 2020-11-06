# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 03:42:45 2020

@author: Keith
"""

from celluloid import Camera
import matplotlib.pyplot as plt
from pipline_torch import envrionment

scene1, cent_loc1 = envrionment(20, 4, 0, 1, 2, 4)
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene1[0,0,:])):
    plt.imshow(scene1[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('scene1.gif', writer = 'pillow', fps=25)

scene2, cent_loc2 = envrionment(20, 3, 1, 1, 2, 4)
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene2[0,0,:])):
    plt.imshow(scene2[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('scene2.gif', writer = 'pillow', fps=25)

scene3, cent_loc3 = envrionment(20, 5, -1, 1, 2, 4)
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene3[0,0,:])):
    plt.imshow(scene3[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('scene3.gif', writer = 'pillow', fps=25)

scene4, cent_loc4 = envrionment(20, 2, -1, 3, 2, 4)
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene4[0,0,:])):
    plt.imshow(scene4[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('scene4.gif', writer = 'pillow', fps=25)

scene5, cent_loc5 = envrionment(20, 4, 1, 3, 2, 4)
fig = plt.figure()
camera = Camera(fig)
for i in range(len(scene5[0,0,:])):
    plt.imshow(scene5[:,:,i], cmap='hot', interpolation='nearest')
    camera.snap()
animation = camera.animate()
animation.save('scene5.gif', writer = 'pillow', fps=25)