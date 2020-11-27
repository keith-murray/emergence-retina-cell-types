# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:19:56 2020

@author: Keith
"""
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import torch
dtype = torch.float
device = torch.device("cuda:0")


class moveObjs():
    def __init__(self, dims, direction, initial_pos, unique=False):
        '''
        Initialization.
        INPUT:
            - dims: the dimensions of the scene to determine the range of motion
            - initial_pos: the inital position of the object
            - unique: is this object the unique one?
        '''
        self.dims = dims
        self.direction = direction
        self.initial_x = initial_pos[0]
        self.initial_y = initial_pos[1]
        self.x_coords = [initial_pos[0],]
        self.y_coords = [initial_pos[1],]
        self.unique = unique
    
    def forwardMotion(self,):
        '''
        Create forward motion for the object one step at a time.
        '''
        if self.unique and (self.x_coords[-1] != None or self.y_coords[-1] != None):
            x_step = self.x_coords[-1] - self.direction[0]
            y_step = self.y_coords[-1] - self.direction[1]
        elif self.x_coords[-1] == None or self.y_coords[-1] == None:
            x_step = -1
            y_step = -1
        else:
            x_step = self.x_coords[-1] + self.direction[0]
            y_step = self.y_coords[-1] + self.direction[1]
        if x_step < 0 or y_step < 0 or x_step >= self.dims[0] or y_step >= self.dims[0]:
            self.x_coords.append(None)
            self.y_coords.append(None)
            return None
        else:
            self.x_coords.append(x_step)
            self.y_coords.append(y_step)
            return (x_step, y_step)
    
    def getHistory(self,):
        '''
        Get the coordinates of the entire position of the object.
        '''
        x_hist = [p for p in self.x_coords if p != None]
        y_hist = [p for p in self.y_coords if p != None]
        return [(x_hist[p], y_hist[p]) for p in range(len(x_hist))]


class scene():
    def __init__(self, pixels, frames, num_objects, num_unique, r_discrim=1):
        '''
        Initialization.
        INPUT:
            - pixels: total pixels to include in the square like scene
            - frames: total amount of frames to be created
            - num_objects: number of objects in the scene
            - num_unique: number of objects moving in the unique direction
            - r_discrim: the rate of discrimination in the unique direction
                - value between 0 and 1, usually assign 1
        '''
        self.dims = (pixels,pixels,frames)
        self.stack = np.zeros(self.dims)
        self.num_objects = num_objects
        self.num_unique = num_unique
        self.radius = int(frames/20)
        self.objs = []
        self.index_unique = np.random.choice(num_objects, num_unique, replace=False).tolist()
        
    def findPoints(self, point, fr):
        '''
        Given a point, return all the points in the circle around the point.
            ( x - h )^2 + ( y - k )^2 = r^2
        '''
        list_points = []
        for x in range(point[0]-self.radius-1,point[0]+self.radius+2):
            for y in range(point[1]-self.radius-1,point[1]+self.radius+2):
                if (x - point[0])**2 + (y - point[1])**2 <= self.radius**2 and not (x < 0 or y < 0 or x >= self.dims[0] or y >= self.dims[1]):
                    self.stack[x,y,fr] = 1

    def createObjs(self,):
        '''
        Create the objects for the scene.
        '''
        direc_dist = np.random.randint(-3, high=4, size=2)
        if direc_dist[0] == 0 and direc_dist[1] == 0:
            print('ERR')
            direction = (0,1)
        else:
            direction = (direc_dist[0],direc_dist[1])
        print(direction)
        xChoice = np.random.choice(self.dims[0], self.num_objects, replace=False)
        yChoice = np.random.choice(self.dims[0], self.num_objects, replace=False)
        initial_positions = [(xChoice[p], yChoice[p]) for p in range(len(xChoice))]
        for p in range(self.num_objects):
            self.findPoints(initial_positions[p], 0)
            if p in self.index_unique:
                self.objs.append(moveObjs(self.dims,direction,initial_positions[p], unique=True))
            else:
                self.objs.append(moveObjs(self.dims,direction,initial_positions[p]))
    
    def moveObjs(self,):
        '''
        Move all the objects to their final positions.
        '''
        for p in range(1,self.dims[-1]):
            for q in range(len(self.objs)):
                newPos = self.objs[q].forwardMotion()
                if newPos != None:
                    self.findPoints(newPos, p)
    
    def createScene(self,):
        '''
        Create the entire scene from start to finish.
        '''
        self.createObjs()
        self.moveObjs()
        return self.stack
    
    
if __name__ == "__main__":
    test = scene(150,30,60,1)
    testScene = test.createScene()
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(testScene[0,0,:])):
        plt.imshow(testScene[:,:,i], cmap='hot', interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('testScene.gif', writer = 'pillow', fps=5)

















