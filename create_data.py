# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 00:43:33 2021

@author: Keith
"""

import numpy as np
from celluloid import Camera
import os
import matplotlib.pyplot as plt
import torch
dtype = torch.float
device = torch.device("cuda:0")


def reConfigure(data, label, devices):
    return torch.unsqueeze(torch.squeeze(data.to(devices)), dim=1), torch.squeeze(label.to(devices))


def tens_np(tens):
    return tens.cpu().detach().numpy()


class moveObjs():
    def __init__(self, dims, angle, initial_pos, speed, control, unique=False):
        '''
        Initialization.
        INPUT:
            - dims: the dimensions of the scene to determine the range of motion
            - initial_pos: the inital position of the object
            - unique: is this object the unique one?
        '''
        self.dims = dims
        self.angle = angle
        self.initial_x = initial_pos[0]
        self.initial_y = initial_pos[1]
        self.x_coords = [initial_pos[0],]
        self.y_coords = [initial_pos[1],]
        self.unique = unique
        self.speed = speed
        self.control = control
    
    def forwardMotion(self,):
        '''
        Create forward motion for the object one step at a time.
        '''
        if self.x_coords[-1] == None or self.y_coords[-1] == None:
            x_step = -1
            y_step = -1
        else:
            if self.unique:
                x_step = self.x_coords[-1] + self.speed*self.control*np.cos(self.angle)
                y_step = self.y_coords[-1] + self.speed*self.control*np.sin(self.angle)
            else:
                x_step = self.x_coords[-1] + self.control*np.cos(self.angle)
                y_step = self.y_coords[-1] + self.control*np.sin(self.angle)
        if x_step < 0 or y_step < 0 or x_step >= self.dims[0] or y_step >= self.dims[0]:
            self.x_coords.append(None)
            self.y_coords.append(None)
            return None
        else:
            self.x_coords.append(x_step)
            self.y_coords.append(y_step)
            return (int(x_step), int(y_step))
    
    def getHistory(self,):
        '''
        Get the coordinates of the entire position of the object.
        '''
        x_hist = [p for p in self.x_coords if p != None]
        y_hist = [p for p in self.y_coords if p != None]
        return [(x_hist[p], y_hist[p]) for p in range(len(x_hist))]


class scene():
    def __init__(self, pixels, frames, num_objects, perct_unique, direc_var, r_discrim, speed_diff, initAngle=None, slow_vel=None):
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
        self.dims = (pixels*3,pixels*3,frames)
        self.pixels = pixels
        self.stack = np.zeros(self.dims)
        self.num_objects = num_objects
        self.num_unique = int(num_objects*perct_unique)
        self.radius = int(pixels/50)
        self.objs = []
        self.index_unique = np.random.choice(num_objects, self.num_unique, replace=False).tolist()
        self.speed = speed_diff
        if initAngle is not None:
            self.initial_ang = initAngle
        else:
            self.initial_ang = np.random.random_sample()*2*np.pi
        self.initial_un_ang = self.initial_ang - np.pi*r_discrim
        self.variance = direc_var*np.pi
        if slow_vel is not None:
            self.control_speed = slow_vel
        else:
            self.control_speed = np.random.random_sample() + 0.5
        
    def findPoints(self, point, fr):
        '''
        Given a point, return all the points in the circle around the point.
            ( x - h )^2 + ( y - k )^2 = r^2 
        '''
        for x in range(point[0]-self.radius-1,point[0]+self.radius+2):
            for y in range(point[1]-self.radius-1,point[1]+self.radius+2):
                if (x - point[0])**2 + (y - point[1])**2 <= self.radius**2 and not (x < 0 or y < 0 or x >= self.dims[0] or y >= self.dims[1]):
                    self.stack[x,y,fr] = 1

    def createObjs(self,):
        '''
        Create the objects for the scene.
        '''
        xChoice = np.random.choice(self.dims[0], self.num_objects, replace=False)
        yChoice = np.random.choice(self.dims[0], self.num_objects, replace=False)
        initial_positions = [(xChoice[p], yChoice[p]) for p in range(len(xChoice))]
        for p in range(self.num_objects):
            self.findPoints(initial_positions[p], 0)
            angleVariance = 2*self.variance*(2*np.random.random_sample()-1) - self.variance
            if p in self.index_unique:
                self.objs.append(moveObjs(self.dims, self.initial_un_ang + angleVariance, initial_positions[p], self.speed, self.control_speed, unique=True))
            else:
                self.objs.append(moveObjs(self.dims, self.initial_ang + angleVariance, initial_positions[p], self.speed, self.control_speed))
    
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
        return self.stack[self.pixels:2*self.pixels,self.pixels:2*self.pixels,:], self.initial_un_ang


def environment(pixels, frames, num_objects, perct_unique, speed_diff, right=True, slow_speed=None):
    '''
    Function environment creates the set of stimuli specified by the parameters given.
    INPUTS:
        rounds - number of stimuli to include
        pixels - number of pixels to include in the horrizontal and vertical direction
        frames - number of frames to include in each stimuli draw
        num_objects - number of objects to include in total
        perct_unique - percent of the objects that go in the nonstandard direction
        speed_diff - the magintude in speed difference between the two
    '''
    direc_var = 0
    r_discrim = 1
    if right:
        angleUse=3*np.pi/2
    else:
        angleUse=1*np.pi/2
    stimuli = []
    cent_loc = []
    cent_loc1 = None
    scene1, cent_loc1 = scene(pixels, frames, num_objects, perct_unique, direc_var, 
                              r_discrim, speed_diff, initAngle=angleUse, slow_vel=slow_speed).createScene()
    stimuli.append(scene1)
    cent_loc.append(cent_loc1)

    envi = None
    for i, e in enumerate(stimuli):
        hold = np.moveaxis(e, -1, 0)
        hold = torch.from_numpy(hold).float()
        hold_x = torch.unsqueeze(hold,0)
        for x in range(1):
            if x == 0:
                hold = hold_x
            else:
                hold = torch.cat((hold, hold_x),0)
        hold = torch.unsqueeze(hold,0)
        if envi is None:
            envi = hold
        else:
            envi = torch.cat((envi, hold),0)
    envi.requires_grad_(True)
    
    res = None
    if right:
        vec = np.array([0.0,1.0])
    else:
        vec = np.array([1.0,0.0])
    res = torch.from_numpy(vec)
    res.requires_grad_(True)
    
    return envi, res


def createSet(save_location,itera,dotDiscrim,speedDiscrim,slow_vel=None):
    '''
    Parameters
    ----------
    save_location : TYPE
        save_location = 'Q:/Documents/TDS SuperUROP/binary_50_1_dataset'
    itera : TYPE
        itera = 250
    dotDiscrim : TYPE
        dotDiscrim = .75
    speedDiscrim : TYPE
        speedDiscrim = 3

    Returns
    -------
    None.
    '''
    os.mkdir(save_location)
    left_right_ind = np.random.randint(0, high=2, size=itera, dtype=int)
    for x in range(itera):
        if left_right_ind[x] == 1:
            left_right = True
        else:
            left_right = False
        stim, res = environment(255,51,150,dotDiscrim,speedDiscrim, right=left_right, slow_speed=slow_vel)
        
        os.mkdir(save_location+str(x))
        torch.save(stim, save_location+str(x)+'\stimulus.pt')
        torch.save(res, save_location+str(x)+'\label.pt')


def createSet_definedDist(save_location,itera,speedDiscrim,slow_vel):
    '''
    Parameters
    ----------
    save_location : TYPE
        save_location = 'Q:/Documents/TDS SuperUROP/binary_50_1_dataset'
    itera : TYPE
        itera = 250
    dotDiscrim : TYPE
        dotDiscrim = .75
    speedDiscrim : TYPE
        speedDiscrim = 3

    Returns
    -------
    None.
    '''
    os.mkdir(save_location)
    save_location = save_location + os.sep
    left_right_ind = np.random.randint(0, high=2, size=itera, dtype=int)
    for x in range(itera):
        if left_right_ind[x] == 1:
            left_right = True
        else:
            left_right = False
        choosen_velocity = slow_vel()
        stim, res = environment(255,51,150,0.5,speedDiscrim, right=left_right, slow_speed=choosen_velocity)
        
        os.mkdir(save_location+str(x))
        torch.save(stim, save_location+str(x)+'\stimulus.pt')
        torch.save(res, save_location+str(x)+'\label.pt')


def createDataTest(fileName,speedDiscrim,dataNum,testNum,slow_speed=None):
    '''
    Parameters
    ----------
    fileName : 'binary_50_1'
        DESCRIPTION.
    dotDiscrim : 0.75
        DESCRIPTION.
    speedDiscrim : 3
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if dataNum > 0:
        createSet('dataset_'+fileName,dataNum,0.5,speedDiscrim,slow_vel=slow_speed)
    if testNum > 0:
        createSet('testset_'+fileName,testNum,0.5,speedDiscrim,slow_vel=slow_speed)


def createSet_definedDist_direction(save_location,slow_vel):
    '''
    Parameters
    ----------
    save_location : TYPE
        save_location = 'Q:/Documents/TDS SuperUROP/binary_50_1_dataset'
    itera : TYPE
        itera = 250
    dotDiscrim : TYPE
        dotDiscrim = .75
    speedDiscrim : TYPE
        speedDiscrim = 3

    Returns
    -------
    None.
    '''
    os.mkdir(save_location)
    save_location = save_location + os.sep
    # for x in range(15):
    #     left_right = True
    #     stim, res = environment(255,51,150,1,1, right=left_right, slow_speed=slow_vel)
        
    #     os.mkdir(save_location+str(x))
    #     torch.save(stim, save_location+str(x)+'\stimulus.pt')
    #     torch.save(res, save_location+str(x)+'\label.pt')

    for x in range(29,30):
        left_right = False
        stim, res = environment(255,51,150,1,1, right=left_right, slow_speed=slow_vel)
        
        os.mkdir(save_location+str(x))
        torch.save(stim, save_location+str(x)+'\stimulus.pt')
        torch.save(res, save_location+str(x)+'\label.pt')


def Testsets1x():
    for x in range(0,51):
        speed = x*0.05 + 0.50
        createSet_definedDist_direction('1x_'+str(x)+'_dist', speed)
        print(speed)
    return None


def TestsetVelocityDist():
    for x in range(0,34):
        speed = 0.50 + x*.03
        createDataTest('2x_'+str(x)+'_dist', 2, 0, 100, slow_speed=speed)
        print(speed)
    return None


def plotStimulus(testSet, name):
    '''
    testSet = 'binary_75_3_testset'
    '''
    fig = plt.figure()
    camera = Camera(fig)
    test = testSet.cpu().detach().numpy()[0,0,:,:,:]
    for i in range(test.shape[0]):
        plt.imshow(test[i,:,:], cmap='hot', vmin=0.0, vmax=1.0)
        camera.snap()
    animation = camera.animate()
    animation.save(name+'.gif', writer = 'pillow', fps=15)
    return


if __name__ == "__main__":
    createDataTest('2x_speed',2,1000,100)
    print('Done')
    
    
    