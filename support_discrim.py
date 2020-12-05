# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:19:56 2020

@author: Keith
"""
import numpy as np
from celluloid import Camera
import seaborn as sns
import matplotlib.pyplot as plt
import torch
dtype = torch.float
device = torch.device("cuda:0")


class moveObjs():
    def __init__(self, dims, angle, initial_pos, unique=False):
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
    
    def forwardMotion(self,):
        '''
        Create forward motion for the object one step at a time.
        '''
        if self.x_coords[-1] == None or self.y_coords[-1] == None:
            x_step = -1
            y_step = -1
        else:
            x_step = self.x_coords[-1] + np.cos(self.angle)
            y_step = self.y_coords[-1] + np.sin(self.angle)
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
    def __init__(self, pixels, frames, num_objects, perct_unique, direc_var, r_discrim):
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
        self.initial_ang = np.random.random_sample()*2*np.pi
        self.initial_un_ang = self.initial_ang - np.pi*r_discrim
        self.variance = direc_var*2*np.pi
        
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
            angleVariance = 2*self.variance*np.random.random_sample() - self.variance
            if p in self.index_unique:
                self.objs.append(moveObjs(self.dims, self.initial_un_ang + angleVariance, initial_positions[p], unique=True))
            else:
                self.objs.append(moveObjs(self.dims, self.initial_ang + angleVariance, initial_positions[p]))
    
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


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel() # + lambda*recreate


def environment(rounds, pixels, frames, num_objects, perct_unique, direc_var, r_discrim):
    '''
    Function environment creates the set of stimuli specified by the parameters given.
    INPUTS:
        rounds - number of stimuli to include
        pixels - number of pixels to include in the horrizontal and vertical direction
        frames - number of frames to include in each stimuli draw
        num_objects - number of objects to include in total
        perct_unique - percent of the objects that go in the nonstandard direction
        direc_var - variance in the direction from 0 to 1
        r_discrim - the amount of discrimination in the standard and nonstandard direction (0 - 1)
    '''
    stimuli = []
    cent_loc = []
    for i in range(rounds):
        scene1, cent_loc1 = scene(pixels, frames, num_objects, perct_unique, direc_var, r_discrim).createScene()
        stimuli.append(scene1)
        cent_loc.append(cent_loc1)
    # for i,e in enumerate(cent_loc):
    #     cent_loc[i] = e - 50*np.ones((len(e[:,0]), len(e[0,:])))
        
    envi = None
    for i, e in enumerate(stimuli):
        hold = np.moveaxis(e, -1, 0)
        hold = torch.from_numpy(hold).to(device).float()
        hold = torch.unsqueeze(hold,0)
        hold = torch.unsqueeze(hold,0)
        if envi is None:
            envi = hold
        else:
            envi = torch.cat((envi, hold),0)
    envi.requires_grad_(True)
    res = None
    for i, e in enumerate(cent_loc):
        hold = np.array(e)
        hold = torch.from_numpy(hold).to(device).float()
        hold = torch.unsqueeze(hold, 0)
        if res is None:
            res = hold
        else:
            res = torch.cat((res, hold),0)
    res.requires_grad_(True)
    return envi, res


def optimize_func(scence, cent_loc, net, num_iter):
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    x = []
    
    for i in range(num_iter):
        optimizer.zero_grad()
        output = net(scence)
        loss = mse(output, cent_loc)
        x.append(loss.item())
        loss.backward()
        optimizer.step()
    return x, loss, output, net


def createTest(net, pixels, frames, num_objects, perct_unique, direc_var, r_discrim):
    envi, res = environment(2, pixels, frames, num_objects, perct_unique, direc_var, r_discrim)
    loss_vals, loss, pred_py, net = optimize_func(envi, res, net, 1)
    return loss


def create_summary_apriori(ret):
    fig, axs = plt.subplots(2, 2)
    axs[1,0].plot(torch.squeeze(ret.temporal_conv.weight).detach().cpu().numpy())
    axs[1,1] = sns.heatmap(torch.squeeze(ret.space_conv.weight).detach().cpu().numpy())
    axs[0,0].plot(torch.squeeze(ret.amacrine_kernel.weight).detach().cpu().numpy())
    axs[0,1].plot(torch.squeeze(ret.ganglion_kernel.weight).detach().cpu().numpy())
    axs[1,0].set_title('Bipolar Time Kernel')
    axs[1,1].set_title('Bipolar Space Kernel')
    axs[0,0].set_title('Amacrine Time Kernel')
    axs[0,1].set_title('Ganglion Time Kernel')
    axs[0,0].xaxis.set_ticklabels([])
    axs[0,1].xaxis.set_ticklabels([])
    plt.savefig('Q:/Documents/TDS SuperUROP/kernels.png')


def make_kernel(alpha):
    ray = torch.arange(1,102,1,device=device).float()
    ker = alpha[0]*torch.cos(alpha[1]*torch.log(alpha[2]*ray+torch.abs(alpha[3])))+\
    alpha[4]*torch.ones(101,device=device).float()
    return torch.flip(ker, (0,)).detach().cpu().numpy()


def create_summary_prior(ret):
    fig, axs = plt.subplots(2, 2)
    axs[1,0].plot(make_kernel(ret.temporal_conv.alpha)) # temporal_conv
    axs[1,1] = sns.heatmap(torch.squeeze(ret.space_conv.weight).detach().cpu().numpy())
    axs[0,0].plot(make_kernel(ret.amacrine_kernel.alpha)) # amacrine_kernel
    axs[0,1].plot(make_kernel(ret.ganglion_kernel.alpha)) # ganglion_kernel
    axs[1,0].set_title('Bipolar Time Kernel')
    axs[1,1].set_title('Bipolar Space Kernel')
    axs[0,0].set_title('Amacrine Time Kernel')
    axs[0,1].set_title('Ganglion Time Kernel')
    axs[0,0].xaxis.set_ticklabels([])
    axs[0,1].xaxis.set_ticklabels([])
    plt.savefig('Q:/Documents/TDS SuperUROP/kernels.png')

    
if __name__ == "__main__":
    test = scene(250,200,240,0.05,0.05,0.75)
    testScene, another = test.createScene()
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(testScene[0,0,:])):
        plt.imshow(testScene[:,:,i], cmap='hot', interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('testScene.gif', writer = 'pillow', fps=30)
















