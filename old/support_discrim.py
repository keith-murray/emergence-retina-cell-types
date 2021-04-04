# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:19:56 2020

@author: Keith
"""
import numpy as np
from celluloid import Camera
import os
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
    def __init__(self, pixels, frames, num_objects, perct_unique, direc_var, r_discrim, initAngle=None):
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
        if initAngle != None:
            self.initial_ang = initAngle
        else:
            self.initial_ang = np.random.random_sample()*2*np.pi
        self.initial_un_ang = self.initial_ang - np.pi*r_discrim
        self.variance = direc_var*np.pi
        
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


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_name, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.file_name = file_name

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        dirr = 'Q:\Documents\TDS SuperUROP\\'+self.file_name + os.sep + str(ID)

        # Load data and get label
        X = torch.load(dirr+'\stimulus.pt')
        y = torch.load(dirr+'\label.pt')
        return X, y


def mse(diff):
    return -1*(torch.sum(diff * diff) / diff.numel()) # + lambda*recreate


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


def environment_5_channels(rounds, pixels, frames, num_objects, perct_unique, direc_var, r_discrim, angleUse=None):
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
    cent_loc1 = None
    for i in range(rounds):
        scene1, cent_loc1 = scene(pixels, frames, num_objects, perct_unique, direc_var, r_discrim, initAngle=angleUse).createScene()
        stimuli.append(scene1)
        cent_loc.append(cent_loc1)

    envi = None
    for i, e in enumerate(stimuli):
        hold = np.moveaxis(e, -1, 0)
        hold = torch.from_numpy(hold).float()
        hold_x = torch.unsqueeze(hold,0)
        for x in range(5):
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
    for i, e in enumerate(cent_loc):
        hold = np.array(e)
        hold = torch.from_numpy(hold).float()
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
    cos = torch.nn.CosineSimilarity(dim=0)
    
    for i in range(num_iter):
        optimizer.zero_grad()
        output = net(scence)
        new_out =  torch.stack((torch.cos(output),torch.sin(output)))
        new_cent_loc =  torch.stack((torch.cos(cent_loc),torch.sin(cent_loc)))
        loss = mse(cos(new_out, new_cent_loc))
        x.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
    return x, loss, output, net


def testFunc(scence, cent_loc, net):
    net.eval()
    cos = torch.nn.CosineSimilarity(dim=0)
    output = net(scence)
    new_out =  torch.stack((torch.cos(output),torch.sin(output)))
    new_cent_loc =  torch.stack((torch.cos(cent_loc),torch.sin(cent_loc)))
    loss = mse(cos(new_out, new_cent_loc))
    x = loss.item()
    return x


def createTest(net, pixels, frames, num_objects, perct_unique, direc_var, r_discrim):
    envi, res = environment(2, pixels, frames, num_objects, perct_unique, direc_var, r_discrim)
    loss_vals, loss, pred_py, net = testFunc(envi, res, net)
    return loss


def createTest5Channel(net, pixels, frames, num_objects, perct_unique, direc_var, r_discrim):
    envi, res = environment_5_channels(2, pixels, frames, num_objects, perct_unique, direc_var, r_discrim)
    loss_vals, loss, pred_py, net = testFunc(envi.to(device), res.to(device), net)
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
    ray = torch.arange(1,42,1,device=device).float()
    ker = alpha[0]*torch.cos(alpha[1]*torch.log(alpha[2]*ray+torch.abs(alpha[3])))+\
    alpha[4]*torch.ones(41,device=device).float()
    return torch.flip(ker, (0,)).detach().cpu().numpy()


def make_matrix(alpha):
    x_t_1 = alpha[0]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float).float()/alpha[1])**2)
    x_t_2 = alpha[2]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float).float()/alpha[3])**2)
    y_t_1 = alpha[4]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float).float()/alpha[5])**2)
    y_t_2 = alpha[6]*torch.exp(-0.5*(torch.arange(-2,3,device=device,dtype=torch.float).float()/alpha[7])**2)
    x_t = x_t_1 - x_t_2
    y_t = y_t_1 - y_t_2
    base = torch.squeeze(torch.mul(torch.unsqueeze(x_t,-1),torch.unsqueeze(y_t,0))).detach().cpu().numpy()
    return base


def create_summary_prior(ret):
    fig, axs = plt.subplots(2, 2)
    axs[1,0].plot(make_kernel(ret.temporal_conv.alpha)) # temporal_conv
    axs[1,1] = sns.heatmap(make_matrix(ret.space_conv.alpha))
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
    save_location = 'Q:/Documents/TDS SuperUROP/75_0_50_testset'
    os.mkdir(save_location)
    save_location = save_location + os.sep
    itera = 100
    for x in range(itera):
        angl = 2*np.pi*x/itera
        envi, res = environment_5_channels(1,255,30,200,0.75,0,0.5, angleUse=angl)
        
        os.mkdir(save_location+str(x))
        torch.save(envi, save_location+str(x)+'\stimulus.pt')
        torch.save(res, save_location+str(x)+'\label.pt')
    
    # testScene, another = test.createScene()
    # fig = plt.figure()
    # camera = Camera(fig)
    # for i in range(len(testScene[0,0,:])):
    #     plt.imshow(testScene[:,:,i], cmap='hot', interpolation='nearest')
    #     camera.snap()
    # animation = camera.animate()
    # animation.save('testScene.gif', writer = 'pillow', fps=10)
















