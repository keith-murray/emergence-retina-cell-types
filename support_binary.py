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
import shutil
dtype = torch.float
device = torch.device("cuda:0")


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
    def __init__(self, pixels, frames, num_objects, perct_unique, direc_var, r_discrim, speed_diff, initAngle=None):
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
        if initAngle != None:
            self.initial_ang = initAngle
        else:
            self.initial_ang = np.random.random_sample()*2*np.pi
        self.initial_un_ang = self.initial_ang - np.pi*r_discrim
        self.variance = direc_var*np.pi
        self.control_speed = np.random.random_sample()/2 + 0.3
        
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


def environment_1_channels(pixels, frames, num_objects, perct_unique, speed_diff, right=True):
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
    scene1, cent_loc1 = scene(pixels, frames, num_objects, perct_unique, direc_var, r_discrim, speed_diff, initAngle=angleUse).createScene()
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


def optimize_func(scence, cent_loc, net, num_iter):
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.01)
    x = []
    loss = torch.nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        optimizer.zero_grad()
        estimation = net(scence)
        output = loss(estimation, cent_loc[:,1].long()) # Change this?
        x.append(output.item())
        output.backward(retain_graph=True)
        optimizer.step()
    return x, output, net


def testFunc(scence, cent_loc, net):
    loss = torch.nn.CrossEntropyLoss()
    estimation = net(scence)
    output = loss(estimation, cent_loc[:,0].long())
    x = output.item()
    return x


def evalFunc(scence, cent_loc, net):
    estimation = net(scence)
    accuracy = []
    for i in range(scence.shape[0]):
        if estimation[i][0] < estimation[i][1]:
            if cent_loc[i][0] < cent_loc[i][1]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        else:
            if cent_loc[i][0] > cent_loc[i][1]:
                accuracy.append(1)
            else:
                accuracy.append(0)
    return sum(accuracy)/len(accuracy)


def reConfigure(envi, res, devices):
    return torch.unsqueeze(torch.squeeze(envi.to(devices)), dim=1), torch.squeeze(res.to(devices))



def plotStimulus(testSet):
    '''
    testSet = 'binary_75_3_testset'
    '''
    fig = plt.figure()
    camera = Camera(fig)
    test = testSet.cpu().detach().numpy()[0,0,:,:,:]
    for i in range(test.shape[0]):
        plt.imshow(test[i,:,:], cmap='hot', interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/testScene.gif', writer = 'pillow', fps=30)
    return


def createSet(save_location,itera,dotDiscrim,speedDiscrim):
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
        stim, res = environment_1_channels(255,51,100,dotDiscrim,speedDiscrim, right=left_right)
        
        os.mkdir(save_location+str(x))
        torch.save(stim, save_location+str(x)+'\stimulus.pt')
        torch.save(res, save_location+str(x)+'\label.pt')

def createDataTest(fileName,dotDiscrim,speedDiscrim,dataNum,testNum):
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
    save_location = 'Q:/Documents/TDS SuperUROP' + os.sep
    createSet(save_location+'dataset_'+fileName,dataNum,dotDiscrim,speedDiscrim)
    createSet(save_location+'testset_'+fileName,testNum,dotDiscrim,speedDiscrim)


def test_model(modelName, testSet):
    '''
    modelName = 'binary_75_3_3333'
    testSet = 'binary_75_3_testset'
    '''
    net = Retinal_NET_binary().to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP' + os.sep + modelName + os.sep + 'model.pt'
    net.load_state_dict(torch.load(save_loc))
    
    # Parameters

    test_params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 0}
    
    # Datasets
    test_labels = range(100)
    
    # Generators
    
    test_set = Dataset(testSet,test_labels)
    test_generator = torch.utils.data.DataLoader(test_set, **test_params)
        
    # Testing
    net.eval()
    testStore = []
    testRun = 0
    for local_batch, local_labels in test_generator:
        # Transfer to GPU
        local_batch, local_labels = reConfigure(local_batch, local_labels, device)
        loss = evalFunc(local_batch, local_labels, net)
        results = net.outputCell(local_batch)
        testStore.append(loss)
        testRun += loss
        break
    
    ans = abs(testRun/len(testStore))
    print(modelName + '--' + testSet + '--Accuracy: ' + str(ans))
    return results


def testModelFully(file):
    '''

    Parameters
    ----------
    file : TYPE
            file = 'model_binary_50_2_no_drop'

    Returns
    -------
    None.

    '''
    try:
        shutil.rmtree('Q:/Documents/TDS SuperUROP/testset_test')
    except:
        print('No Testset')
    try:
        shutil.rmtree('Q:/Documents/TDS SuperUROP/dataset_test')
    except:
        print('No Dataset')
    perct = 0.50
    rate = [x*0.1 for x in range(30)]
    perct_rate = np.zeros((len(rate),1))
    for r in range(len(rate)):
        print(str(perct) + ' :: ' + str(rate[r]))
        createDataTest('test',perct,rate[r])
        perct_rate[r,0] = test_model(file,'testset_test')
        shutil.rmtree('Q:/Documents/TDS SuperUROP/testset_test')
        shutil.rmtree('Q:/Documents/TDS SuperUROP/dataset_test')


def tens_np(tens):
    return tens.cpu().detach().numpy()


def graph_model(model):
    '''
    modelName = 'binary_75_3_3333'
    '''
    for i in range(5):
        fig, axs = plt.subplots(3, 2)
        sns.heatmap(tens_np(model['bipolar_space.weight'][i,0,0,:,:]), ax=axs[0, 0])
        axs[0, 0].set_ylabel('Bipolar')
        axs[0, 0].set_title('Spacial Kernel')
        axs[0, 1].plot(np.flip(tens_np(model['bipolar_temporal.weight'][i,0,:,0,0])))
        axs[0, 1].set_title('Temporal Kernel')
        sns.heatmap(tens_np(model['amacrine_space.weight'][i,0,0,:,:]), ax=axs[1, 0])
        axs[1, 0].set_ylabel('Amacrine')
        axs[1, 1].plot(np.flip(tens_np(model['amacrine_temporal.weight'][i,0,:,0,0])))
        sns.heatmap(tens_np(model['ganglions_space.weight'][i,0,0,:,:]), ax=axs[2, 0])
        axs[2, 0].set_ylabel('Ganglion')
        axs[2, 1].plot(np.flip(tens_np(model['ganglion_temporal.weight'][i,0,:,0,0])))
        for a in axs:
            for b in a:
                # b.get_yaxis().set_ticks([])
                if b != axs[2,1]:
                    b.get_xaxis().set_ticks([])
                else:
                    axs[2,0].set_xlabel('Channel # ' + str(i+1))
        plt.show()
        plt.clf()
    
    return model


if __name__ == "__main__":
    createDataTest('binary_2_sparse_slow',0.50,2,2,100)

# 2/21/2021 1:19 PM ::: stim, res = environment_5_channels(255,50,250,0.75,3, right=left_right)

# 2/25/2021 3:00 PM ::: stim, res = environment_5_channels(255,50,250,0.67,1.5, right=left_right)

# 2/27/2021 12:02 AM ::: createDataTest('binary_50_1',0.50,1) #Note: 5 stimuli setting

# 3/3/2021 8:30 AM ::: createDataTest('binary_50_1',0.75,3)










