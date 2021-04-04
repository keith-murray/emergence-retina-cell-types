# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:16:25 2020

@author: Keith
"""
import seaborn as sns
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import torch
dtype = torch.float
device = torch.device("cuda:0")


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel() # + lambda*recreate


def create_circ_gradient(radius, width, heigth):
    background = np.zeros((heigth,heigth))
    single_line = np.zeros((1,heigth))
    half = width/2
    count = 0
    for x in range(heigth):
        if count < half:
            single_line[0,x] = 1
            count += 1
        elif width-1 == count:
            count = 0
    count = 0
    for x in range(heigth):
        if count < half:
            background[x,:] = single_line
            count += 1
        elif width-1 == count:
            background[x,:] = np.roll(single_line,1)
            count = 0
        else:
            background[x,:] = np.roll(single_line,1)
            count += 1
    return background


def create_gradient(line_size, roll_factor, skip_factor, height):
    background = np.zeros((1,4001))
    met_lines = 0
    met_crit = line_size
    met_lin_crit = met_crit*2 + 1
    for x in range(len(background[0,:])):
        if met_lines < met_crit:
            pass
        elif met_lines >= met_crit and met_lines < met_lin_crit:
            background[0,x] = 1
        else:
            met_lines = -1
        met_lines += 1
    n_bck = np.zeros((height,4001))
    skip_num = 0
    for x in range(len(n_bck[:,0])):
        if x == 0:
            n_bck[x,:] = background
            skip_num += 1
        elif skip_num < skip_factor:
            skip_num += 1
            n_bck[x,:] = np.roll(n_bck[x-1,:],0)
        else:
            n_bck[x,:] = np.roll(n_bck[x-1,:],roll_factor)
            skip_num = 0
    
    white = False
    for x in range(1000,len(n_bck[0,:])):
        if n_bck[0,x] == 0:
            white = True
        if white and n_bck[0,x] == 1:
            res = n_bck[:,x:x+height]
            break
    return res


def create_slide_deck(height,n_o_g,lines,roll_factor,skip_factor):
    # n_o_g is number of gratings
    scene = np.zeros((height,height*n_o_g))
    for x in range(n_o_g):
        scene[:,int(x*height):int(x*height+height)] = create_gradient(lines,roll_factor,skip_factor,height)
        
    place = np.zeros((height,height,height*n_o_g-height-1))
    for x in range(len(place[0,0,:])):
        place[:,:,x] = scene[:,x:x+height]
    return place


def generate_center_location_row_lock(slide_leng):
    col_place = [50,]
    loc = 50
    cur_run = 0
    if np.random.random_sample() < 0.5:
        base = False
    else:
        base = True
    for x in range(1,slide_leng):
        col_place.append(50)
    row_place = [50,]
    loc = 50
    cur_run = 0
    if np.random.random_sample() < 0.5:
        base = False
    else:
        base = True
    for x in range(1,slide_leng):
        if loc == 4:
            base = True
            cur_run = 3
            loc += 1
            row_place.append(loc)
            continue
        elif loc == 96:
            base = False
            cur_run = 3
            loc -= 1
            row_place.append(loc)
            continue
        elif (loc <= 10 and not base) or (loc >= 90 and base):
            j = 1 - (cur_run + 3)/(cur_run + 4)
        else:
            j = (cur_run + 3)/(cur_run + 4)
        p = np.random.random_sample()
        if p <= j:
            if base:
                loc = loc + 1
            else:
                loc = loc - 1
            cur_run += 1
        else:
            if base:
                loc = loc - 1            
            else:
                loc = loc + 1
            base = not base
            cur_run = 0
        row_place.append(loc)
    place = np.zeros((2,slide_leng+50))
    for x in range(slide_leng+45):
        if x == 0:
            place[0,x] = row_place[0]
            r_p = 1
            place[1,x] = col_place[0]
            c_p = 0
        elif x%2 != 0:
            place[0,x] = row_place[r_p]
            place[1,x] = col_place[c_p]
            c_p += 1
        else:
            place[0,x] = row_place[r_p]
            place[1,x] = col_place[c_p]
            r_p += 1
    place[0,x+1] = row_place[r_p-1]
    place[1,x+1] = col_place[c_p]
    return place[:,:slide_leng]


def stimuli(nog, line1, roll1, skip1, circ_width, radius):
    '''
    height is the size of the stimulus frame
    ###########################
    nog is number of gradients
    line1 is the line size
    roll1 is the horrizontal roll factor
    skip1 is the vertical row skip factor
    circ_width is the width of the circle checker pattern
    radius is the radius of the circle
    '''
    height = 100
    center_h = (height - 1)/2
    scence = create_slide_deck(height,nog,line1,roll1,skip1)
    circle = create_circ_gradient(radius, circ_width, height)
    slide_len = height*nog-height-1
    cent_loc = generate_center_location_row_lock(slide_len)
    ct_loc = [[0.0,0.0],]
    for x in range(1,len(cent_loc[0,:])):
        if cent_loc[0,x]-cent_loc[0,x-1] > 0:
            ct_loc.append([0.0,0.0])
        else:
            ct_loc.append([0.0,1.0])
    for z in range(len(scence[0,0,:])):
        new_ccp = cent_loc[0,z]
        new_rrp = cent_loc[1,z]
        for x in range(len(scence[0,:,0])):
            for y in range(len(scence[:,0,0])):
                x_column_j = x-new_ccp
                y_row_j = y-new_rrp
                if radius**2 >= (x_column_j)**2 + (y_row_j)**2:
                    scence[y,x,z] = circle[int(center_h+y_row_j), int(center_h+x_column_j)]
    return scence, ct_loc


def environment(samples,leng):
    '''
    1 - 6
    -3 - 3
    0 - 4
    1 - 2
    3 - 5
    '''
    scence = []
    cent_loc = []
    for i in range(samples):
        scene1, cent_loc1 = stimuli(leng, np.random.randint(1,6), np.random.randint(-2,2), np.random.randint(1,3), np.random.randint(1,2), 4)
        scence.append(scene1)
        cent_loc.append(cent_loc1)
    # for i,e in enumerate(cent_loc):
    #     cent_loc[i] = e - 50*np.ones((len(e[:,0]), len(e[0,:])))
        
    scene = None
    for i, e in enumerate(scence):
        hold = np.moveaxis(e, -1, 0)
        hold = torch.from_numpy(hold).to(device).float()
        hold = torch.unsqueeze(hold,0)
        hold = torch.unsqueeze(hold,0)
        if scene is None:
            scene = hold
        else:
            scene = torch.cat((scene, hold),0)
    scene.requires_grad_(True)
    cent = None
    for i, e in enumerate(cent_loc):
        # hold = np.moveaxis(e, -1, 0)
        hold = np.array(e)
        hold = torch.from_numpy(hold).to(device).float()
        hold = torch.unsqueeze(hold, 0)
        if cent is None:
            cent = hold
        else:
            cent = torch.cat((cent,hold),0)
    cent.requires_grad_(True)
    return scene, cent


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


def environment_to_gpu(scence, cent_loc):
    scene = None
    hold = np.moveaxis(scence, -1, 0)
    hold = torch.from_numpy(hold).to(device).float()
    hold = torch.unsqueeze(hold,0)
    hold = torch.unsqueeze(hold,0)
    if scene is None:
        scene = hold
    else:
        scene = torch.cat((scene, hold),0)
    scene.requires_grad_(True)
    cent = None
    hold = np.moveaxis(cent_loc, -1, 0)
    hold = torch.from_numpy(hold).to(device).float()
    hold = torch.unsqueeze(hold, 0)
    if cent is None:
        cent = hold
    else:
        cent = torch.cat((cent,hold),0)
    cent.requires_grad_(True)
    return scene, cent


def make_test(net, scenef, cent_locf):
    scenef, cent_locff = environment_to_gpu(scenef, cent_locf)
    output = net(scenef)
    # lossf = mse(output, cent_locff)
    pred_pyf = output.cpu().detach().numpy()
    pred_pyf = np.moveaxis(pred_pyf, -1, 1)    
    cent_locf = np.array(cent_locf)
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(cent_locf[:,0])):
        plt.scatter(cent_locf[i,0],cent_locf[i,1],c='b')
        plt.scatter(pred_pyf[i,0],pred_pyf[i,1],c='r')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/track_torch_prediction.gif', writer = 'pillow', fps=25)
    # return lossf
     

def make_output_test_video(pred_py, cent_loc):
    pred_py = pred_py.cpu().detach().numpy()
    cent_loc = cent_loc.cpu().detach().numpy()
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(cent_loc[0,:,0])):
        plt.xlim(-10,10)
        plt.scatter(cent_loc[-1,i,0],cent_loc[-1,i,1],c='b')
        plt.scatter(pred_py[-1,i,0],pred_py[-1,i,1],c='r')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/track_torch.gif', writer = 'pillow', fps=25)
    

if __name__ == "__main__":
    sc, cent = stimuli(20, 5, 1, 4, 2, 10)
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(sc[0,0,:])):
        plt.imshow(sc[:,:,i], cmap='Greys',  interpolation='nearest')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/test.gif', writer = 'pillow', fps=40)
























