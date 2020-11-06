# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 03:41:46 2020

@author: Keith
"""

import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import math
import torch
dtype = torch.float
device = torch.device("cuda:0")

pi = math.pi
factorial = lambda x : math.factorial(x)

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


def nonlin_bi(sig):
    res = np.zeros(len(sig))
    for x in range(len(sig)):
        val = sig[x]
        if val>= 0:
            res[x] = val
        else:
            res[x] = 0
    return res


def mexican_hat(x,y,sig):
    ex_com = np.exp(-(x**2+y**2)/(2*sig**2))
    middle_com = 1-0.5*((x**2+y**2)/(sig**2))
    return middle_com*ex_com/(pi*sig**4)


def mat_conv(backgrnd, ker):
    out_put_1 = np.zeros((10,10))
    for y in range(10):
        for x in range(10):
            int_res = backgrnd[x*4:x*4+5,y*4:y*4+5]@ker
            out_put_1[x,y] = np.sum(int_res)
    return out_put_1


def nonlin(sig,g,h,b):
    res = np.zeros(len(sig))
    for x in range(len(sig)):
        val = sig[x]
        res[x] = b/(1+np.exp(g*(val+h)))
    return res


def spike_gen(sig, width):
    spike = np.zeros(len(sig))
    for x in range(len(sig)):
        if x < width:
            y_sp = np.sum(spike[:x])
        else:
            y_sp = np.sum(spike[x-width:x])
        p = np.exp(-width*sig[x])*(width*sig[x])**(y_sp)/factorial(y_sp)
        propp = np.random.uniform(0,1.0)
        if propp < p:
            spike[x] = 1
    return spike


def create_rgc_spike(paramm, sig_g, sig_a):
    [a,b,c,d,e,f,g,h,i,j,k,l,m,n] = paramm
    base = np.linspace(.5,4.5,num=15,endpoint=False)
    
    temp_ker = np.zeros(15)
    for x in range(len(base)):
        temp_ker[x] = a*np.cos(b*np.log(base[x]))+c*np.cos(d*np.log(base[x]))+e
    temp_ker = temp_ker/np.trapz(temp_ker)
    a_nonlin_res_s = np.zeros(len(sig_a))
    a_nonlin_res_s = np.convolve(sig_a,temp_ker,'same')
            
    temp_ker_g = np.zeros(45)
    for x in range(len(base)):
        temp_ker_g[x] = f*np.cos(g*np.log(base[x]))+h*np.cos(i*np.log(base[x]))+j
    temp_ker_g = temp_ker_g/np.trapz(temp_ker_g)
    g_nonlin_res_s = np.zeros(len(sig_g))
    g_nonlin_res_s = np.convolve(sig_g,temp_ker_g,'same')
            
    new_res = k*a_nonlin_res_s + g_nonlin_res_s
    
    pre_spike = nonlin(new_res,l,m,n)
    
    spikes = spike_gen(pre_spike, 5)
    return spikes


def find_min(base, sfunc):
    min_v = None
    cur_base = None
    cur_val = None
    for x in range(len(sfunc)):
        if cur_base == None:
            cur_base = base[x]
            cur_val = sfunc[x]
            min_v = sfunc[x]
        elif min_v == sfunc[x] and abs(base[x]) < abs(cur_base):
            cur_base = base[x]
            cur_val = sfunc[x]
            min_v = sfunc[x]
        elif min_v > sfunc[x]:
            cur_base = base[x]
            cur_val = sfunc[x]
            min_v = sfunc[x]
    return (cur_base, cur_val)


def find_space(funncc):
    base =  np.linspace(-5,5,num=100,endpoint=True)
    a_res = []
    for x in base:
        a_res.append(funncc(x))
    (min_base, min_val) = find_min(base, a_res)
        
    base =  np.linspace(min_base-1,min_base+1,num=50,endpoint=True)
    a_res = []
    for x in base:
        a_res.append(funncc(x))
    (min_base, min_val) = find_min(base, a_res)
    return (min_base, min_val)


def generate_center_location(slide_leng):
    col_place = [20,]
    loc = 20
    cur_run = 0
    if np.random.random_sample() < 0.5:
        base = False
    else:
        base = True
    for x in range(1,slide_leng):
        if loc == 4:
            base = True
            cur_run = 2
            loc += 1
            col_place.append(loc)
            continue
        elif loc == 36:
            base = False
            cur_run = 2
            loc -= 1
            col_place.append(loc)
            continue
        elif (loc <= 10 and not base) or (loc >= 30 and base):
            j = 1/2
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
        col_place.append(loc)
    row_place = [20,]
    loc = 20
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
        elif loc == 36:
            base = False
            cur_run = 3
            loc -= 1
            row_place.append(loc)
            continue
        elif (loc <= 10 and not base) or (loc >= 30 and base):
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
    place = np.zeros((2,slide_leng+21))
    for x in range(slide_leng+15):
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

def create_rgc_response(paramm, sig_g, sig_a):
    [a,b,c,d,e,f,g,h,i,j,k,l,m,n] = paramm
    base = np.linspace(.5,4.5,num=15,endpoint=False)
    
    temp_ker = np.zeros(15)
    for x in range(len(base)):
        temp_ker[x] = a*np.cos(b*np.log(base[x]))+c*np.cos(d*np.log(base[x]))+e
    temp_ker = temp_ker/np.trapz(temp_ker)
    a_nonlin_res_s = np.convolve(sig_a,temp_ker,'same')
            
    temp_ker_g = np.zeros(45)
    for x in range(len(base)):
        temp_ker_g[x] = f*np.cos(g*np.log(base[x]))+h*np.cos(i*np.log(base[x]))+j
    temp_ker_g = temp_ker_g/np.trapz(temp_ker_g)
    g_nonlin_res_s = np.convolve(sig_g,temp_ker_g,'same')
            
    new_res = k*a_nonlin_res_s + g_nonlin_res_s
    
    nonlin_response = nonlin(new_res,l,m,n)
    return nonlin_response


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


def compute_gradient(funncc, init_guess, t):
    sigma = .5
    grads = []
    for x in range(len(init_guess)):
        vvall = init_guess[x]
        grads.append(vvall - (funncc(x, vvall+sigma) - funncc(x, vvall-sigma))/(2*sigma*t))
    return grads


def generate_center_location_row_lock(slide_leng):
    col_place = [20,]
    loc = 20
    cur_run = 0
    if np.random.random_sample() < 0.5:
        base = False
    else:
        base = True
    for x in range(1,slide_leng):
        col_place.append(20)
    row_place = [20,]
    loc = 20
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
        elif loc == 36:
            base = False
            cur_run = 3
            loc -= 1
            row_place.append(loc)
            continue
        elif (loc <= 10 and not base) or (loc >= 30 and base):
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
    place = np.zeros((2,slide_leng+21))
    for x in range(slide_leng+15):
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
    height = 41
    center_h = (height - 1)/2
    scence = create_slide_deck(height,nog,line1,roll1,skip1)
    circle = create_circ_gradient(radius, circ_width, height)
    slide_len = height*nog-height-1
    cent_loc = generate_center_location_row_lock(slide_len)
    for z in range(len(scence[0,0,:])):
        new_ccp = cent_loc[0,z]
        new_rrp = cent_loc[1,z]
        for x in range(len(scence[0,:,0])):
            for y in range(len(scence[:,0,0])):
                x_column_j = x-new_ccp
                y_row_j = y-new_rrp
                if radius**2 >= (x_column_j)**2 + (y_row_j)**2:
                    scence[y,x,z] = circle[int(center_h+y_row_j), int(center_h+x_column_j)]
    return scence, cent_loc


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
    cent_locf = cent_locf - 20*np.ones((len(cent_locf[:,0]), len(cent_locf[0,:])))
    scenef, cent_locff = environment_to_gpu(scenef, cent_locf)
    output = net(scenef)
    lossf = mse(output, cent_locff)
    pred_pyf = output.cpu().detach().numpy()
    pred_pyf = np.moveaxis(pred_pyf, -1, 1)    
    
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(cent_locf[0,:])):
        plt.xlim(-10,10)
        plt.scatter(cent_locf[1,i],cent_locf[0,i],c='b')
        plt.scatter(pred_pyf[i,1],pred_pyf[i,0],c='r')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/track_torch_prediction.gif', writer = 'pillow', fps=25)
    return lossf
     

def make_output_test_video(pred_py, cent_loc):
    pred_py = pred_py.cpu().detach().numpy()
    cent_loc = cent_loc.cpu().detach().numpy()
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(cent_loc[0,:,0])):
        plt.xlim(-10,10)
        plt.scatter(cent_loc[-1,i,1],cent_loc[-1,i,0],c='b')
        plt.scatter(pred_py[-1,i,1],pred_py[-1,i,0],c='r')
        camera.snap()
    animation = camera.animate()
    animation.save('Q:/Documents/TDS SuperUROP/track_torch.gif', writer = 'pillow', fps=25)


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
    for i,e in enumerate(cent_loc):
        cent_loc[i] = e - 20*np.ones((len(e[:,0]), len(e[0,:])))
        
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
        hold = np.moveaxis(e, -1, 0)
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




