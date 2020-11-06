# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 02:01:14 2020

@author: Keith
"""

import numpy as np
from scipy import signal
from support import create_slide_deck, mexican_hat, mat_conv, nonlin_bi, create_rgc_spike, generate_center_location, create_circ_gradient,find_space


def envrionment(nog, line1, roll1, skip1, circ_width, radius):
    # height is the size of the stimulus frame
    # nog is number of gradients
    # line1 is the line size
    # roll1 is the horrizontal roll factor
    # skip1 is the vertical row skip factor
    # circ_width is the width of the circle checker pattern
    # radius is the radius of the circle
    height = 41
    center_h = (height - 1)/2
    scence = create_slide_deck(height,nog,line1,roll1,skip1)
    circle = create_circ_gradient(radius, circ_width, height)
    slide_len = height*nog-height-1
    cent_loc = generate_center_location(slide_len)
    for z in range(len(scence[0,0,:])):
        new_ccp = cent_loc[1,z]
        new_rrp = cent_loc[0,z]
        for x in range(len(scence[0,:,0])):
            for y in range(len(scence[:,0,0])):
                x_column_j = x-new_ccp
                y_row_j = y-new_rrp
                if radius**2 >= (x_column_j)**2 + (y_row_j)**2:
                    scence[y,x,z] = circle[int(center_h+y_row_j), int(center_h+x_column_j)]
    return scence, cent_loc


def create_response(params, scence):
    # Will only work for height 41
    x_range = range(-2,3,1)
    y_range = range(-2,3,1)
    ret_func = np.zeros((5,5))
    for x in x_range:
        for y in y_range:
            ret_func[x+2,y+2] = mexican_hat(x,y,.66) # Make another parameter
    a_temp_res = np.zeros((10,10,len(scence[0,0,:])))
    for x in range(len(scence[0,0,:])):
        a_temp_res[:,:,x] = mat_conv(scence[:,:,x], ret_func) # Have a temporal kernel on biopolar cell?
    a_rgc_a_res = np.zeros((10,10,len(scence[0,0,:])))
    for y in range(10):
        for x in range(10):
            int_res = nonlin_bi(a_temp_res[x,y,:])
            a_rgc_a_res[x,y,:] = int_res
    amacrine_1 = np.sum(a_rgc_a_res[0:3,0:3,:],axis=(0,1))
    amacrine_2 = np.sum(a_rgc_a_res[0:3,7:,:],axis=(0,1))
    amacrine_3 = np.sum(a_rgc_a_res[7:,0:3,:],axis=(0,1))
    amacrine_4 = np.sum(a_rgc_a_res[7:,7:,:],axis=(0,1))
    
    gang_cell_1 = np.sum(a_rgc_a_res[0:3,4:7,:],axis=(0,1))
    gang_cell_2 = np.sum(a_rgc_a_res[3:6,6:9,:],axis=(0,1))
    gang_cell_3 = np.sum(a_rgc_a_res[6:9,4:7,:],axis=(0,1))
    gang_cell_4 = np.sum(a_rgc_a_res[3:6,1:4,:],axis=(0,1))
    
    sspp_1 = create_rgc_spike(params,gang_cell_1,amacrine_1)
    sspp_2 = create_rgc_spike(params,gang_cell_2,amacrine_2)
    sspp_3 = create_rgc_spike(params,gang_cell_3,amacrine_3)
    sspp_4 = create_rgc_spike(params,gang_cell_4,amacrine_4)
    
    window = signal.gaussian(21, std=2.5)
    window_tp = np.trapz(window)
    window = window/window_tp
    
    conspi_1 = np.convolve(sspp_1, window, 'same')
    conspi_2 = np.convolve(sspp_2, window, 'same')
    conspi_3 = np.convolve(sspp_3, window, 'same')
    conspi_4 = np.convolve(sspp_4, window, 'same')
    
    ret = np.zeros((2,len(scence[0,0,:])))
    for t in range(len(scence[0,0,:])):
        ret[0,t] = -3*conspi_1[t] + 3*conspi_3[t] # Row component
        ret[1,t] = 2*conspi_2[t] - 3*conspi_4[t] # Component Component
        # TODO : Make a decoder based on a linear regression algo (least squared?)
    return ret
    

def loss_function(cent_loc, pred_loc):
    '''
    There are many ways to make a loss function. Mean squared error across all
    time points is an attractive metric, but could be uninformative. Squared
    sum error may be interesting because the number could become quite large.
    
    This implimentation is with mean squared error across all time points.
    '''
    total_frames = len(cent_loc[0,:])
    cent_loc = np.roll(cent_loc,-4,1) # NOTE: We are predicting just 4 frames ahead
    row_running_tot = 0
    col_running_tot = 0
    for t in range(total_frames):
        row_running_tot = (cent_loc[0,t]-20-pred_loc[0,t])**2
        col_running_tot = (cent_loc[1,t]-20-pred_loc[1,t])**2
    loss_result = row_running_tot/total_frames + col_running_tot/total_frames
    # NOTE: Loss result is not the average of rows and columns, but added
    return loss_result


def weak_optimization_function(initial_val, scence, cent_loc):
    '''
    I classify this as a weak optimization function because it is doing a simple
    parameter space to look for a local optimum set of parameters.
    '''
    new_val = []
    for val in range(len(initial_val)):
        def place_func(vvaall):
            test_vals = [x for x in initial_val]
            test_vals[val] = vvaall
            pred_res = create_response(test_vals, scence)
            res = loss_function(cent_loc, pred_res)
            return res
        (min_base, min_val) = find_space(place_func)
        new_val.append(min_base)
    new_val_2 = []
    for val in range(len(new_val)):
        def place_func(vvaall):
            test_vals = [x for x in new_val]
            test_vals[val] = vvaall
            pred_res = create_response(test_vals, scence)
            res = loss_function(cent_loc, pred_res)
            return res
        (min_base, min_val) = find_space(place_func)
        new_val_2.append(min_base)
    for val in range(len(new_val_2)):
        def place_func(vvaall):
            test_vals = [x for x in new_val_2]
            test_vals[val] = vvaall
            pred_res = create_response(test_vals, scence)
            res = loss_function(cent_loc, pred_res)
            return res
        (min_base, min_val) = find_space(place_func)
        new_val_2[val] = min_base
    for val in range(len(new_val_2)):
        def place_func(vvaall):
            test_vals = [x for x in new_val_2]
            test_vals[val] = vvaall
            pred_res = create_response(test_vals, scence)
            res = loss_function(cent_loc, pred_res)
            return res
        (min_base, min_val) = find_space(place_func)
        new_val_2[val] = min_base
    return (new_val_2, min_val)



if __name__ == "__main__":
    scence, cent_loc = envrionment(20, 4, 0, 1, 2, 4)
    initial_val = [-2,3,2,1,4,-2,5,-1,1,-1,-2,-1,-3, 3]
    (new_vals, min_val) = weak_optimization_function(initial_val, scence, cent_loc)

#   [3.4388785817357244, 1.7880849309420734, 3.9806225520511243, -5.022469593898165, -4.620490620490621, -0.7779839208410639, -1.6227581941867657, -3.604205318491033, -2.3042671614100185, -2.3228200371057515, -4.181818181818182, 0.838177695320552, 1.0830756545042253, 0.45351473922902463]
#   Loss value of 0.00032381174009297125








