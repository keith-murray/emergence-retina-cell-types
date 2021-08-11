# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:49:37 2021

@author: Keith
"""

import torch
from retina_model import AnalysisModel

device = torch.device("cuda:0")

def TransplantModel():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = weights['bipolar7.bipolar_space.weight']
    net.bipolar1.bipolar_space.bias.data = weights['bipolar7.bipolar_space.bias']
    net.bipolar1.bipolar_temporal.weight.data = weights['bipolar7.bipolar_temporal.weight']
    net.bipolar1.bipolar_temporal.bias.data = weights['bipolar7.bipolar_temporal.bias']

    x = weights['amacrine3.amacrine_space.weight']
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine3.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine3.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine3.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    net.amacrine1.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = weights['amacrine7.amacrine_space.bias']
    net.amacrine1.amacrine_temporal.weight.data = weights['amacrine7.amacrine_temporal.weight']
    net.amacrine1.amacrine_temporal.bias.data = weights['amacrine7.amacrine_temporal.bias']

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,3:4,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,3:4,:,:,:], x[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_2_types.pt')


def TransplantModel_TopDist():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine2.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine2.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine2.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist.pt')


def TransplantModel_TopDist_san_bipolar():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = torch.zeros(weights['bipolar4.bipolar_space.weight'].shape)
    net.bipolar0.bipolar_space.bias.data = torch.zeros(weights['bipolar4.bipolar_space.bias'].shape)
    net.bipolar0.bipolar_temporal.weight.data = torch.zeros(weights['bipolar4.bipolar_temporal.weight'].shape)
    net.bipolar0.bipolar_temporal.bias.data = torch.zeros(weights['bipolar4.bipolar_temporal.bias'].shape)
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine2.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine2.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine2.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_san_bipolar.pt')


def TransplantModel_TopDist_SanAma():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = torch.zeros(weights['amacrine2.amacrine_space.bias'].shape)
    net.amacrine0.amacrine_temporal.weight.data = torch.zeros(weights['amacrine2.amacrine_temporal.weight'].shape)
    net.amacrine0.amacrine_temporal.bias.data = torch.zeros(weights['amacrine2.amacrine_temporal.bias'].shape)
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((y[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((y[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_san_ama.pt')


def TransplantModel_TopDist_san_ganglion0():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine2.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine2.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine2.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = torch.zeros(weights['ganglion0.ganglion_bipolar_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = torch.zeros(weights['ganglion0.ganglion_bipolar_space.bias'].shape)
    x = torch.zeros(weights['ganglion0.ganglion_amacrine_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = torch.zeros(weights['ganglion0.ganglion_amacrine_space.bias'].shape)
    net.ganglion0.ganglion_temporal.weight.data = torch.zeros(weights['ganglion0.ganglion_temporal.weight'].shape)
    net.ganglion0.ganglion_temporal.bias.data = torch.zeros(weights['ganglion0.ganglion_temporal.bias'].shape)
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_ganglion0.pt')


def TransplantModel_TopDist_san_ganglion1():
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine2.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine2.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine2.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = torch.zeros(weights['ganglion2.ganglion_bipolar_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = torch.zeros(weights['ganglion2.ganglion_bipolar_space.bias'].shape)
    x = torch.zeros(weights['ganglion2.ganglion_amacrine_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = torch.zeros(weights['ganglion2.ganglion_amacrine_space.bias'].shape)
    net.ganglion1.ganglion_temporal.weight.data = torch.zeros(weights['ganglion2.ganglion_temporal.weight'].shape)
    net.ganglion1.ganglion_temporal.bias.data = torch.zeros(weights['ganglion2.ganglion_temporal.bias'].shape)
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_ganglion1.pt')


def TransplantAmacrine_TopDist(amacrine_number):
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    net.bipolar0.bipolar_space.weight.data = weights['bipolar4.bipolar_space.weight']
    net.bipolar0.bipolar_space.bias.data = weights['bipolar4.bipolar_space.bias']
    net.bipolar0.bipolar_temporal.weight.data = weights['bipolar4.bipolar_temporal.weight']
    net.bipolar0.bipolar_temporal.bias.data = weights['bipolar4.bipolar_temporal.bias']
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    exec('''
x = weights['amacrine{0}.amacrine_space.weight']
y = torch.zeros(x.shape).to(device)
net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
net.amacrine0.amacrine_space.bias.data = weights['amacrine{0}.amacrine_space.bias']
net.amacrine0.amacrine_temporal.weight.data = weights['amacrine{0}.amacrine_temporal.weight']
net.amacrine0.amacrine_temporal.bias.data = weights['amacrine{0}.amacrine_temporal.bias']
    '''.format(amacrine_number))
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,amacrine_number:amacrine_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,amacrine_number:amacrine_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_'+str(amacrine_number)+'.pt')


def TransplantBipolar_TopDist(bipolar_number):
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    exec('''
net.bipolar0.bipolar_space.weight.data = weights['bipolar{0}.bipolar_space.weight']
net.bipolar0.bipolar_space.bias.data = weights['bipolar{0}.bipolar_space.bias']
net.bipolar0.bipolar_temporal.weight.data = weights['bipolar{0}.bipolar_temporal.weight']
net.bipolar0.bipolar_temporal.bias.data = weights['bipolar{0}.bipolar_temporal.bias']
    '''.format(bipolar_number))
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = weights['amacrine2.amacrine_space.bias']
    net.amacrine0.amacrine_temporal.weight.data = weights['amacrine2.amacrine_temporal.weight']
    net.amacrine0.amacrine_temporal.bias.data = weights['amacrine2.amacrine_temporal.bias']
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_top_dist_'+str(bipolar_number)+'.pt')


def TransplantBiAm(bipolar_number, amacrine_number):
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    exec('''
net.bipolar0.bipolar_space.weight.data = weights['bipolar{0}.bipolar_space.weight']
net.bipolar0.bipolar_space.bias.data = weights['bipolar{0}.bipolar_space.bias']
net.bipolar0.bipolar_temporal.weight.data = weights['bipolar{0}.bipolar_temporal.weight']
net.bipolar0.bipolar_temporal.bias.data = weights['bipolar{0}.bipolar_temporal.bias']
    '''.format(bipolar_number))
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)

    exec('''
x = weights['amacrine{0}.amacrine_space.weight']
y = torch.zeros(x.shape).to(device)
net.amacrine0.amacrine_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
net.amacrine0.amacrine_space.bias.data = weights['amacrine{0}.amacrine_space.bias']
net.amacrine0.amacrine_temporal.weight.data = weights['amacrine{0}.amacrine_temporal.weight']
net.amacrine0.amacrine_temporal.bias.data = weights['amacrine{0}.amacrine_temporal.bias']
    '''.format(amacrine_number))
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)

    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,amacrine_number:amacrine_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = weights['ganglion2.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = weights['ganglion2.ganglion_bipolar_space.bias']
    x = weights['ganglion2.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,amacrine_number:amacrine_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = weights['ganglion2.ganglion_amacrine_space.bias']
    net.ganglion1.ganglion_temporal.weight.data = weights['ganglion2.ganglion_temporal.weight']
    net.ganglion1.ganglion_temporal.bias.data = weights['ganglion2.ganglion_temporal.bias']
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']

    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_'+str(bipolar_number)+'_'+str(amacrine_number)+'.pt')


def TransplantBipolar_san_ama_gang1(bipolar_number):
    net = AnalysisModel(2, 0.00).to(device)
    save_loc = 'Q:\Documents\TDS SuperUROP\\model\\model.pt'
    weights = torch.load(save_loc)
    
    exec('''
net.bipolar0.bipolar_space.weight.data = weights['bipolar{0}.bipolar_space.weight']
net.bipolar0.bipolar_space.bias.data = weights['bipolar{0}.bipolar_space.bias']
net.bipolar0.bipolar_temporal.weight.data = weights['bipolar{0}.bipolar_temporal.weight']
net.bipolar0.bipolar_temporal.bias.data = weights['bipolar{0}.bipolar_temporal.bias']
    '''.format(bipolar_number))
    
    net.bipolar1.bipolar_space.weight.data = torch.zeros(weights['bipolar7.bipolar_space.weight'].shape)
    net.bipolar1.bipolar_space.bias.data = torch.zeros(weights['bipolar7.bipolar_space.bias'].shape)
    net.bipolar1.bipolar_temporal.weight.data = torch.zeros(weights['bipolar7.bipolar_temporal.weight'].shape)
    net.bipolar1.bipolar_temporal.bias.data = torch.zeros(weights['bipolar7.bipolar_temporal.bias'].shape)
    
    x = weights['amacrine2.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine0.amacrine_space.weight.data = torch.cat((y[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine0.amacrine_space.bias.data = torch.zeros(weights['amacrine2.amacrine_space.bias'].shape)
    net.amacrine0.amacrine_temporal.weight.data = torch.zeros(weights['amacrine2.amacrine_temporal.weight'].shape)
    net.amacrine0.amacrine_temporal.bias.data = torch.zeros(weights['amacrine2.amacrine_temporal.bias'].shape)
    
    x = weights['amacrine7.amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.amacrine1.amacrine_space.weight.data = torch.cat((y[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.amacrine1.amacrine_space.bias.data = torch.zeros(weights['amacrine7.amacrine_space.bias'].shape)
    net.amacrine1.amacrine_temporal.weight.data = torch.zeros(weights['amacrine7.amacrine_temporal.weight'].shape)
    net.amacrine1.amacrine_temporal.bias.data = torch.zeros(weights['amacrine7.amacrine_temporal.bias'].shape)
    
    x = weights['ganglion0.ganglion_bipolar_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_bipolar_space.weight.data = torch.cat((x[:,bipolar_number:bipolar_number+1,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_bipolar_space.bias.data = weights['ganglion0.ganglion_bipolar_space.bias']
    x = weights['ganglion0.ganglion_amacrine_space.weight']
    y = torch.zeros(x.shape).to(device)
    net.ganglion0.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion0.ganglion_amacrine_space.bias.data = weights['ganglion0.ganglion_amacrine_space.bias']
    net.ganglion0.ganglion_temporal.weight.data = weights['ganglion0.ganglion_temporal.weight']
    net.ganglion0.ganglion_temporal.bias.data = weights['ganglion0.ganglion_temporal.bias']
    
    x = torch.zeros(weights['ganglion2.ganglion_bipolar_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_bipolar_space.weight.data = torch.cat((x[:,4:5,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_bipolar_space.bias.data = torch.zeros(weights['ganglion2.ganglion_bipolar_space.bias'].shape)
    x = torch.zeros(weights['ganglion2.ganglion_amacrine_space.weight'].shape).to(device)
    y = torch.zeros(x.shape).to(device)
    net.ganglion1.ganglion_amacrine_space.weight.data = torch.cat((x[:,2:3,:,:,:], y[:,7:8,:,:,:]), dim=1)
    net.ganglion1.ganglion_amacrine_space.bias.data = torch.zeros(weights['ganglion2.ganglion_amacrine_space.bias'].shape)
    net.ganglion1.ganglion_temporal.weight.data = torch.zeros(weights['ganglion2.ganglion_temporal.weight'].shape)
    net.ganglion1.ganglion_temporal.bias.data = torch.zeros(weights['ganglion2.ganglion_temporal.bias'].shape)
    
    x = weights['decisionLeft.decision_space.weight']
    net.decisionLeft.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionLeft.decision_space.bias.data = weights['decisionLeft.decision_space.bias']
    
    x = weights['decisionRight.decision_space.weight']
    net.decisionRight.decision_space.weight.data = torch.cat((x[:,0:1,:,:,:], x[:,2:3,:,:,:]), dim=1)
    net.decisionRight.decision_space.bias.data = weights['decisionRight.decision_space.bias']
    
    torch.save(net.state_dict(), 'Q:\Documents\TDS SuperUROP\\model\\model_bipolar_'+str(bipolar_number)+'.pt')


if __name__ == "__main__":
    # for i in range(8):
        # TransplantBipolar_TopDist(i)
    #     TransplantAmacrine_TopDist(i)
    # bipolar = [1,2,4,6,7]
    # amacrine = [0,2,3,6,7]
    # for b in bipolar:
    #     for a in amacrine:
    #         TransplantBiAm(b, a)
    # TransplantModel_TopDist_san_bipolar()
    # TransplantModel_TopDist_SanAma()
    # TransplantModel_TopDist_san_ganglion0()
    # TransplantModel_TopDist_san_ganglion1()
    for i in range(8):
        TransplantBipolar_san_ama_gang1(i)
    
    print('Milhouse is not a meme.')
    