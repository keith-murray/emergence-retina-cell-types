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


if __name__ == "__main__":
    print('Mihouse is not a meme.')
