#!/usr/bin/env python
# coding: utf-8
'''
Generates gaussian spike trains
'''

#%% Load libraries

from brian2 import*
get_ipython().run_line_magic('matplotlib', 'inline')
from random import sample
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from IPython.core.debugger import Pdb
ipdb = Pdb()
import scipy
import scipy as sp
from scipy import signal

#%% Sim. details

import parameters_sim # import parameter file

parameters_sim.neuron_num() # load neuron numbers

# Simtime
simtime = 5000*ms
dt_value = 0.1

#%% Gaussian spikes function

def gaussian(spiketrain, truncate = 3, sigma = 500):
    # spiketrain =  binary spiketrain
    
    # Width of function
    window_size = 2*int(truncate*sigma + 0.5) + 1
    
    # Gaussian function
    gaussian_function = signal.gaussian(window_size, sigma)
    
    # Convolution
    spiketrain_conv = np.convolve(gaussian_function, spiketrain, 'same')

    return spiketrain_conv

#%% Gaussian spike generation (for each configuration-- all synaptic ratios; specific thalamic input frequency) 
# store in the same folder as spikes, spike trains etc.

gvariable = linspace(0,10,21) # synaptic ratios

# generate 150 random numbers (zero to neuron number)
idx_4e = np.random.randint(0,parameters_sim.n_e_4,150)
idx_4i = np.random.randint(0,parameters_sim.n_i_4,150)
idx_23e = np.random.randint(0,parameters_sim.n_e_23,150)
idx_23i = np.random.randint(0,parameters_sim.n_i_23,150)

for ii in gvariable:
    ii = ii
    
    # Load spike trains for all gvariables and each thalamic input frequency
    spikes_4e_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_4e_g=%g_r=20.npy'%ii,allow_pickle=True)[0]
    spikes_4i_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_4i_g=%g_r=20.npy' %ii,allow_pickle=True)[0]
    spikes_23i_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_23i_g=%g_r=20.npy'%ii,allow_pickle=True)[0]
    spikes_23e_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_23e_g=%g_r=20.npy' %ii,allow_pickle=True)[0]
    
    e23 = []
    i23 = []
    e4 = []
    i4 = []
    
    gg = ii*20

    # L2/3 - e
    for jj in range(150):
    # Create binary spike train
        train_e_23 = spikes_23e_i[idx_23e[jj]]
        train_e_23 = [int(p) for p in (train_e_23/(dt_value*ms))]
        
        zero_e_23 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        
        zero_e_23[train_e_23] = 1
        
        # Create gaussian spike train from binary spike train
        gaus_e_23 = gaussian(zero_e_23)  
        #  append gaussian spike trains to previous value
        e23.append(gaus_e_23)
    
    # Save spike train
    np.save('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Gaus_23e_g=%i_r=20' %gg, e23)
    print('g=%i'%gg, ii)
    
    # L2/3 - i
    for jj in range(150):
        train_i_23 = spikes_23i_i[idx_23i[jj]]
        train_i_23 = [int(p) for p in (train_i_23/(dt_value*ms))]
        zero_i_23 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_i_23[train_i_23] = 1
        gaus_i_23 = gaussian(zero_i_23)
        
        i23.append(gaus_i_23)
    # Save spike train
    np.save('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Gaus_23i_g=%i_r=20' %gg, i23)
    
    # L4 - excit.
    for jj in range(150):
        train_e_4 = spikes_4e_i[idx_4e[jj]]
        train_e_4 = [int(p) for p in (train_e_4/(dt_value*ms))]
        zero_e_4 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_e_4[train_e_4] = 1
        
        gaus_e_4 = gaussian(zero_e_4)

        e4.append(gaus_e_4)
    # Save spike train
    np.save('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Gaus_4e_g=%i_r=20' %gg, e4)
    
    # L4 - inhib.
    for jj in range(150):
        train_i_4 = spikes_4i_i[idx_4i[jj]]
        train_i_4 = [int(p) for p in (train_i_4/(dt_value*ms))]
        zero_i_4 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_i_4[train_i_4] = 1
        gaus_i_4 = gaussian(zero_i_4)   
        
        i4.append(gaus_i_4)
    # Save spike train
    np.save('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Gaus_4i_g=%i_r=20' %gg, i4)

print('Gaussian spike trains generated')
