# -*- coding: utf-8 -*-
"""
Coefficients of correlation matrix
"""

#%% Load libraries

from brian2 import*

get_ipython().run_line_magic('matplotlib', 'inline')
# ipdb = Pdb()
from IPython.core.debugger import Pdb
import os

from random import sample
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as sp
from scipy import signal
import seaborn as sns
import pickle

prefs.codegen.target = "numpy" # avoid error message
print('Libraries loaded')

#%% Simulation details

# import parameters.py file
import parameters_sim
print('Parameters file imported')

parameters_sim.neuron_num() # input neuron numbers

simtime             = 5000*ms   # simulation time
dt_value = 0.1

#%% Coefficient of correlation Matrix  
#### Calculated per configuration --- at a particular input frequency --- at a particular synaptic ratio ####
count_g = 0
gvariable = linspace(0,10,21)

for g in gvariable:
    g = g
    count_g = count_g + 1
    
    print('Analysing data for g = ' + str(g))
   
    np.random.seed(count_g)                      # set a seed for each synaptic ratio (seed no. = no. of synaptic ratio: 1-21)
    
    # create random index 100 neurons
    x4_e = np.random.randint(0, parameters_sim.n_e_4, 100)
    x4_i = np.random.randint(0, parameters_sim.n_i_4, 100)
    x23_e = np.random.randint(0, parameters_sim.n_e_23, 100)
    x23_i = np.random.randint(0, parameters_sim.n_i_23, 100)

    # create empty matrix to store coefficients
    coef_matrix_4e = full((100,100), nan)
    coef_matrix_4i = full((100,100), nan)
    coef_matrix_23e = full((100,100), nan)
    coef_matrix_23i = full((100,100), nan)

    # Load spike trains
    # L4
    print('Loading spike trains for L4 - excitatory, at g = ' + str(g))
    train_e_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_4e_g=%g_r=20.npy'%(g),
                    allow_pickle=True)[0]
    
    print('Loading spike trains for L4 - inhibitory, at g = ' + str(g))
    train_i_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_4e_g=%g_r=20.npy'%(g),
                    allow_pickle=True)[0]

    # L2/3
    print('Loading spike trains for L2/3 - excitatory, at g = ' + str(g))
    train_e_23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_23e_g=%g_r=20.npy'%(g),
                     allow_pickle=True)[0]
   
    print('Loading spike trains for L2/3 - inhibitory, at g = ' + str(g))
    train_i_23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\Spiketraintest_23i_g=%g_r=20.npy'%(g),
                     allow_pickle=True)[0]

    # Create counter
    count_ii = -1

    for ii in range(100):
        # Per train
    
        # Index Random Spike Trains
        x4_ee = x4_e[ii]
        x4_ii = x4_i[ii]
        x23_ee = x23_e[ii]
        x23_ii = x23_i[ii]
    
        # Create binary spike trains
        print('Creating 1st set of binary spike trains')
    
        # L4 -- e
        train_e_41 = train_e_4[x4_ee]
        train_e_41 = [int(p) for p in (train_e_41/(dt_value*ms))]
        zero_e_41 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_e_41[train_e_41] = 1
        # L4 -- i
        train_i_41 = train_i_4[x4_ii]
        train_i_41 = [int(p) for p in (train_i_41/(dt_value*ms))]
        zero_i_41 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_i_41[train_e_41] = 1
        # L2/3 -- e
        train_e_231 = train_e_23[x23_ee]
        train_e_231 = [int(p) for p in (train_e_231/(dt_value*ms))]
        zero_e_231 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_e_231[train_e_231] = 1
        # L2/3 -- i
        train_i_231 = train_i_23[x23_ii]
        train_i_231 = [int(p) for p in (train_i_231/(dt_value*ms))]
        zero_i_231 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
        zero_i_231[train_i_231] = 1
    
        # Index
        count_ii = count_ii + 1
        count_jj = 0
    
        for jj in range(100):
            
            # Per train
            x4_ee2 = x4_e[jj]
            x4_ii2 = x4_i[jj]
            x23_ee2 = x23_e[jj]
            x23_ii2 = x23_i[jj]
    
            #### Create binary spike trains ####
            print('Creating 2nd set of binary spike trains')

            # L4 -- e
            train_e_42 = train_e_4[x4_ee2]
            train_e_42 = [int(p) for p in (train_e_42/(dt_value*ms))]
            zero_e_42 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
            zero_e_42[train_e_42] = 1
            # L4 -- i
            train_i_42 = train_i_4[x4_ii2]
            train_i_42 = [int(p) for p in (train_i_42/(dt_value*ms))]
            zero_i_42 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
            zero_i_42[train_i_42] = 1
            # L2/3 -- e
            train_e_232 = train_e_23[x23_ee2]
            train_e_232 = [int(p) for p in (train_e_232/(dt_value*ms))]
            zero_e_232 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
            zero_e_232[train_e_232] = 1
            # L2/3 -- i
            train_i_232 = train_i_23[x23_ii2]
            train_i_232 = [int(p) for p in (train_i_232/(dt_value*ms))]
            zero_i_232 = zeros(((int(simtime/ms)*(int(1/dt_value)))))
            zero_i_232[train_i_232] = 1
    
            #### Coefficients of correlation matrix ####
            
            print('Creating Coefficients of correlation matrix for L4 - e')
            # L4 -- e
            coe_4e = np.corrcoef(zero_e_41,zero_e_42)
            coef_4e = coe_4e[1,0]
            if count_ii == count_jj:
                coef_matrix_4e[count_ii][count_jj] = NaN    
            else:
                coef_matrix_4e[count_ii][count_jj] = coef_4e
         
            print('Creating Coefficients of correlation matrix for L4 - i')
            # L4 -- i
            coe_4i = np.corrcoef(zero_i_41,zero_i_42)
            coef_4i = coe_4i[1,0]
            if count_ii == count_jj:
                coef_matrix_4i[count_ii][count_jj] = NaN    
            else:
                coef_matrix_4i[count_ii][count_jj] = coef_4i
            
            print('Creating Coefficients of correlation matrix for L2/3 - e')
            # L2/3 -- e
            coe_23e = np.corrcoef(zero_e_231,zero_e_232)
            coef_23e = coe_23e[1,0]  
            if count_ii == count_jj:
                coef_matrix_23e[count_ii][count_jj] = NaN    
            else:
                coef_matrix_23e[count_ii][count_jj] = coef_23e
            
            print('Creating Coefficients of correlation matrix for L2/3 - i')
            # L2/3 -- i
            coe_23i = np.corrcoef(zero_i_231,zero_i_232)
            coef_23i = coe_23i[1,0]
            if count_ii == count_jj:
                coef_matrix_23i[count_ii][count_jj] = NaN    
            else:
               coef_matrix_23i[count_ii][count_jj] = coef_23i

            count_jj = count_jj + 1
    
    print('Storing data for g = ' + str(g))
    ## save data for each synaptic ratio
    ss = 'E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset25\\config_5\\coeff_matrix_g=%g_r=20.pckl'%(g)
    f = open(ss, 'wb')
    pickle.dump([coef_matrix_4e, coef_matrix_4i, coef_matrix_23e, coef_matrix_23i], f)
    f.close()