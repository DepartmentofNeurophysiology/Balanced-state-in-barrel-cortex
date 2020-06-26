"""
small thalamus[n = 200]; gvariable = 0-10 (stepsize = 0.5);
input frequencies = 0-70Hz
Calculated per configuration (baseline, 3, 4, 5)

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
print('Packages loaded')

#%% Simulation details

# import parameters.py file
import parameters_adi
print('Parameters file imported')

parameters_adi.neuron_num()

simtime             = 5000*ms   # simulation time
#%% Spike count per layer

                    #### requires all input frequencies and gvariables for a configuration ####
count_g             = 0             # counter
gvariable           = linspace(0,10,21)
input_r             = linspace(0,70,8)
output_spikemon_L4  = zeros(((len(input_r)), len(gvariable))) # store data
output_spikemon_L23 = zeros(((len(input_r)), len(gvariable)))

for g in gvariable:
    g = g
    count_r = 0               # counter
    
    for r in input_r:
        r = r                 # making sure previous value of r isnt used for the new iteration
# Load Spike data
        # L4
        print('Loading e4')
        e4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spikestest_4e_i_g=%g_r=%g.npy' 
                          %(g,r))
        print('Loading i4')
        i4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spikestest_4i_i_g=%g_r=%g.npy' 
                          %(g,r))
        # No. of spikes
        output_spikemon_L4[count_r][count_g] = (len(e4) + len(i4))/simtime
        
        # L2/3
        print('Loading e23')
        e23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spikestest_23e_i_g=%g_r=%g.npy' 
                          %(g,r))
        print('Loading i23')
        i23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spikestest_23i_i_g=%g_r=%g.npy' 
                          %(g,r))
        # No. of spikes in L2/3
        output_spikemon_L23[count_r][count_g] = (len(e23) + len(i23))/simtime

        count_r = count_r + 1
    count_g = count_g + 1

print('Spike data loaded')

#%% storing data

f = open('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\firingfreqs_values.pckl', 'wb')
pickle.dump([output_spikemon_L4, output_spikemon_L23], f)
f.close()

#%% Spike Frequency: at specific thalamic input frequency ; plots

                  #### requires all gvariables, one input frequency, all configurations, different neuron nums. ####
# rel. inhibitory strength
count_g             = 0             # counter
gvariable           = linspace(0,10,21)
input_r             = linspace(0,70,8)

# ### empty matrices
# # Original (baseline)
# mean_rate_g_4 = zeros((len(gvariable)))
# mean_rate_g_23 = zeros((len(gvariable)))
# sd_rate_g_4 = zeros((len(gvariable)))
# sd_rate_g_23 = zeros((len(gvariable)))
# temp_4 = []
# temp_23 = []

# # ratio 1/3 (configuration 3)
# mean_rate_g_4_3 = zeros((len(gvariable)))
# mean_rate_g_23_3 = zeros((len(gvariable)))
# sd_rate_g_4_3 = zeros((len(gvariable)))
# sd_rate_g_23_3 = zeros((len(gvariable)))
# temp_4_3 = []
# temp_23_3 = []

# ratio 1/4 (configuration 4)
mean_rate_g_4_4 = zeros(((len(input_r)), len(gvariable)))
mean_rate_g_23_4 = zeros(((len(input_r)), len(gvariable)))
sd_rate_g_4_4 = zeros(((len(input_r)), len(gvariable)))
sd_rate_g_23_4 = zeros(((len(input_r)), len(gvariable)))
temp_4_4 = []
temp_23_4 = []

# # ratio 1/5 (configuration 5)
# mean_rate_g_4_5 = zeros((len(gvariable)))
# mean_rate_g_23_5 = zeros((len(gvariable)))
# sd_rate_g_4_5 = zeros((len(gvariable)))
# sd_rate_g_23_5 = zeros((len(gvariable)))
# temp_4_5 = []
# temp_23_5 = []

# define excitatory neurons in both layers
parameters_adi.n_e_4 = 1374
parameters_adi.n_e_23 = 2232

print('Calculating mean firing rates with standard deviation at thalamic input freq. of ' + str(input_freq))

for g in gvariable:
    g = g
    count_r = 0               # counter
    
    for r in input_r:
        r = r
    
    ##### Original #####
    # # L4 
    #  # load spikes (# set allow_pickle=True explicitly)
    # e4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset1\\baseline\\Spikestest_4e_i_g=%g_r=50.npy' %(g),allow_pickle = True)
    # i4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset11\\baseline\\Spikestest_4i_i_g=%g_r=50.npy' %(g),allow_pickle = True)
    #  # load spike trains
    # e4t = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spiketraintest_4e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i4t = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spiketraintest_4i_g=%g_r=50.npy' %(g),allow_pickle=True)
    #  # Number of spikes per layer
    # mean_rate_g_4[count_g] = len(e4) + len(i4)
    
    # parameters.n_i_4 = 187 # defined per configuration (since the value changes)
    # parameters.n_i_23 = 425
    
    #  # std
    # for i in range(parameters.n_e_4):
    #     if i < parameters.n_i_4:
    #         t = ((len(i4t[0][i])) + (len(e4t[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e4t[0][i]))/simtime
    #     temp_4.append(t)
    # # Store std    
    # sd_rate_g_4[count_g] = std(temp_4)

    # # L2/3
    #  # Load spikes
    # e23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spikestest_23e_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spikestest_23i_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    #  # load spike trains
    # e23t = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spiketraintest_23e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23t = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\baseline\\Spiketraintest_23i_g=%g_r=50.npy' %(g),allow_pickle=True)    
    #  # Number of spikes per layer
    # mean_rate_g_23[count_g] = len(e23) + len(i23)

    #  # std
    # for i in range(parameters.n_e_23):
    #     if i < parameters.n_i_23:
    #         t = ((len(i23t[0][i])) + (len(e23t[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e23t[0][i]))/simtime
    #     temp_23.append(t)
    #  # Store std    
    # sd_rate_g_23[count_g] = std(temp_23)
 
    # ##### ratio 1/3 #####
    # parameters.n_i_4 = 458
    # parameters.n_i_23 = 744

    # # L4
    #  # load spikes
    # e4_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spikestest_4e_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i4_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spikestest_4i_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    #  # load spike trains
    # e4t_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spiketraintest_4e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i4t_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spiketraintest_4i_g=%g_r=50.npy' %(g),allow_pickle=True)    
    #  # Number of spikes per layer
    # mean_rate_g_4_3[count_g] = len(e4_3) + len(i4_3)

    #  # std
    # for i in range(parameters.n_e_4):
    #     if i < parameters.n_i_4:
    #         t = ((len(e4t_3[0][i])) + (len(i4t_3[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e4t_3[0][i]))/simtime
    #     temp_4_3.append(t)
    #  # Store std    
    # sd_rate_g_4_3[count_g] = std(temp_4_3)
  
    # # L2/3
    #  # load spikes
    # e23_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spikestest_23e_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spikestest_23i_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    #  # load spike trains
    # e23t_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spiketraintest_23e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23t_3 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_3\\Spiketraintest_23i_g=%g_r=50.npy' %(g),allow_pickle=True)       
    #  # Number of spikes per layer
    # mean_rate_g_23_3[count_g] = len(e23_3) + len(i23_3)
    
    #  # std
    # for i in range(parameters.n_e_23):
    #     if i < parameters.n_i_23:
    #         t = ((len(e23t_3[0][i])) + (len(i23t_3[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e23t_3[0][i]))/simtime
    #     temp_23_3.append(t)
    #  # Store std    
    # sd_rate_g_23_3[count_g] = std(temp_23_3)

    ##### ratio 1/4 #####
        parameters_adi.n_i_4 = 343
        parameters_adi.n_i_23 = 558
    
    # L4
     # load spikes
        e4_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spikestest_4e_i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
        i4_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spikestest_4i_i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
     # load spike trains
        e4t_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spiketraintest_4e_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
        i4t_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spiketraintest_4i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)  
     # Number of spikes per layer
        mean_rate_g_4_4[count_r][count_g] = len(e4_4) + len(i4_4)
    
     # std
        for i in range(parameters_adi.n_e_4):
            if i < parameters_adi.n_i_4:
                t = ((len(e4t_4[0][i])) + (len(i4t_4[0][i])))/(2)/simtime
            else:
                t = (len(e4t_4[0][i]))/simtime
            temp_4_4.append(t)
     # Store std    
        sd_rate_g_4_4[count_r][count_g] = std(temp_4_4)

    # L2/3
     # load spikes
        e23_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spikestest_23e_i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
        i23_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spikestest_23i_i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
     # load spike trains
        e23t_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spiketraintest_23e_g=%g_r=%g.npy' %(g,r),allow_pickle=True)
        i23t_4 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset15\\config_4\\Spiketraintest_23i_g=%g_r=%g.npy' %(g,r),allow_pickle=True)   
     # Number of spikes per layer
        mean_rate_g_23_4[count_r][count_g] = len(e23_4) + len(i23_4)

     # std
        for i in range(parameters_adi.n_e_23):
            if i < parameters_adi.n_i_23:
                t = ((len(e23t_4[0][i])) + (len(i23t_4[0][i])))/(2)/simtime
            else:
                t = (len(e23t_4[0][i]))/simtime

            temp_23_4.append(t)
     # Store std    
        sd_rate_g_23_4[count_g] = std(temp_23_4)
     
    ##### ratio 1/5 #####
    # parameters.n_i_4 = 274
    # parameters.n_i_23 = 446

    # # L4
    # # load spikes
    # e4_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spikestest_4e_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i4_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spikestest_4i_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # # load spike trains
    # e4t_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spiketraintest_4e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i4t_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spiketraintest_4i_g=%g_r=50.npy' %(g),allow_pickle=True)  
    # # Number of spikes per layer
    # mean_rate_g_4_5[count_g] = len(e4_5) + len(i4_5)

    # # std
    # for i in range(parameters.n_e_4):
    #     if i < parameters.n_i_4:
    #         t = ((len(e4t_5[0][i])) + (len(i4t_5[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e4t_5[0][i]))/simtime

    #     temp_4_5.append(t)
    # # Store std    
    # sd_rate_g_4_5[count_g] = std(temp_4_5)       
    
    # # L2/3
    # # load spikes
    # e23_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spikestest_23e_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spikestest_23i_i_g=%g_r=50.npy' %(g),allow_pickle=True)
    # # load spike trains   
    # e23t_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spiketraintest_23e_g=%g_r=50.npy' %(g),allow_pickle=True)
    # i23t_5 = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset9\\config_5\\Spiketraintest_23i_g=%g_r=50.npy' %(g),allow_pickle=True)  
    # # Number of spikes per layer
    # mean_rate_g_23_5[count_g] = len(e23_5) + len(i23_5)
    
    # # std
    # for i in range(parameters.n_e_23):
    #     if i < parameters.n_i_23:
    #         t = ((len(e23t_5[0][i])) + (len(i23t_5[0][i])))/(2)/simtime
    #     else:
    #         t = (len(e23t_5[0][i]))/simtime

    #     temp_23_5.append(t)
    # # Store std    
    # sd_rate_g_23_5[count_g] = std(temp_23_5)

        count_r = count_g + 1
    count_g = 0

print('Mean firing rates calculated for all input frequencies')

#%% storing data

f = open('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset14\\config_4\\ff_vals.pckl', 'wb')
pickle.dump([output_spikemon_L4, output_spikemon_L23], f)
f.close()