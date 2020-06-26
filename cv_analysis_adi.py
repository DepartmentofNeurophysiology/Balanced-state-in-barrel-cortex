# -*- coding: utf-8 -*-
"""
Coefficient of variation
analysis (function, calculation, pickle store)

"""
#%% Load libraries

from brian2 import*

get_ipython().run_line_magic('matplotlib', 'inline')
# ipdb = Pdb()
from IPython.core.debugger import Pdb
import os
import pickle

from random import sample
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as sp
from scipy import signal
import seaborn as sns

prefs.codegen.target = "numpy" # avoid error message
print('Packages loaded')

#%% Simulation details

# import parameters_adi.py file
import parameters_adi
print('Parameters file imported')

simtime = 5000*ms   # simulation time

#%% define Coefficient of Variation

def CV(spikes):
    '''
    Coefficient of variation
    '''
    if spikes == []:
        return 0
    ISI = diff(spikes) # Interspike intervals
    return std(ISI) / mean(ISI) # standard deviation ISI/mean ISI

#%% Analysis (over input frequencies 10-70Hz ; all synaptic ratios)

parameters_adi.neuron_num()

gvariable = linspace(0,10,21)
input_r = linspace(10,70,7)

# Initialize trains to 0
train_4_e = 0
train_4_i = 0
train_23_e = 0
train_23_i = 0

# Define empty matrices to store data
mean_cv_4_e = full(((len(input_r)),len(gvariable)), nan)
mean_cv_4_i = full(((len(input_r)),len(gvariable)), nan)

mean_cv_23_e = full(((len(input_r)),len(gvariable)), nan)
mean_cv_23_i = full(((len(input_r)),len(gvariable)), nan)

mean_cv_4_layer4 = full(((len(input_r)),len(gvariable)), nan)
mean_cv_23_layer23 = full(((len(input_r)),len(gvariable)), nan)

count_r = 0

# Load data
for r in input_r:
    
    r = r
    count_g = 0
    
    for g in gvariable:
         
        cv_sum_4_e = 0
        cv_sum_4_i = 0
        cv_sum_23_e = 0
        cv_sum_23_i = 0
        cv_sum_23_layer23 = 0
        cv_sum_4_layer4 = 0
    
        a,b,c,d = 0,0,0,0
    
        print("Calculating CV for L23-e")
        # L23--e
        for ii in range(parameters_adi.n_e_23):
            train_23_e = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spiketraintest_23e_g=%g_r=%g.npy'%(g,r),
                             allow_pickle=True)[0][ii]
            cv_23_e = CV(train_23_e)
            if cv_23_e != cv_23_e:
                print('nan e')
            else:
                cv_sum_23_e = cv_sum_23_e + cv_23_e
                # Full layer 
                cv_sum_23_layer23 = cv_sum_23_layer23 + cv_23_e
                a = a + 1  
        
        print("Calculating CV for L23-i")
        # L23--i
        for ii in range(parameters_adi.n_i_23):
            train_23_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spiketraintest_23i_g=%g_r=%g.npy'%(g,r),
                             allow_pickle=True)[0][ii]
            cv_23_i = CV(train_23_i)
            if cv_23_i != cv_23_i:
                print('nan i')
            else:
                cv_sum_23_i = cv_sum_23_i + cv_23_i
                # Full layer 
                cv_sum_23_layer23 = cv_sum_23_layer23 + cv_23_i
                b = b + 1
    
        print("Calculating CV for L4-e")    
        # L4--e
        for ii in range(parameters_adi.n_e_4):
            train_4_e = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spiketraintest_4e_g=%g_r=%g.npy'%(g,r),
                            allow_pickle=True)[0][ii]
            cv_4_e = CV(train_4_e)
            if cv_4_e != cv_4_e:
                print('nan e')
            else:
                cv_sum_4_e = cv_sum_4_e + cv_4_e        
                # Full layer
                cv_sum_4_layer4 = cv_sum_4_layer4 + cv_4_e
                c = c + 1        
        
        print("Calculating CV for L4-i")
        # L4--i
        for ii in range(parameters_adi.n_i_4):
            train_4_i = np.load('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\Spiketraintest_4i_g=%g_r=%g.npy'%(g,r),
                            allow_pickle=True)[0][ii]
            cv_4_i = CV(train_4_i)
            if cv_4_i != cv_4_i:
                print('nan i')
            else:
                cv_sum_4_i = cv_sum_4_i + cv_4_i 
                # Full layer
                cv_sum_4_layer4 = cv_sum_4_layer4 + cv_4_i
                d = d + 1
        # Per layer per neuron type
        mean_cv_4_e[count_r][count_g] = cv_sum_4_e/c
        mean_cv_4_i[count_r][count_g] = cv_sum_4_i/d
        mean_cv_23_e[count_r][count_g] = cv_sum_23_e/a
        mean_cv_23_i[count_r][count_g] = cv_sum_23_i/b
        # Full layer
        mean_cv_4_layer4[count_r][count_g] = cv_sum_4_layer4/(c+d)
        mean_cv_23_layer23[count_r][count_g] = cv_sum_23_layer23/(a+b)
        # Update index
        count_g = count_g + 1

    print('count_g = ' + str(count_g))
    print("CV calculated")
    
    count_r = count_r + 1

print('count_r = ' +str(count_r))

#%% storing data

f = open('E:\\Zeldenrust lab\\pythoncodes\\balanced_state\\dataset16\\config_4\\cv_values.pckl', 'wb')
pickle.dump([mean_cv_4_e, mean_cv_4_i, mean_cv_23_e, mean_cv_23_i, mean_cv_4_layer4, mean_cv_23_layer23], f)
f.close()