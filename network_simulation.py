#!/usr/bin/env python

"""
Simulation script : (parameters loaded from parameters_adi.py)

"""
#%% Import libraries and Parameter file

import IPython.core.debugger
import random
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # plotting in Ipython console
get_ipython().run_line_magic('matplotlib', 'qt') # plotting in external window

import mpl_toolkits.mplot3d
import matplotlib
import matplotlib.ticker
import numpy as np
import scipy as sp
from scipy import signal
import pickle

from brian2 import*
from brian2.core.network import TextReport

prefs.codegen.target = "numpy" # avoid error message during code compilation

#%% Parameters
import parameters_adi # load parameters file

parameters_adi.neuron_num() # neuron numbers from README.md
parameters_adi.prob_vals() # network upscaling probablilty values

# projection widths
sigmaffwd_4, sigmaRe_4, sigmaRi_4 = .1*mmeter, .05*mmeter, .03*mmeter
sigmaffwd_23, sigmaRe_23, sigmaRi_23 = .05*mmeter, .15*mmeter, .03*mmeter

# LIF model characterisitics
# excit.
taumem_e = 15*ms
vT_e = -50*mV
El_e = -60*mV
vre_e = -65*mV
tref_e = 1.5*ms

# inhib.
taumem_i = 10*ms
vT_i = -50*mV
El_i = -60*mV
vre_i = -65*mV
tref_i = .5*ms

# simtime
simtime = 5000*ms
dt_value = 0.1

#%% LIF model

start_scope()
defaultclock.dt = dt_value*ms

# excit.
eqs_e = '''
dv/dt = (-(v - El_e))/taumem_e : volt (unless refractory)
x : meter
y : meter
'''
# inhib.
eqs_i = '''
dv/dt = (-(v - El_i))/taumem_i : volt (unless refractory)
x : meter
y : meter
'''
# thalamic rate input
rate = '''
rate = input_rate : Hz
x : meter
y : meter
'''

#%% Create layer 4, layer 2/3 and Thalamus

layer_4C_e = NeuronGroup(parameters_adi.n_e_4, model = eqs_e, threshold = 'v > vT_e', reset = 'v = vre_e', refractory = tref_e, method = 'euler')
layer_4C_i = NeuronGroup(parameters_adi.n_i_4, model = eqs_i, threshold = 'v > vT_i', reset = 'v = vre_i', refractory = tref_i, method = 'euler')

layer_23_e = NeuronGroup(parameters_adi.n_e_23, model = eqs_e, threshold = 'v > vT_e', reset = 'v = vre_e', refractory = tref_e, method = 'euler')
layer_23_i = NeuronGroup(parameters_adi.n_i_23, model = eqs_i, threshold = 'v > vT_i', reset = 'v = vre_i', refractory = tref_i, method = 'euler')

thalamus = NeuronGroup(parameters_adi.n_thal, rate, threshold = 'rand() < rate*dt')

# Random initial membrane potentials
layer_4C_e.v = 'El_e + rand()*(vT_e - El_e)'
layer_4C_i.v = 'El_i + rand()*(vT_i - El_i)'
layer_23_e.v = 'El_e + rand()*(vT_e - El_e)'
layer_23_i.v = 'El_i + rand()*(vT_i - El_i)'

numbers_r_e_4 = sqrt(np.random.uniform(0, 150**2, (parameters_adi.n_e_4)))*umeter
numbers_rho_e_4 = np.random.uniform(0, 2*np.pi, (parameters_adi.n_e_4))
numbers_r_i_4 = sqrt(np.random.uniform(0, 150**2, (parameters_adi.n_i_4)))*umeter
numbers_rho_i_4 = np.random.uniform(0, 2*np.pi, (parameters_adi.n_i_4))

numbers_r_e_23 = sqrt(np.random.uniform(0, 150**2, (parameters_adi.n_e_23)))*umeter
numbers_rho_e_23 = np.random.uniform(0, 2*np.pi, (parameters_adi.n_e_23))
numbers_r_i_23 = sqrt(np.random.uniform(0, 150**2, (parameters_adi.n_i_23)))*umeter
numbers_rho_i_23 = np.random.uniform(0, 2*np.pi, (parameters_adi.n_i_23))

thalamus_r = sqrt(np.random.uniform(0, 150**2, (parameters_adi.n_thal)))*umeter
thalamus_rho = np.random.uniform(0, 2*np.pi, (parameters_adi.n_thal))

# Position neurons
for k in range(parameters_adi.n_e_4):
    # Inhibitory neurons
    if k < parameters_adi.n_i_4:
        # Polar coordinates
        # Layer 4 inhibitory
        layer_4C_i.x[k] = numbers_r_i_4[k]*cos(numbers_rho_i_4[k])
        layer_4C_i.y[k] = numbers_r_i_4[k]*sin(numbers_rho_i_4[k])
    
    if k < parameters_adi.n_i_23:
        # Layer 2/3 inhibitory
        layer_23_i.x[k] = numbers_r_i_23[k]*cos(numbers_rho_i_23[k])
        layer_23_i.y[k] = numbers_r_i_23[k]*sin(numbers_rho_i_23[k])
    
    if k < parameters_adi.n_thal:
        # Layer 4 excitatory
        thalamus.x[k] = thalamus_r[k]*cos(thalamus_rho[k])
        thalamus.y[k] = thalamus_r[k]*sin(thalamus_rho[k])

    # Layer 2/3 excitatory
    layer_23_e.x[k] = numbers_r_e_23[k]*cos(numbers_rho_e_23[k])
    layer_23_e.y[k] = numbers_r_e_23[k]*sin(numbers_rho_e_23[k])

    # Layer 4 excitatory
    layer_4C_e.x[k] = numbers_r_e_4[k]*cos(numbers_rho_e_4[k])
    layer_4C_e.y[k] = numbers_r_e_4[k]*sin(numbers_rho_e_4[k])

#%% Create layer 4, layer 2/3 and Thalamus

# layer_4C_e = NeuronGroup(Ne_4, model = eqs_e, threshold = 'v > vT_e', reset = 'v = vre_e', refractory = tref_e, method = 'euler')
# layer_4C_i = NeuronGroup(Ni_4, model = eqs_i, threshold = 'v > vT_i', reset = 'v = vre_i', refractory = tref_i, method = 'euler')

# layer_23_e = NeuronGroup(Ne_23, model = eqs_e, threshold = 'v > vT_e', reset = 'v = vre_e', refractory = tref_e, method = 'euler')
# layer_23_i = NeuronGroup(Ni_23, model = eqs_i, threshold = 'v > vT_i', reset = 'v = vre_i', refractory = tref_i, method = 'euler')

# thalamus = NeuronGroup(Nx, rate, threshold = 'rand() < rate*dt')

# # Random initial membrane potentials
# layer_4C_e.v = 'El_e + rand()*(vT_e - El_e)'
# layer_4C_i.v = 'El_i + rand()*(vT_i - El_i)'
# layer_23_e.v = 'El_e + rand()*(vT_e - El_e)'
# layer_23_i.v = 'El_i + rand()*(vT_i - El_i)'

#%% Synapses
# Connections within layer 4C (recurrent layer)
# Connect excitatory synapses
S_e_4C_e = Synapses(layer_4C_e, layer_4C_e, on_pre = 'v_post += Jee_4', delay = 0.0*ms, name = 'L4_rec_ee', namespace={'rec_4_e': parameters_adi.rec_4_e, 'sigmaRe_4': sigmaRe_4})
S_e_4C_e.connect(condition='i!= j', p='rec_4_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRe_4)**2))')
S_e_4C_e._connect_called

S_e_4C_i = Synapses(layer_4C_e, layer_4C_i, on_pre = 'v_post += Jie_4', delay = 0.0*ms, name = 'L4_rec_ei', namespace={'rec_4_e': parameters_adi.rec_4_e, 'sigmaRe_4': sigmaRe_4})
S_e_4C_i.connect(condition='i!= j', p='rec_4_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRe_4)**2))')
S_e_4C_i._connect_called

# Connect inhibitory synapses
S_i_4C_i = Synapses(layer_4C_i, layer_4C_i, on_pre = 'v_post += Jii_4', delay = 0.0*ms, name = 'L4_rec_ii', namespace={'rec_4_i': parameters_adi.rec_4_i, 'sigmaRi_4': sigmaRi_4})
S_i_4C_i.connect(condition='i!= j', p='rec_4_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRi_4)**2))')
S_i_4C_i._connect_called

S_i_4C_e = Synapses(layer_4C_i, layer_4C_e, on_pre = 'v_post += Jei_4', delay = 0.0*ms, name = 'L4_rec_ie', namespace={'rec_4_i': parameters_adi.rec_4_i, 'sigmaRi_4': sigmaRi_4})
S_i_4C_e.connect(condition='i!= j', p='rec_4_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRi_4)**2))')
S_i_4C_e._connect_called

# Connections within layer 2/3 (recurrent layer)
# Connect excitatory synapses
S_e_23_e = Synapses(layer_23_e, layer_23_e, on_pre = 'v_post += Jee_23', delay = 0.0*ms, name = 'L23_rec_ee', namespace={'rec_23_e': parameters_adi.rec_23_e, 'sigmaRe_23': sigmaRe_23})
S_e_23_e.connect(condition='i!= j', p='rec_23_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRe_23)**2))')
S_e_23_e._connect_called

S_e_23_i = Synapses(layer_23_e, layer_23_i, on_pre = 'v_post += Jie_23', delay = 0.0*ms, name = 'L23_rec_ei', namespace={'rec_23_e': parameters_adi.rec_23_e, 'sigmaRe_23': sigmaRe_23})
S_e_23_i.connect(condition='i!= j', p='rec_23_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRe_23)**2))')
S_e_23_i._connect_called

# Connect inhibitory synapses
S_i_23_i = Synapses(layer_23_i, layer_23_i, on_pre = 'v_post += Jii_23', delay = 0.0*ms, name = 'L23_rec_ii', namespace={'rec_23_i': parameters_adi.rec_23_i, 'sigmaRi_23': sigmaRi_23})
S_i_23_i.connect(condition='i!= j', p='rec_23_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRi_23)**2))')
S_i_23_i._connect_called

S_i_23_e = Synapses(layer_23_i, layer_23_e, on_pre = 'v_post += Jei_23', delay = 0.0*ms, name = 'L23_rec_ie', namespace={'rec_23_i': parameters_adi.rec_23_i, 'sigmaRi_23': sigmaRi_23})
S_i_23_e.connect(condition='i!= j', p='rec_23_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaRi_23)**2))')
S_i_23_e._connect_called

# Connections from thalamus to layer 4
# Connect to excitatory neurons
S_th_e = Synapses(thalamus, layer_4C_e, on_pre = 'v_post += Jex_4', delay = 0.0*ms, name = 'thal_ffwd_e', namespace={'ffwd_4_e': parameters_adi.ffwd_4_e, 'sigmaffwd_4': sigmaffwd_4})
S_th_e.connect(p='ffwd_4_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaffwd_4)**2))')
S_th_e._connect_called

# Connect to inhibitory neurons
S_th_i = Synapses(thalamus, layer_4C_i, on_pre = 'v_post += Jix_4', delay = 0.0*ms, name = 'thal_ffwd_i', namespace={'ffwd_4_i': parameters_adi.ffwd_4_i, 'sigmaffwd_4': sigmaffwd_4})
S_th_i.connect(p='ffwd_4_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaffwd_4)**2))')
S_th_i._connect_called

# Connect layers 4 and 2/3
# Excitatory synapses
S_e_e = Synapses(layer_4C_e, layer_23_e, on_pre = 'v_post += Je_4_23', delay = 0.0*ms, name = 'L423_ffwd_ee', namespace={'ffwd_23_e': parameters_adi.ffwd_23_e, 'sigmaffwd_23': sigmaffwd_23})
S_e_e.connect(condition='i!= j', p='ffwd_23_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaffwd_23)**2))')
S_e_e._connect_called

S_e_i = Synapses(layer_4C_e, layer_23_i, on_pre = 'v_post += Je_4_23', delay = 0.0*ms, name = 'L423_ffwd_ei', namespace={'ffwd_23_e': parameters_adi.ffwd_23_e, 'sigmaffwd_23': sigmaffwd_23})
S_e_i.connect(condition='i!= j', p='ffwd_23_e*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaffwd_23)**2))')
S_e_i._connect_called

# Connect inhibitory synapses
S_i_i = Synapses(layer_4C_i, layer_23_i, on_pre = 'v_post += Ji_4_23', delay = 0.0*ms, name = 'L423_ffwd_ii', namespace={'ffwd_23_i': parameters_adi.ffwd_23_i, 'sigmaffwd_23': sigmaffwd_23})
S_i_i.connect(condition='i!= j', p='ffwd_23_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2) / (2*(sigmaffwd_23)**2))')
S_i_i._connect_called

S_i_e = Synapses(layer_4C_i, layer_23_e, on_pre = 'v_post += Ji_4_23', delay = 0.0*ms, name = 'L423_ffwd_ie', namespace={'ffwd_23_i': parameters_adi.ffwd_23_i, 'sigmaffwd_23': sigmaffwd_23})
S_i_e.connect(condition='i!= j', p='ffwd_23_i*exp(-((x_pre - x_post)**2 + (y_pre - y_post)**2)/(2*(sigmaffwd_23)**2))')
S_i_e._connect_called

# how objects are scheduled to be executed during the simulation
print(scheduling_summary())
print(magic_network.schedule)

#%% Store network data

spikemon_4_e = SpikeMonitor(layer_4C_e)
spikemon_4_i = SpikeMonitor(layer_4C_i)
spikemon_23_e = SpikeMonitor(layer_23_e)
spikemon_23_i = SpikeMonitor(layer_23_i)

store() # stores initial state of the network

#%% Run network simulation for different inhibitory strengths

BrianLogger.log_level_debug()

# input frequency and inhibitory strength
gvariable = linspace(0,10,21)
input_r = linspace(0,70,8)

count_g = 0

# Inhibitory strength
for g in gvariable:
    
    # Define g
    g = g
    print(g)

    # synaptic strength
    # L4
    Jee_4 = 0.18*mV
    Jei_4 = -g*1.8*mV # original scaling = -g * 0.18 mV
    Jie_4 = 0.54*mV
    Jii_4 = -g*1.8*mV # original scaling = -g * 0.18 mV
    # L2/3
    Jee_23 = 0.18*mV
    Jei_23 = -g*1.8*mV # original scaling = -g * 0.18 mV
    Jie_23 = 0.54*mV
    Jii_23 = -g*1.8*mV # original scaling = -g * 0.18 mV
    # Thalamus --> layer 4
    Jex_4 = 0.50*mV # original value = 0.54 mV
    Jix_4 = 0.50*mV # original value = 0.54 mV
    # Between layers
    Je_4_23 = 0.18*mV
    Ji_4_23 = 0.18*mV
    
    # Input rate
    r = 0 # entered manually after each for loop
    
    # Define input rate
    input_rate = r*Hz

    # Restore network
    restore() # puts state of the system to original
    
    # Run simulation
    run(simtime, report='text', profile = True)
    
    profiling_summary()
    
    # Store all data
    # L4 -spikes (indices-i, time-t)
    np.save('Spikestest_4e_i_g=%g_r=%g' %(g,r), spikemon_4_e.i)
    np.save('Spikestest_4e_t_g=%g_r=%g' %(g,r), spikemon_4_e.t)
    np.save('Spikestest_4i_i_g=%g_r=%g' %(g,r), spikemon_4_i.i)
    np.save('Spikestest_4i_t_g=%g_r=%g' %(g,r), spikemon_4_i.t)
    # spike trains
    np.save('Spiketraintest_4e_g=%g_r=%g' %(g,r), [spikemon_4_e.spike_trains()])
    np.save('Spiketraintest_4i_g=%g_r=%g' %(g,r), [spikemon_4_i.spike_trains()])

    # L2/3
    np.save('Spikestest_23e_i_g=%g_r=%g' %(g,r), spikemon_23_e.i)
    np.save('Spikestest_23e_t_g=%g_r=%g' %(g,r), spikemon_23_e.t)
    np.save('Spikestest_23i_i_g=%g_r=%g' %(g,r), spikemon_23_i.i)
    np.save('Spikestest_23i_t_g=%g_r=%g' %(g,r), spikemon_23_i.t)
    # spike trains
    np.save('Spiketraintest_23i_g=%g_r=%g' %(g,r), [spikemon_23_i.spike_trains()])
    np.save('Spiketraintest_23e_g=%g_r=%g' %(g,r), [spikemon_23_e.spike_trains()])
