# -*- coding: utf-8 -*-
"""
Balanced state simulation Parameters
Neuron numbers
Network upscaling values

"""
#%% Neurons in each layer - check README.md for configuration values

def neuron_num():
    global config, n_e_4, n_i_4, n_e_23, n_i_23, n_thal
    
    # network configurations: 0 (baseline),3,4,5 ; baseline L4- 7.35 , L2/3 - 5.25
    config = input('Enter the network simulation: ')  # config = 0 = Baseline
    # input neurons in L4
    n_e_4 = 1374
    n_i_4 = input('L4: number of INHIBITORY neurons: ')
    # input neurons in L2/3
    n_e_23 = 2232
    n_i_23 = input('L2/3: number of INHIBITORY neurons: ')
    n_thal = 200 # thalamic neurons
    
    config = int(config)
    n_i_4  = int(n_i_4)
    n_i_23 = int(n_i_23)

    print('Network configuration : ' + str(config))
    print('Num. of excitatory neurons in L4: ' + str(n_e_4))
    print('Num. of inhibitory neurons in L4: ' + str(n_i_4))
    print('Num. of excitatory neurons in L2/3: ' + str(n_e_23))
    print('Num. of inhibitory neurons in L2/3: ' + str(n_i_23))
    print('Num. of thalamic neurons: ' + str(n_thal))
    
#%% Upscale probability to compensate for smaller network size

def prob_vals():
    global rec_4_e, rec_4_i, rec_23_e, rec_23_i, ffwd_4_e, ffwd_4_i, ffwd_23_e, ffwd_23_i
    
    ### Recurrent
     # L4 - excit.
    Nnieuw, Nold = n_e_4, 40000
    Eold =  0.05*Nold #0.05
    rec_4_e = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw
     # L4 - inihib.
    Nnieuw, Nold = n_i_4, 10000
    Eold = 0.05*Nold #0.05
    rec_4_i = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw

     # L23- excit.
    Nnieuw, Nold = n_e_23, 40000
    Eold = 0.05*Nold #0.05
    rec_23_e = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw
     # L23- inhib.
    Nnieuw, Nold = n_i_23, 10000
    Eold = 0.05*Nold #0.05
    rec_23_i = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw

    #### Feed-forward
    ## thalamus-L4
     # excit.
    Nnieuw, Nold = n_thal, 5625
    Eold = 0.25*Nold
    ffwd_4_e = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw
     # inihib.
    Nnieuw, Nold = n_thal, 5625
    Eold = 0.08*Nold # 0.08
    ffwd_4_i = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw

    # L4 - L2/3
     # excit.
    Nnieuw, Nold = n_e_4, 40000
    Eold = 0.25*Nold
    ffwd_23_e = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw
     # inhib.
    Nnieuw, Nold = n_i_4, 10000
    Eold = 0.08*Nold
    ffwd_23_i = (((1/Eold) - (1/Nold) + (1/Nnieuw))**-1)/Nnieuw