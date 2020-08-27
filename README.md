# Balanced-state-in-barrel-cortex

network_sim.py  
                      -- runs the simulation for the model  
                      -- requires parameters_sim.py script for loading neuron numbers and connection probabilities  

parameters_adi.py  
                      -- asks user input for network configuration (r) and corresponding excitatory (e) and inhibitory (i) neuron numbers in layer 2/3 (L2/3) and layer 4 (L4)   
                            r = 0 (7.35 - L4 and 5.25 - L2/3)  
                                  L4   : n_e_4 = 1374 ; n_i_4 = 187  
                                  L2/3 : n_e_23 = 2232 ; n_i_23 = 425     
                            r = 3  
                                  L4   : n_e_4 = 1374 ; n_i_4 = 458  
                                  L2/3 : n_e_23 = 2232 ; n_i_23 = 744  
                            r = 4  
                                  L4   : n_e_4 = 1374 ; n_i_4 = 343  
                                  L2/3 : n_e_23 = 2232 ; n_i_23 = 558  
                            r = 5  
                                  L4   : n_e_4 = 1374 ; n_i_4 = 274  
                                  L2/3 : n_e_23 = 2232 ; n_i_23 = 446  
                                  
  firingfreq_analysis.py  
  calculates the firing rates of the balanced network model  
    
  cv_analysis.py  
  calculates the coefficient of variation of the balanced network model  
