
import numpy as np

import likelihood as lk
import Chain as c
import sampler

if __name__ == '__main__':
    sample_num = 1000    #total number of sample drawn
    cov_ite_num = 10     #total iterate number to get a proper covariant matrix
    initial_condition = {'Omega_m': 0.30, 'Omega_lambda': 0.7, 'H0': 72.0,
                          'M_nuisance': -19.0, 'Omega_k': 0.0}             
    initial_condition = list(initial_condition.values())
    cov = np.identity(5)   #The initial covariant matrix
    sam = sampler.Sampler(initial_condition)
    
    #iterate to have a convergent covariant matrix for the 4 parameters
    for _ in range(cov_ite_num):
        cha = c.Chain()
        for step in range(sample_num):
            #draw sample using sampler, to be finished
            sample = sam.draw_sample()
            cha.add_sample()
        
        #calculate covariant matrxi from chain data, to be finished
        cov = cha.cov_cal()
        sam.learncov(cov)

    #plot the data stored in chain, to be finished
    cha.plot()
