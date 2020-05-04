
import numpy as np

import likelihood as lk
import Chain as c
import MCsampler

if __name__ == '__main__':
    sample_num = 100    #total number of sample drawn
    cov_ite_num = 10     #total iterate number to get a proper covariant matrix
    parms = ['Omega_m','Omega_lambda','H0','M_nuisance','Omega_k']
    initial_condition = {'Omega_m': 0.30, 'Omega_lambda': 0.7, 'H0': 72.0,
                          'M_nuisance': -19.0, 'Omega_k': 0.0}
    priors = {'Omega_m': 0.0,
              'Omega_lambda': 0.0,
              'H0': 0.0,
              'M_nuisance': 0.042,
              'Omega_k': 0.0}             
    cov = np.identity(5)   #The initial covariant matrix
    sam = MCsampler.MCMC(initial_condition,priors)
    
    #iterate to have a convergent covariant matrix for the 4 parameters
    for _ in range(cov_ite_num):
        cha = c.Chain(parms)
        for step in range(sample_num):
            #draw sample using sampler, to be finished
            sample = sam.take_step()
            cha.add_sample(sample)
        
        #calculate covariant matrxi from chain data, to be finished
        cov = cha.cov_cal()
        sam.learncov(cov)
    
    print(cha.samples)
    #plot the data stored in chain, to be finished
    #cha.plot()
