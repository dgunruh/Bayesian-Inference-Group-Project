
import numpy as np

import likelihood as lk
import Chain as c
import MCsampler

if __name__ == '__main__':
    runiterate = True
    sample_num = 1000    #total number of sample drawn
    cov_ite_num = 1     #total iterate number to get a proper covariant matrix
    parms = ['Omega_m','Omega_lambda','H0','M_nuisance','Omega_k']
    initial_condition = {'Omega_m': 0.30, 'Omega_lambda': 0.7, 'H0': 72.0,
                          'M_nuisance': -19.0, 'Omega_k': 0.0}
    priors = {'Omega_m': 0.0,
              'Omega_lambda': 0.0,
              'H0': 0.0,
              'M_nuisance': 0.042,
              'Omega_k': 0.0} 
    #cov = np.loadtxt("cov.txt")            
    sam = MCsampler.MCMC(initial_condition,priors)
    
    #iterate to have a convergent covariant matrix for the 4 parameters
    if runiterate == True:
        for _ in range(cov_ite_num):
            sam.accepted = 0
            for _ in range(sample_num):
                sam.add_to_chain()
            cha = sam.return_chain()
            #calculate covariant matrxi from chain data, to be finished
            cov = cha.cov_cal()
            sam.learncov(cov)
            sam.reset_chain()

        np.savetxt("cov.txt",cov)

    else:
        sam.learncov(cov)
        for _ in range(sample_num):
            sam.add_to_chain()
        cha = sam.return_chain()
    print(cha.samples)
    acceptance_rate = (1.0*sam.accepted)/(1.0*sample_num)
    print("acceptance rate = ",acceptance_rate)
    #plot the data stored in chain, to be finished
    #cha.plot()
