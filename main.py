
import numpy as np

import likelihood as lk
import Chain as c
import MCsampler
import plot_mc

if __name__ == '__main__':
    runiterate = False
    sample_num = 50000    #total number of sample drawn
    cov_ite_num = 2    #total iterate number to get a proper covariant matrix
    parms = ['Omega_m','Omega_lambda','H0','M_nuisance','Omega_k']
    initial_condition = {'Omega_m': 0.3, 'Omega_lambda': 0.7, 'H0': 74.0,
                          'M_nuisance': -19.23, 'Omega_k': 0.0}
    priors = {'Omega_m': 0.0,
              'Omega_lambda': 0.0,
              'H0': 0.0,
              'M_nuisance': 0.0,
              'Omega_k': 0.0} 
    cov = np.loadtxt("cov.txt")            
    sam = MCsampler.MCMC(initial_condition,priors)
    
    #iterate to have a convergent covariant matrix for the 4 parameters
    if runiterate == True:
        for _ in range(cov_ite_num):
            sam.accepted = 0
            for _ in range(sample_num):
                sam.add_to_chain()
            cha = sam.return_chain()

            #calculate covariant matrix from chain data
            cov = cha.cov_cal()

            #reset alpha to be a new value. Starting value is 0.1
            sam.cov_alpha = 100.0
            sam.learncov(cov)
            sam.reset_chain()

        np.savetxt("cov.txt",cov)

    else:
        sam.learncov(cov)
        for _ in range(sample_num):
            sam.add_to_chain()
        cha = sam.return_chain()
    #print(cha.samples)

    #"""All the plots
    
    #samples = cha.samples #plot the MCMC data
    samples = c.simulator(1000)  #plot data from simulator, just a test
    
    plot_mc.mcmc_result(cha.samples) #check all the parameters
    
    plot_mc.trace_plot(cha.samples) #trace plot as a sanity check
    
    omega_m, omega_lambda, prob_nosys, quantile_nosys = plot_mc.samples_process(samples=cha.samples, x_range=[0, 1.6], y_range=[0, 2.5], xbin=30, ybin=40)  #fig 18
    #omega_m, omega_lambda, prob_sys, quantile_sys = plot_mc.samples_process(samples=cha.samples, x_range=[0, 1.6], y_range = [0, 2.5], xbin=30, ybin=40) #fig 18
    plot_mc.fig18(omega_m, omega_lambda, prob_nosys=prob_nosys, prob_sys=[], quantile_nosys=quantile_nosys)  #fig 18
    #plot_mc.fig18(omega_m, omega_lambda, prob_nosys=prob_nosys, prob_sys=prob_sys,
    #              quantile_nosys=quantile_nosys, quantile_sys=quantile_sys)  #final fig 18
    #"""

    acceptance_rate = (1.0*sam.accepted)/(1.0*sample_num)
    print("acceptance rate = ",acceptance_rate)
    #plot the data stored in chain, to be finished
    #cha.plot()
