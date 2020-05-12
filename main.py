
import numpy as np

import likelihood as lk
import Chain as c
import MCsampler
import plot_mc
import os

def prep_fig18(systematic, det_cov=False):
    '''
    A function which creates a replica of Fig. 18 from the paper.

    Inputs:
    ------------
    systematic: Boolean
        Determines whether the plot will include the systematic error,
        or just use the statistical error. 

    det_cov: Boolean
        Determines whether the program needs to determine the covariance
        or not. If not, it is assumed that it was already calculated, and 
        the program will load it. 

    Returns:
    ------------
    omega_m : array like;
    grid for omega_m in the final plot

    omega_lambda : array like;
    grid for omega_lambda in the final plot

    prob : array like
    normalized 2D probability distribution

    quantile : array like
    the mean value and 3 sigma value for omega_m and omega_lambda
    '''
    sample_num = 10000    #total number of samples drawn
    cov_ite_num = 2    #total number of iterations to get a proper covariant matrix

    #parms = ['Omega_m','Omega_lambda','H0','M_nuisance','Omega_k']
    initial_condition = {'Omega_m': 0.3, 'Omega_lambda': 0.7, 'H0': 74.0,
                          'M_nuisance': -19.23, 'Omega_k': 0.0}
    #All priors will be uniform
    priors = {'Omega_m': 0.0,
              'Omega_lambda': 0.0,
              'H0': 0.0,
              'M_nuisance': 0.0,
              'Omega_k': 0.0} 

    #Determines the standard deviation of the distributions used
    #by the generating function to get new parameter values
    scaling = [0.1, 0.1, 1.0, 0.1, 0.1]
    sam = MCsampler.MCMC(initial_condition,priors, scaling, systematic, True)

    #iterate to have a convergent covariant matrix for the 4 parameters
    if det_cov:
        for _ in range(cov_ite_num):
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
        try:
            cov = np.loadtxt("cov.txt")
            sam.cov_alpha = 100.0
            sam.learncov(cov)
        except OSError:
            print("Covariance matrix not found. Running with default matrix and cov_alpha.")

        for _ in range(sample_num):
            sam.add_to_chain()
        cha = sam.return_chain()

    if systematic:
        omega_m, omega_lambda, prob, quantile = plot_mc.samples_process(samples=cha.samples, x_range=[0, 1.6], y_range=[0, 2.5], xbin=50, ybin=50)
    else:
        omega_m, omega_lambda, prob, quantile = plot_mc.samples_process(samples=cha.samples, x_range=[0, 1.6], y_range=[0, 2.5], xbin=50, ybin=50)
        
    return omega_m, omega_lambda, prob, quantile

def create_fig18(det_cov=False):
    """
    This function is used to create figure 18.

    Inputs:
    ---------
    det_cov: Boolean
        Determines whether the program needs to determine the 
        generating function covariance. If not, it is assumed
        that one was previously generated. Either way, only
        the first function run will need it. 
    """
    #Get non-systematic probabilities
    omega_m, omega_lambda, prob_nosys, quantile_nosys = prep_fig18(False, det_cov)
    
    #Get systematic probabilities
    omega_m, omega_lambda, prob_sys, quantile_sys = prep_fig18(True, False)

    plot_mc.fig18(omega_m, omega_lambda, prob_nosys=prob_nosys, prob_sys=prob_sys,
                quantile_nosys=quantile_nosys, quantile_sys=quantile_sys, savepath=os.getcwd() + '/results/fig18.png')

def create_H_posterior(systematic, det_cov=False):
    '''
    A function which creates the posterior distribution of H0.

    Inputs:
    ------------
    systematic: Boolean
        Determines whether the plot will include the systematic error,
        or just use the statistical error. 

    det_cov: Boolean
        Determines whether the program needs to determine the covariance
        or not. If not, it is assumed that it was already calculated, and 
        the program will load it. 
    '''
    sample_num = 100000    #total number of samples drawn
    cov_ite_num = 2    #total number of iterations to get a proper covariant matrix

    #parms = ['Omega_m','Omega_lambda','H0','M_nuisance','Omega_k']
    initial_condition = {'Omega_m': 0.3, 'Omega_lambda': 0.7, 'H0': 74.0,
                          'M_nuisance': -19.23, 'Omega_k': 0.0}
    
    #M prior is not uniform, but instead has std .042
    priors = {'Omega_m': 0.0,
              'Omega_lambda': 0.0,
              'H0': 0.0,
              'M_nuisance': 0.042,
              'Omega_k': 0.0} 

    #Determines the standard deviation of the distributions used
    #by the generating function to get new parameter values
    scaling = [0.1, 0.1, 1.0, 0.042, 0.1]
    sam = MCsampler.MCMC(initial_condition,priors, scaling, systematic, False)

    #iterate to have a convergent covariant matrix for the 4 parameters
    if det_cov:
        for _ in range(cov_ite_num):
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
        try:
            cov = np.loadtxt("cov.txt")
            sam.cov_alpha = 100.0
            sam.learncov(cov)
        except OSError:
            print("Covariance matrix not found. Running with default matrix and cov_alpha.")

        for _ in range(sample_num):
            sam.add_to_chain()
        cha = sam.return_chain()

    plot_mc.mcmc_result(cha.samples, savepath=os.getcwd() + '/results/mcmc.png') #check all the parameters
    plot_mc.trace_plot(cha.samples, savepath=os.getcwd() + '/results/trace.png') #trace plot as a sanity check
    plot_mc.post_prob(cha.samples, element='H0',
                      xbin=50, savepath=os.getcwd() + '/results/post_prob_H0.png')

if __name__ == '__main__':
    #Create Fig. 18, including containing only statistical error, and including
    # both statistical and systematic error
    create_fig18(False)

    #Create posterior distribution of H assuming M has a prior of .042
    create_H_posterior(True)