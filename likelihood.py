import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii

def loading_data(dat_dir, show=False):
    """ 
    read Pantheon data from Binned_data directory
    -----------
    usage example:
    stat_err, sys_err = loading_data(dat_dir=os.getcwd() + '/Binned_data/')

    Parameters:	
    -----------
    dat_dir : string
    Point to the directory where the data files are stored

    show : bool
    Plot covariance matrix if needed

    Returns:	
    --------
    stat_err : array_like
    return the stat_err as a diagnal matrix
    sys_err: array_like
    return the systematic covariance matrix
    """
    Pantheon_data = ascii.read(dat_dir+'lcparam_DS17f.txt', names=['name', 'zcmb', 'zhel', 'dz', 
                                                          'mb', 'dmb', 'x1', 'dx1', 
                                                          'color', 'dcolor', '3rdvar', 'd3rdvar', 
                                                          'cov_m_s', 'cov_m_c', 'cov_s_c', 'set', 
                                                          'ra', 'dec'])

    #read the redshift, apparent magnitude and statistic error
    z = Pantheon_data['zcmb']
    m_B = Pantheon_data['mb']
    stat_err = np.diag(Pantheon_data['dmb']) #convert the array to a dignal matrix

    #read the systematic covariance matrix from sys_DS17f.txt
    error_file = ascii.read(dat_dir+'sys_DS17f.txt')
    error_file = error_file['40'] #where 40 is the first line of the file and indicates the size of the matrix
    sys_err = []
    cnt = 0 
    line = []
    for i in np.arange(np.size(error_file)):
        cnt += 1
        line.append(error_file[i])
        if cnt % 40 == 0:
            cnt = 0
            if len(line) > 0:
                sys_err.append(line)
                line = []

    if show is True: #plot the covariance matrix if needed
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,6.18))
        imgplot = plt.imshow(sys_err, cmap='bone', vmin=-0.001, vmax=0.001)   
        ax1.set_xticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
        ax1.set_yticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
        ax1.set_xlabel('z')
        ax1.set_ylabel('z')
        fig.colorbar(imgplot)
        plt.show()
    return stat_err, sys_err

def likelihood_cal(par=[], dat_dir=os.getcwd() + '/Binned_data/', ifsys=True):
    """ 
    Calculate likelihood for the parameters from sampler

    Parameters:	
    -----------
    par : dictionary
    parameters from sampler
    dat_dir : string
    Point to the directory where the data files are stored
    ifsys: bool
    calculate likelihood with and without systematic error

    Returns:	
    --------
    likelihood : float
    return the log-likelihood to the sampler
    """
    stat_err, sys_err = loading_data(dat_dir)
    stat_err = np.matrix(stat_err)
    sys_err = np.matrix(sys_err)
    if ifsys is True: #For the red contour
        tot_err = sys_err + stat_err
    else: # For the grey contour
        tot_err = stat_err
        
    """
    below is a test, we need to work on the code to calculate delta_mu from par
    """
    delta_mu = np.random.uniform(0, 1, 40)  #fake delta_mu values for test
    delta_mu = np.matrix(delta_mu)

    #Claculate Chi2 according to Equation 8
    Chi2 = np.float(delta_mu * np.linalg.inv(tot_err) * np.transpose(delta_mu))
    
    #temporary output for test, will be deleted
    return stat_err, sys_err, tot_err, Chi2
    #return log_likelihood
    