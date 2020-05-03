#!/usr/bin/env python
import os
from decimal import Decimal, getcontext

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from scipy.integrate import quad

NEED_PARAMS = ['Omega_m', 'Omega_lambda', 'H0']
NEED_NUISANCE = ['M_nuisance']

class LK:

    def __init__(self, dat_dir=os.getcwd() + '/Binned_data/'):

        """
        likelihood class


        Params
        -----------
        data_dir: location where pantheon data is found

        Attributes
        ----------

        data: dict
        A dictionary containing all the data from the dataset

        stat_err: [floats]
        List containing all the statistical error read in from the dataset

        sys_err: [floats]
        List containing all the systematic errors read in from the datset

        m_B: [floats]
        List containing apparent magnitudes read in from the dataset

        z: [floats]
        List containing redshifts read in from the dataset

        tot_err: [floats]
        List containing total error (sum of stat_err and sys_err)

        usage example:
        -----------
        import likelihood
        like = likelihood.LK()
        pars = {'Omega_m': 0.30, 'Omega_lambda': 0.7, 'H0': 72.0,
                'M_nuisance': -19.0, 'Omega_k': 0.0}
        log_likelihood, parans = like.likelihood_cal(pars)
        """



        self.data = self.loading_data(dat_dir)
        self.stat_err = self.data['stat_err']
        self.sys_err = self.data['sys_err']
        self.m_B = self.data['m_B']
        self.z = self.data['z']
        self.tot_err = self.sys_err + self.stat_err #For the red contour

    def loading_data(self, dat_dir, show=False):
        """
        read Pantheon data from Binned_data directory
        -----------
        usage example:
        stat_err, sys_err = loading_data(dat_dir=os.getcwd() + '/Binned_data/')

        Parameters:
        -----------
        dat_dir : string;
        Point to the directory where the data files are stored

        show : bool;
        Plot covariance matrix if needed

        Returns:
        --------
        data : dictionary;
        return a dictionary that contains stat_err, sys_err, z and m_B
        """

        Pantheon_data = ascii.read(dat_dir+'lcparam_DS17f.txt', names=['name', 'zcmb', 'zhel', 'dz',
                                                            'mb', 'dmb', 'x1', 'dx1',
                                                            'color', 'dcolor', '3rdvar', 'd3rdvar',
                                                            'cov_m_s', 'cov_m_c', 'cov_s_c', 'set',
                                                            'ra', 'dec'])

        #read the redshift, apparent magnitude and statistic error
        z = Pantheon_data['zcmb']
        m_B = Pantheon_data['mb']
        stat_err = np.diag(Pantheon_data['dmb'])**2 #convert the array to a dignal matrix
        stat_err = np.matrix(stat_err)

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
        sys_err = np.matrix(sys_err)

        if show is True: #plot the covariance matrix if needed
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,6.18))
            imgplot = plt.imshow(sys_err, cmap='bone', vmin=-0.001, vmax=0.001)
            ax1.set_xticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
            ax1.set_yticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
            ax1.set_xlabel('z')
            ax1.set_ylabel('z')
            fig.colorbar(imgplot)
            plt.show()
        return {'stat_err':stat_err, 'sys_err':sys_err, 'z':z, 'm_B':m_B}

    def likelihood_cal(self, pars={}, ifsys=True):
        """
        Calculate likelihood for the parameters from sampler

        Parameters:
        -----------
        par : dictionary {string: float};
        dictionary of parameters and their values

        ifsys: bool;
        calculate likelihood with and without systematic error

        Returns:
        --------
        likelihood, pars : float, dict;
        return the log-likelihood to the sampler, as well as parameters
        """
        _pars = pars.copy()
        #print(pars.keys())
        for param in NEED_PARAMS: #check for all neede params
            assert param in _pars.keys(), 'Error: likelihood calculation'\
                                         ' requires a value for parameter {}'.format(param)

        for param in NEED_NUISANCE: #check for all needed nuisance parameters
            assert param in _pars.keys(), 'Error: Likelihood requires nuisance'\
                                          ' parameter {}'.format(param)

        for k,v in _pars.items(): #check that value are valid
            assert isinstance(v, float), 'Error: value of paramater {} is not a float'.format(k)

        model_mus = self.compute_model(_pars) + pars.get('M_nuisance')
        delta_mu = self.m_B - model_mus
        delta_mu = np.matrix(delta_mu)

        if(ifsys): #loads the error
            error = self.tot_err
        else:
            error = self.stat_err

        #Claculate Chi2 according to Equation 8
        Chi2 = np.float(delta_mu * np.linalg.inv(error) * np.transpose(delta_mu))
        if np.isnan(Chi2):
            Chi2 = 1e10 #give a very large value here

        return -Chi2/2, _pars #returns the log-likelihood


    def compute_model(self, pars):
        '''
        Computes the model values  given a set of parameters.
        The validity of the input parameters is
        checked by the calling function (likelihood_cal)

        Parameters:
        -----------
        par : dictionary {string: float};
        dictionary of parameters and their values

        Returns:
        ------------
        mus: [float]
        Values of the distance moduli for the given model
        '''

        if not 'Omega_k' in pars.keys(): #calculates omegak, if needed
            omega_k = 1 - pars.get('Omega_lambda') - pars.get('Omega_m')
            if np.abs(omega_k) < (10**-7):
                omega_k = 0 # compensating for floating point arithmetic errors
            pars.update({'Omega_k': omega_k})

        lds = self.luminosity_distances(pars)
        mus = 25 + 5*np.log10(lds)
        #luminosity distances are in units of megaparsecs
        return mus



    def integrand(self, z, pars):
        '''
        Returns the value of the integrand E(z) that is used in all luminosity
        distance calculations

        Parameters:
        -----------
        z: float
        redshift value

        pars: dict {string: float}
        Values of cosmological parameters

        Returns:
        --------
        The value of the integrand for the given redshift and cosmological
        parameters

        '''

        assert z > 0, 'Error: Invalid value for redshift passed. z must be > 0'
        sum = pars['Omega_m']*((1+z)**3) + pars['Omega_lambda'] + pars['Omega_k']*((1+z)**2)
        return 1/np.sqrt(sum)

    def luminosity_distances(self, pars):
        '''
        Calculates the luminosity distances for a given model, in units of mpc

        Parameters:
        -----------
        pars: dict {string: float}
        Values of cosmological parameters

        Returns:
        --------
        lds: [floats]
        Calculated luminosity distances in units of megaparsecs
        '''
        num_points = len(self.z)
        lds = np.zeros(num_points)

        integral_val = quad(self.integrand, 0, self.z[0], args=(pars,))[0]
        lds[0] = self.luminosity_delegate(self.z[0], integral_val, pars)

        for i in range(1, num_points):
            integral_val += quad(self.integrand, self.z[i-1], self.z[i], args=(pars,))[0]
            lds[i] = self.luminosity_delegate(self.z[i], integral_val, pars)
            # Here we avoid excess calulation by integrating over redshift
            # ranges and summing
        return lds

    def luminosity_delegate(self, z, integral_val, pars):
        '''
        Helper function for calculating the luminosity distnaces

        Parameters:
        -----------
        z: float
        Redshift value

        integral_val: float
        value of the integral E(z) for the given redshift

        pars: dict
        dictionary of cosmological parameters and their values

        Returns:
        --------
        luminosity distance: float
        Calculated luminosity distances in units of megaparsecs
        '''
        d_hubble = self.hubble_distance(pars.get('H0'))
        Omega_k = pars.get('Omega_k')
        if Omega_k > 0:
            return (1+z)*d_hubble*np.sinh(np.sqrt(Omega_k)*integral_val)/np.sqrt(Omega_k)
        elif Omega_k < 0:
            return (1+z)*d_hubble*np.sin(np.sqrt(np.abs(Omega_k))*integral_val)/np.sqrt(np.abs(Omega_k))
        else:
            return (1+z)*d_hubble*integral_val



    def hubble_distance(self, H0):
        '''
        Calculates the hubble distance in unites of mpc for a given H0

        Parameters:
        -----------
        H0: float
        Value of the hubble constant

        Returns:
        -----------
        hubble distance: float
        value of the hubble distance
        '''

        c = 3*10**5
        return c/H0



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lk = LK()
    params = {'Omega_m': 0.3, 'Omega_lambda': 0.7, 'H0': 72.0, 'M_nuisance': -19.0, 'Omega_k': 0.0}
    chi2, pars = lk.likelihood_cal(params)
    print(chi2)
