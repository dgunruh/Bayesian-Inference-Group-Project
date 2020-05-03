import math
import numpy as np

import likelihood as lk

class Sampler(object):

    def __init__(self,init_condition=[0.3,0.7,72.0,-19.0,0.0]):
        '''
        sampler class

        Parameters:
        ----------
        initial_condition: The sarting point of the sampler
        In the order of Omega_m, Omega_Lambda, H0, M_nuisance, Omega_k

        Attributes:
        ----------
        current:[float]
        The current sample value

        candidate:[float]
        The candidate value that could be a future current

        usage exampe:
        ----------
        import sampler
        initial_condition = [1,1,1,1,1]
        cov = np.identity()
        sam = sampler.Sampler(initial_condition)
        sample = sam.draw_sample(cov)

        '''
        self.current = init_condition
        self.candidate = []
        self.cov = np.identity(4)


    def gen_func(self,pars=[]):
        '''
        generating function

        Parameters:
        ----------
        pars:[float]
        A list of value in the order of Omega_m, Omega_Lambda, H0, M_nuisance, Omega_k

        Returns:
        ----------
        nonnorm_pdf:float
        A non-normalized generating function
        '''
        index = 0
        for i in range(4):
            for j in range(4):
                index = index + (pars[i] - self.current[i])*self.cov[i][j]*(pars[j] - self.current[j])
        
        nonnorm_pdf = math.exp(-1*index)

        return nonnorm_pdf

    def draw_candidate(self):
        '''
        Sampling from the generating fucntion to genrare a candidate.
        A real customized 5d random sampling would be wild, this is just a work around that
        makes sense to me. PLEASE let me know any possible improvement.

        '''
     
        deny = True
        steps = 0
        while deny:
            steps = steps + 1
            assert steps < 100, 'Error,value is too small to judge'
            potential_candidate = []
            for i in range(5):
                x = np.random.normal(loc=self.current[i])
                potential_candidate.append(x)
            value = self.gen_func(potential_candidate)
            judger = np.random.random_sample()
            if judger < value:
                deny = False
        self.candidate = potential_candidate

    def learncov(self,cov):
        self.cov = cov

    def draw_sample(self):
        #To be done by Davis
        something = [1.0,1.0,1.0,1.0,1.0]
        self.current = something
        current_dict = {'Omega_m':1.0, 'Omega_Lambda':1.0,
                        'H0':1.0, 'M_nuisance':1.0, 'Omega_k':1.0}

        return current_dict
        






