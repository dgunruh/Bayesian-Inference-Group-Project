import math
import numpy as np
import scipy.stats as stat
import likelihood
import Chain


class MCMC(object):
    def __init__(self, initial_condition, param_priors, step_sigmas, systematic_error=False, degeneracy=True):
        """
        sampler class

        Parameters:
        ----------
        initial_condition: Dictionary{String: float}
            A dictionary defining the starting parameter values of the sampler
        
        param_priors: Dictionary{String: String}
            A dictionary defining the prior distributions of the input parameters

        step_sigma: [floats]
            A list containing the parameter step standard deviations. The parameter
            standard deviations determine how far each step of the generating function
            is allowed to attempt to travel

        systematic_error: Boolean
            Tells the sampler whether to calculate the likelihood 
            with or without using the systematic error

        degeneracy: Boolean
            Tells the sampler whether the degenerate relationship between
            M and H0 should be preserved or discarded.

        Attributes:
        ----------
        chain: Chain
            An instance of the Chain class which contains the
            Markov Chain which the sampler is creating
            
        LK: likelihood
            An instance of the likelihood class which will
            return the log-likelihood of each set of parameter
            values
            
        sys_error: boolean
            The storage of the systematic_error input

        initial_params: Dictionary{String: float}
            The dictionary mapping the starting parameter values
            to the parameters in question. Stored for the purpose
            of calculating prior probability densities.
        
        current_params: Dictionary{String: float}
            The dictionary mapping the current parameter
            values to the parameters in question

        current_prior_p: float
            The combined prior probability of the current parameter values

        candidate_params: Dictionary{string: float}
            The dictionary mapping the candidate parameter
            values to the parameters in question.

        candidate_prior_p: float
            The combined prior probability of the candidate parameter values
                
        param_priors: Dictionary{String: float}
            The dictionary mapping each parameter to a 
            prior distribution. The float is assumed to
            be the std of a gaussian distribution. If the value
            is zero, then it is instead a uniform distribution

        step_sigmas: [floats]
            Storing the input step_sigmas as described above

        cov_alpha: float
            Determines the coefficient alpha multiplying the covariance
            matrix

        cov: array[floats]
            The covariance matrix of the generating function

        cov_inverse: array[floats]
            The inverse of the covariance matrix

        degeneracy: Boolean
            Described above

        Usage example:
        ---------------
        import MCsampler
        cov = np.identity()
        initial_params = {"Omega_m": 0.3,
                          "Omega_lambda": 0.7,
                          "H0": 74.0,
                          "M_nuisance": -19.23,
                          "Omega_k": 0.0,
                         }
        priors = {"Omega_m": 0.0,
                  "Omega_lambda": 0.0,
                  "H0": 0.0,
                  "M_nuisance": 0.042,
                  "Omega_k": 0.0
        }
        scaling = [0.1, 0.1, 1.0, 0.1, 0.1]
        mcmc = MCsampler.MCMC(initial_params, priors, scaling, True, False)
        mcmc.learncov(cov)
        for _ in range(number_steps):
            mcmc.add_to_chain()
        chain = mcmc.return_chain()
        """

        self.chain = Chain.Chain(initial_condition)
        self.LK = likelihood.LK()
        self.sys_error = systematic_error
        self.initial_params = initial_condition
        self.current_params = initial_condition
        self.current_prior_p = 1.0
        self.candidate_params = {}
        self.candidate_prior_p = 1.0
        self.param_priors = param_priors
        self.step_sigmas = step_sigmas
        self.cov_alpha = 0.1
        self.cov = self.cov_alpha*np.identity(5)
        self.cov_inverse = np.linalg.inv(self.cov)
        self.degeneracy = degeneracy

    def gen_func(self, pars=[], current=[]):
        """
        generating function

        Parameters:
        ----------
        pars:[float]
        A list of value in the order of Omega_m, Omega_Lambda, H0, M_nuisance, Omega_k

        current:[float]
            A list of values in the same order as the input parameter dictionary

        Returns:
        ----------
        nonnorm_pdf:float
        A non-normalized generating function
        """
        index = 0
        #current = list(self.current_params.values())
        for i in range(5):
            for j in range(5):
                #index = index + (pars[i] - current[i]) * self.cov_inverse[i][j] * (
                #    pars[j] - current[j]
                #)
                index = index + (current[i] - pars[i]) * self.cov_inverse[i][j] * (
                    current[j] - pars[j]
                )

        #Alternate numpy version which gives same result
        # delta = np.asarray(current) - np.asarray(pars)
        # index = np.dot(np.dot(delta,self.cov_inverse),np.transpose(delta))
        try:
            nonnorm_pdf = math.exp(-1 * index)
        except OverflowError:
            nonnorm_pdf = 0

        return nonnorm_pdf

    def draw_candidate(self):
        """
        Sampling from the generating fucntion to generate a candidate.
        A real customized 5d random sampling would be wild, this is just a work around that
        makes sense to me. PLEASE let me know any possible improvement.

        """

        current = list(self.current_params.values())
        val = self.current_params["M_nuisance"] + 5 * np.log10(self.current_params["H0"])
        deny = True
        steps = 0
        #scaling = [0.1, 0.1, 1.0, 0.042, 0.1]
        while deny:
            steps = steps + 1
            assert steps < 1000, "Error,value is too small to judge"
            potential_candidate = []
            for i in range(5):
                scaling = self.step_sigmas[i]
                x = np.random.normal(loc=current[i],scale = scaling)
                potential_candidate.append(x)
            
            potential_candidate[4] = 1 - potential_candidate[0] - potential_candidate[1]

            if self.degeneracy:
                potential_candidate[3] = val - 5 * np.log10(potential_candidate[2])

            value = self.gen_func(potential_candidate, current)
            judger = np.random.random_sample()
            if judger < value:
                deny = False

        return potential_candidate

    def learncov(self, cov):
        '''
        A function which feeds a new generating function covariance
        matrix into MCsampler. This allows for iterative determination
        of the covariance matrix. 
        '''
        self.cov = self.cov_alpha * cov
        self.cov_inverse = np.linalg.inv(self.cov)

    def calc_p(self):
        """
        Calculate the probability of moving to a new region
        of parameter space. Note: we assume that the 
        generating functions are symmetric, so the probability
        of the generating function moving from the old parameters
        to the new parameters is the same as vice-versa.
            
        Outputs:
        -----------
        weight: float
            Float between 0 and 1, which is the probability of
            moving to the proposed region of parameter space
        """

        log_likelihood_old, self.current_params = self.LK.likelihood_cal(
            self.current_params, self.sys_error
        )
        log_likelihood_new, self.candidate_params = self.LK.likelihood_cal(
            self.candidate_params, self.sys_error
        )

        weight = min(
           1,
           np.exp(log_likelihood_new)
           * self.candidate_prior_p
           / (np.exp(log_likelihood_old) * self.current_prior_p),
        )

        return weight

    def calc_prior_p(self, params):
        """
        Determine the combined prior probability of all the parameters
        It is assumed that if the value of the prior std is given as 0.0,
        then the prior is a uniform prior and the probability of getting
        that parameter value is 1.
        Otherwise, it is assumed that the prior is a gaussian prior, centered
        on the parameter value given in the intiial conditions, and the
        probability of getting that parameter value is measured

        Inputs:
        ---------
        params: Dictionary{String: float}
            The dictionary mapping all the parameters to values

        Outputs:
        ----------
        combined_prior_probability: float
            The combined probability of all the priors (e.g. P(A_1)*...*P(A_N))
        """

        combined_prior_probability = 1.0
        for key, value in self.param_priors.items():
            if value == 0.0:
                combined_prior_probability *= 1.0
            else:
                mean = self.initial_params[key]
                test_value = params[key]
                p = stat.norm(mean, value).pdf(test_value)
                combined_prior_probability *= p

        self.candidate_prior_p = combined_prior_probability
        return combined_prior_probability

    def take_step(self):
        """
        Determine potential step, and whether step is taken or not. If not,
        keeps the parameters the same.

        Outputs:
        ---------
        new_params: Dictionary{String: float}
            The dictionary containing the mapping of parameter values to
            parameter names of the new parameters.
        """

        #Get candidate parameters and their prior probabilities
        candidate_param_values = self.draw_candidate()
        self.candidate_params = dict(
            zip(list(self.current_params.keys()), candidate_param_values)
        )
        self.candidate_prior_p = self.calc_prior_p(self.candidate_params)

        #Calculate likelihood of these parameters being visited, and take step
        step_weight = self.calc_p()
        r = np.random.random_sample()
        if r <= step_weight:
            new_params = self.candidate_params
            self.current_prior_p = self.candidate_prior_p
        else:
            new_params = self.current_params

        return new_params

    def add_to_chain(self):
        """
        Take a step, and add the new parameter values
        to the Markov Chain
        """

        self.current_params = self.take_step()
        self.chain.add_sample(self.current_params)

    def return_chain(self):
        """
        Return the Markov Chain which the sampler has constructed
        
        Outputs:
        ---------
        self.chain: Chain
            The list of all visited parameter dictionaries, where
            each dictionary is of the type {string: float}, and map
            parameter values to the parameter names
        """

        return self.chain

    def reset_chain(self):
        """
        Resets the Markov Chain which the sampler has constructed
        """

        self.chain = Chain.Chain(self.initial_params)
        self.current_params = self.initial_params
