import numpy as np
import likelihood
import Chain


class MCMC:
    def __init__(self, systematic_error=True):
        """
        Class which perform the MCMC tasks
        
        Inputs:
        ---------
        systematic_error: boolean
            Tells the sampler whether to calculate
            the likelihood with or without using the
            systematic error
            
        Attributes:
        ------------------
        chain: Chain
            The Markov Chain which the sampler is creating
            
        LK: likelihood
            An instance of the likelihood class which will
            return the log-likelihood of each set of parameter
            values
            
        sys_error: boolean
            The storage of the systematic_error input
        
        current_params: Dictionary{string: float}
            The dictionary mapping the current parameter
            values to the parameters in question
            
        Usage example:
        ------------------
        import Sampler
        mcmc = Sampler.MCMC()
        for _ in range(number_steps):
            mcmc.add_to_chain()
        chain = mcmc.return_chain()
        """

        self.chain = Chain.Chain()
        self.LK = likelihood.LK()
        self.sys_error = systematic_error
        self.current_params = {
            "Omega_m": 0.3,
            "Omega_lambda": 0.7,
            "H0": 74.0,
            "M_nuisance": -19.23,
            "Omega_k": 0.0,
        }

    def generate_params(self):
        """
        Generate new values for Omega_m, Omega_λ, and H0
        """

        # degeneracy: M + 5*np.log10(H0) can be considered one number.
        # For the 1st chain: keep these degenerate. Future chains: prevent degeneracy
        x = self.params["M_nuisance"] + 5 * np.log10(self.params["H0"])
        new_M_nuisance = x - 5 * np.log10(new_H0)

        return {
            "Omega_m": 0.0,
            "Omega_lambda": 0.0,
            "H0": 0.0,
            "M_nuisance": new_M_nuisance,
            "Omega_k": 0.0,
        }

    def calc_p(self, new_params):
        """
        Calculate the probability of moving to a new region
        of parameter space. Note: we assume that the 
        generating functions are symmetric, so the probability
        of the generating function moving from the old parameters
        to the new parameters is the same as vice-versa.
        
        Inputs:
        ------------
        new_params: dictionary{string: float}
            Dictionary of the new parameters and their values
            
        Outputs:
        -----------
        weight: float
            Float between 0 and 1, which is the probability of
            moving to the proposed region of parameter space
        """

        log_likelihood_old, self.params = self.LK.likelihood_cal(
            self.current_params, self.sys_error
        )
        log_likelihood_new, new_params = self.LK.likelihood_cal(
            new_params, self.sys_error
        )

        # Calculate the weight only using the log-likelihood, due to large numbers
        weight = min(1, log_likelihood_new / log_likelihood_old)

        return weight

    def take_step(self):
        """
        Determine whether step is taken or not. If not,
        keeps the parameters the same
        """

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
