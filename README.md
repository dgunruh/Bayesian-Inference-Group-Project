# Results:
## Classes and functions
**There are four main parts for our project:**
1. [likelihood.py](/likelihood.py): loads the raw data, calculates the model luminosity distance and calculate the log-likelihood
2. [MCsampler.py](/MCsampler.py): Includes generating function, use the likelihood calculated from likelihood.py to take steps
3. [Chain.py](/Chain.py): Generates a chain for storing MCMC samples, also calculate the covariance matrix for generating function
4. [plot_mc.py](/plot_mc.py): All the functions about plot. 
## Top level calls
All the modules are loaded in [main.py](/main.py), and the results can be produced by running:
```
python main.py
```
All the final plots are saved in [results](/results)
## Tests
In each file, we implement corresponding tests for the functions and classes in that file. For example, the test for [likelihood.py](/likelihood.py) can be done by running 
```
python likelihood.py
```
in a terminal.
## final plots
**reproduce figure 18:**
![figure 18](/results/fig18.png)
**mcmc result after adding the prior on M:**
![MCMC result](/results/mcmc.png)
**trace plot:**
![trace plot](/results/trace.png)
**posterior probability distribution of H0 after adding prior on M:**
![posterior probability distribution of H0](/results/post_prob_H0.png)
# Group Project Plan
Yize Dong, Davis Unruh, Patrick Wells, Tianqi Zhang
# Code Modules:

**Sampler: Davis/ Tianqi**
1. Samples over parameters. The four parameters to sample over are: H0, M, Ωm, and Ωƛ. 
2. The sampling will occur in a standard MCMC fashion: 
    - A random step in parameter space is proposed via a generating function
    - Degeneracy in the parameter space is removed via an analytic restriction
    - The probability of entering this point in parameter space is assessed by calling the likelihood function, and comparing the two parameter space points. Note, the output of the likelihood function for the current parameter space point is always stored. 
    - A random number is generated to assess whether or not this step is taken
    - If successful, the sampler moves to the proposed point, and the output of the likelihood function for that point is stored. If not successful, an alternate point is proposed. 
    - Results are stored in a Markov Chain, which can itself be sampled at a later time
3. It is noted that we can eliminate M, and H0 after our parameter sampling, and restrict ourselves to a 2-d sampling space. This can be done after the sampling process by only selecting the values of Ωm and Ωƛ, and ignoring the values of M and H0. This will be critical for reproducing Fig. 18. 
4. We can then calculate the posterior probability density of H0 from this 2d space. 
- Specific tasks that need to be completed for the sampler:
    - Write a routine that can move from one point in the 4-d Ωm/ Ωƛ/ H0/ M space to another -Davis/Tianqi
    - This routine will include a generating function that will propose new values of the parameters -Tianqi
    - This routine will also include a restriction of the possible parameter values, including an analytic elimination of the sampler reaching degenerate points -Davis
    - The routine will also include a calculation of probability: how likely is it for the parameters to evolve to be equal to the new parameters. This will include an external call to the likelihood function, as well as the a forementioned generating function -Davis
    - Finally, the routine will include a random number calculation to assess whether or not the step in parameter space is taken -Tianqi
    - All results will be stored in the external Markov-Chain data structure -Davis



**Likelihood: Patrick/Yize**
1. Loads data and stores it in a way that is easy to access - Yize
2. Calculates log-likelihood given values of the cosmological parameters
3. Does not actually have to calculate the likelihood at every point, since it’s always the ratio of likelihoods that matters. If we save the chi-squared at every point, we can always use it to calculate the ratio of likelihoods and save some computation time.
4. Should include tests that ensures that the result of a likelihood calculation makes sense (for example, that it’s not infinite)
- Specific tasks that need to be completed for the likelihood:
    - Move input routine from python notebook into [likelihood.py](/likelihood.py) - Yize
    - Write routine that selects relevant data from input - Yize
    - Write routine that calculates the model based on current value of sampled parameters -  Patrick
    - Write a routine that returns the log-likelihood based on model - Patrick


**Chain**
1. A data structure which will contain the parameter space points visited by the MCMC sampler. The points in the Markov chain can themselves be sampled, and will resemble the output of the likelihood function. 
- Specific tasks that need to be completed for the chain:
    - Make a basic data structure that can store successive steps in the monte-carlo space (probably needs to be done early) - Patrick
    - Make an interface for plotting results


**Results**
1. Plots results - Yize
    - A trace plot for three parameters to check if the model is converged.
    - A final pot that reproduces the red and grey contours of Figure 18 in the paper
    - Maybe also a plot to show the residuals to the best-fit model parameters, like the bottom panel of Figure 11.


**Manager**
1. Implemented in a jupyter notebook
2. Loads other modules from .py files
3. Reads in MCMC inputs (number of steps, priors, etc.)
4. Passes data between modules 
- Specific tasks that need to be completed for the manager (Tianqi,using modules that can have different name in the future):
    - Write a routine that loads in any parameters associated with the given run (initial values, priors, etc.)
    - Write a routine that loads the MCMC and Likelihood modules and passes them any needed data to start run
    - Write a routine takes the chain returned by the MCMC and writes it to file for later use

# Schedule:
- Complete main code (ie sampler and likelihood function) by May 4th.
- Complete secondary code by May 5th. Produce final figures (reproduction of Fig. 18, and posterior probability density of H0). Begin work on the presentation.
- May 6th: Review presentation, and decide who will present each portion
- May 7th: present completed work to entire class

# Structure of repository:
- Various modules will be created in their own .py files
- Jupyter notebook will tie the whole thing together
- 3 folders - Data, modules, config



# Some data formatting things:
- Sampler will pass a dictionary to the likelihood function in the following format : {Omega_m: float, Omega_L: float, Omega_k: float, H0: float, M: float}
- Likelihood function will return the log-likelihood to the sampler
