import numpy as np

class Chain:

    def __init__(self, params=[]):
        '''
        A basic implementation of a chain for storing MCMC samples

        Parameters
        ----------
        params: list of strings:
                Parameters that will be included in samples
                The log likelihood (loglkl) is assumed to be a parameter and
                does not need to be passed

        Attributes
        ----------
        params: list of strings:
                Parameters that will be included in the samples
        '''
        #self.params = params + ['loglkl']
        self.params = params
        self.samples = []
        self.sample_values = []

    def add_sample(self, sample={}):
        '''
        Adds a sample to the chain. Will throw an error if an expected
        MCMC parameter is missing from the sample, or if one of the values
        is not a number

        Parameters:
        -----------
        sample: {string: float}
            Dictionary where the keys are the MCMC parameter names and
            the entries are the values of those parameters respectively.

        '''

        for param in self.params:
            assert param in sample.keys(), 'Error: sample does not'\
                                           ' contain entry for parameter {}'.format(param)
        for param, value in sample.items():
            try:
                float(value)
            except ValueError:
                print("Error: value of paramater {} is not a number".format(param))
                return
        sample_value = list(sample.values())
        self.samples.append(sample)
        self.sample_values.append(sample_value)


    def cov_cal(self,scale=1):
        samplelist = np.array(self.sample_values)
        samplelist_t = samplelist.transpose()
        sample_mean = []
        delta_list = []
        cov = []
        for row in samplelist_t:
            mean = sum(row)/len(row)
            sample_mean.append(mean)
            delta = row - mean
            delta_list.append(delta)

        for i in range(5):
            for j in range(5):
                element = np.multiply(delta_list[i],delta_list[j])
                cov_element = sum(element)/len(element)
                cov.append(cov_element)

        #Alternate numpy version which gives same result
        # new_delta = samplelist_t - samplelist_t.mean(axis = 1, keepdims = True)
        # new_cov_array = np.dot(new_delta, new_delta.transpose())/len(samplelist)
        
        cov_array = np.array(cov)
        cov_matrix = cov_array.reshape(5,5)

        return cov_matrix

        
def simulator(num=500, sigma=[0.05, 0.1, 5, 1, 0.01]):
    samples = []
    for i in np.arange(num):
        pars = {'Omega_m': abs(np.random.normal(0.3,sigma[0],1)[0]), 
                'Omega_lambda': abs(np.random.normal(0.7,sigma[1],1)[0]), 
                'H0': np.random.normal(70,sigma[2],1)[0], 
                'M_nuisance': np.random.normal(-19, sigma[3], 1)[0],
                'Omega_k': np.random.normal(0., sigma[4], 1)[0]}
        #_loglk, pars = LK.likelihood_cal(pars, ifsys=False)
        samples.append(pars)
    return samples

def test_Chain():
    #test the function add_sample and cov_cal in Chain.py
    sigma = [0.05, 0.1, 5, 1, 0.01]
    chain = Chain(simulator(1, sigma=sigma)[0])
    for i in np.arange(1000):
        sample = simulator(1, sigma=sigma)
        chain.add_sample(sample[0])
    cov_obs = chain.cov_cal()
    cov = []
    for i in range(5):
        for j in range(5):
            cov_element = sigma[i] * sigma[j]
            cov.append(cov_element)
        
    cov_expc = np.array(cov)
    cov_expc = cov_expc.reshape(5, 5)
    isclose = np.all((np.isclose(np.diag(cov_expc), np.diag(cov_obs), 0.1)))
    if isclose:
        print ('Chain.py is tested!')
    else:
        assert isclose, "The cov matrix of generating function is not well calculated"

if __name__ == '__main__':
    test_Chain()
