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

        




