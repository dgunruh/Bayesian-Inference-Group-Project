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
        self.params = params + ['loglkl']
        self.samples = []

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

        self.samples.append(sample)
