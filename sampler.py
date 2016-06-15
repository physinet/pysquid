from __future__ import (division, print_function)
import numpy as np


class MHSampler(object):
    """
    Very basic Metropolis-Hasting sampler.

    """
    
    def __init__(self, stepsize, *args, **kwargs):
        """ 
        input:
            stepsize : float or np.array to specify stepsize
                       for each parameter

        """
        self.stepsize = stepsize
        self.N_samples = 0
        self.rejection = 0

    def sample(self, model, p0 = None, names = None):
        """
        One sampling step.
        input:
            model : Instance of FluxModel, must have 
                    computeNLL method which returns
                    negative-log-Likelihood
            p0 : numpy array of parameters to evaluate
                 sampling step at. Default None uses the
                 current state of model. Order of parameters
                 in array depends on names.
            names : list of strings of the names of parameters
                    over which sampling will occur. The order
                    of names must match the order in the flat
                    array p0. Default None is all parameters in
                    default order (determined by ParameterMap)
        returns:
            log-Likelihood of new point

        """
        self.N_samples += 1
        p0 = p0 if p0 is not None else model.getParams(names)
        N_params = len(p0)
        step = self.stepsize * (2*np.random.rand(N_params) - 1)
        ll_before = -1*model.computeNLL(p0, names)
        proposal = p0 + step
        
        ll_after = -1*model.computeNLL(proposal, names)
        if np.log(np.random.rand()) < ll_after - ll_before: 
            return -1*model.computeNLL()
        self.rejection += 1
        return -1*model.computeNLL(p0, names)

    @property
    def acceptance(self):
        """
        Acceptance rate of steps. Aim for 0.5,
        half rejected and half accepted. Smaller
        step sizes are accepted more often.

        """
        return 1-self.rejection/self.N_samples
