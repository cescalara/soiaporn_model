import numpy as np
from matplotlib import pyplot as plt

from .soiaporn_functions import *

F_T_MIN = 0.0
F_T_MAX = 0.5

f_MIN = 0.0
f_MAX = 1.0


__all__ = ['MetropolisWithinGibbs', 'InputData', 'InputParameters']


class MetropolisWithinGibbs():
    """
    Manage Metropolis-within-Gibbs sampling of the Soiaporn et al. 2012 model
    for UHECR arrival directions
    @author Francesca Capel
    @date July 2018
    """

    def __init__(self, input_data, input_parameters):
        """
        Manage Metropolis-within-Gibbs sampling of the Soiaporn et al. 2012 model
        for UHECR arrival directions
        @author Francesca Capel
        @date July 2018
        """

        self.input_data = input_data
        self.input_parameters = input_parameters
        self.samples = []
        self.total_samples = Chain()
        self.rhat = Rhat()

        
    def Sample(self, Niter = 1000, Nchain = 1):
        """
        Run the sampler for Niter on Nchain chains.
        """

        self.Niter = Niter
        self.Nchain = Nchain

        # initialise
        F_T_init = np.random.uniform(F_T_MIN, F_T_MAX, Nchain)
        f_init = np.random.uniform(f_MIN, f_MAX, Nchain)

        # run chains
        for i in range(Nchain):
            chain = self.RunChain(Niter, F_T_init[i], f_init[i])
            self.samples.append(chain)

        # asses convergence
        self.calculate_rhat()

        # accepted fraction
        accept_count_tot = 0
        for s in self.samples:
            accept_count_tot += s.accept_count
        accept_fraction = accept_count_tot / (self.Niter * self.Nchain)
        
        print('Sampling completed')
        print('------------------')
        print('rhat f: %.2f' % self.rhat.f)
        print('rhat F_T: %.2f' % self.rhat.F_T)
        print('accepted fraction: %.2f' % accept_fraction)
        print('')
        self.get_total_samples()
        
            
    def RunChain(self, Niter, F_T_init, f_init):
        """
        Run a chain for initial values F_T_init and f_init 
        for Niter iterations
        """

        F_T = F_T_init
        f = f_init
        N_C = self.input_data.N_C
        eps = self.input_data.eps
        w = self.input_data.w
        varpi = self.input_data.varpi
        d = self.input_data.d
        theta = self.input_data.theta
        A = self.input_data.A
        
        a = self.input_parameters.a
        b = self.input_parameters.b
        s = self.input_parameters.s
        kappa = self.input_parameters.kappa
        kappa_c = self.input_parameters.kappa_c

        print('kappa:', kappa)
        print('eps', eps)

        chain = Chain()
        
        for i in range(Niter):

            # F_T
            try:
                F_T = np.random.gamma(N_C + 1, scale_F_T(s, f, eps, w))
            except:
                print(scale_F_T(s, f, eps, w), f, eps)
            chain.F_T.append(F_T)
    
            # lambda
            lam = []
            for i in range(N_C):
                p = get_p_lam(f, eps, kappa, kappa_c, d[i], theta[i], A[i], varpi, w)
                # p = get_p_lam_alt(F_T, eps, f, w)
                sample = np.asarray(np.random.multinomial(1, p))
                try:
                    lam.append(np.where(sample == 1)[0][0])
                except:
                    print(p, sample, f, kappa)
                    
            # f 
            f_new = np.random.normal(f, 0.4)
            while (f_new < 0 or f_new > 1):
                f_new = np.random.normal(f, 0.4)
            p_f_old = log_p_f(F_T, f, eps, lam, w, N_C, a, b)
            p_f_new = log_p_f(F_T, f_new, eps, lam, w, N_C, a, b)
            if(p_f_new > p_f_old):
                f = f_new
                chain.accept_count += 1
            else:
                accept_ratio = np.exp(p_f_new - p_f_old)
                check = np.random.uniform(0, 1)
                if check <= accept_ratio:
                    f = f_new
                    chain.accept_count += 1
                else:
                    f = f
            chain.f.append(f)

        return self.remove_burn_in(chain)

    
    def remove_burn_in(self, chain):
        """
        Remove burn in from a chain.
        """
        if self.Niter < 500:
            self.Nburn = self.Niter
        else:
            self.Nburn = 500

        chain.F_T = chain.F_T[self.Nburn :]
        chain.f = chain.f[self.Nburn :]

        return chain
    
        
    def calculate_rhat(self):
        """
        calculate rhat
        """
        num_samples = self.Niter - self.Nburn
        
        f = np.zeros((self.Nchain, num_samples))
        F_T = np.zeros((self.Nchain, num_samples))
        
        for i in range(self.Nchain):
            f[i] = self.samples[i].f
            F_T[i] = self.samples[i].F_T

        def rscore(var, num_samples):
            
            # between chain variance
            B = num_samples * np.var(np.mean(var, axis = 1), axis = 0, ddof = 1)
            
            # within_chain variance
            W = np.mean(np.var(var, axis = 1, ddof = 1), axis = 0)
            
            # marginal posterior variance estimate
            Vhat = W * (num_samples - 1) / num_samples + B / num_samples

            return np.sqrt(Vhat / W)

        self.rhat.f = rscore(f, num_samples)
        self.rhat.F_T = rscore(F_T, num_samples)
        

    def get_total_samples(self):
        """
        Merge samples from all chains for easy storage.
        """
        for s in self.samples:
            self.total_samples.f += s.f
            self.total_samples.F_T += s.F_T
        
        
    def traceplot(self):
        """
        Print a stan-style summary traceplot.
        """
        num_samples_tot = (self.Niter - self.Nburn) * self.Nchain
        x = range(num_samples_tot)

        f = []
        F_T = []
        for s in self.samples:
            f += s.f
            F_T += s.F_T
        
        fig, axarr = plt.subplots(2, sharex = True)
        axarr[0].plot(x, f)
        axarr[0].set_title('f')
        axarr[1].plot(x, F_T)
        axarr[1].set_title('F_T')


        
class InputData():
    """
    Input data to the MetropolisWithinGibbs object.
    """

    def __init__(self, d, A, theta, varpi, D, eps):
        """
        Input data to the MetropolisWithinGibbs object.
        """

        self.d = d
        self.A = A
        self.theta = theta
        self.varpi = varpi
        self.D = D
        self.eps = eps
        self.N_C = len(self.theta)
        self.w = get_weights(self.D)
    

class InputParameters():
    """
    Input data to the MetropolisWithinGibbs object.
    """

    def __init__(self, kappa, kappa_c, a, b, s):
        """
        Input data to the MetropolisWithinGibbs object.
        """

        self.kappa = kappa
        self.kappa_c = kappa_c
        self.a = a
        self.b = b
        self.s = s
        

class Chain():
    """
    Store samples for a single chain.
    """

    def __init__(self):
        """
        Store samples for a single chain.
        """

        self.F_T = []
        self.f = []
        self.accept_count = 0

class Rhat():
    """
    Store rhat values for each parameter.
    """

    def __init__(self):
        """
        Store rhat values for each parameter.
        """

        self.f = None
        self.F_T = None
