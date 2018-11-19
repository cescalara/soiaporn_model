# Functions for use with the end of course assignment
import numpy as np
import scipy.constants as const
import seaborn as sns
import matplotlib.pyplot as plt
from hpd import *

# Define the flux density
# input wl [m], T [K], R [km], D [pc].
def flux_density(wl, T, R, D):
    one_parsec = 3.0857e16
    conv_fac = ((1e3)**2/(one_parsec**2))*1e26
    a = (( 2*const.pi*const.speed_of_light*const.h) / wl**3)
    b = np.longfloat( (const.h*const.speed_of_light)/(wl*const.k*T) )
    return ( a *
             (1 / (np.exp(b) - 1)) *
             (R**2 / D**2) ) * conv_fac 

# Define the Chi-square
def chi_square(flux, wl, sig, T, R, D):
    n_b = np.size(flux)
    if ((np.size(T) == 1000) & (np.size(R) == 1)):
        # for making the temperature plot
        chi_sq = np.zeros(np.size(T))
    elif ((np.size(T) == 1000) & (np.size(R) == 1000)):
        # for making the 2d plots
        chi_sq = np.zeros([1000, 1000])
        T, R = np.meshgrid(T, R)
    else:
        # for single values
        chi_sq = 0
    for i in range (0, n_b):
            chi_sq += ( (flux[i] - flux_density(wl[i], T, R, D)) / sig[i] )**2
    return chi_sq

# Define the heaviside step function
def H_step(x):
    return 1 * (x > 0)

# Define the conditional posterior distibution 
def cond_posterior(flux, wl, sig, T, R, D):
    n_b = len(flux)
    likelihood = np.zeros([1000,1000])
    prior = H_step(T) * H_step(R)
    likelihood = (( 1 / ((2*const.pi)**(n_b/2) * np.prod(sig)) ) *
                  np.exp( -0.5 * chi_square(flux, wl, sig, T, R, D) ) )
    post = prior*likelihood
    return post

# Define the full prior distribution
def full_prior(T, R, D):
    return (H_step(T)*np.exp( (-T/20000) ) *
            H_step(R)*np.exp( -0.5*(((R - 8000e3)/1000e3)**2) ) *
            H_step(D)*D**2*np.exp( (-(D/(500*3.0857e16))) ) )

# Define the full posterior distribution
def full_posterior(flux, wl, sig, T , R, D):
    n_b = len(flux)
    likelihood = 0
    prior = full_prior(T, R, D)
    likelihood = (( 1 / ((2*const.pi)**(n_b/2) * np.prod(sig)) ) *
                  np.exp( -0.5 * chi_square(flux, wl, sig, T, R, D) ) )
    post = prior*likelihood
    return post

# Define the log posterior distribution
def full_posterior_log(flux, wl, sig, T , R, D):
    return - ( (T/20000) + 0.5*(((R - 8000)/1000)**2) - (2*np.log(D)) + (D/500) +
               0.5*chi_square(flux, wl, sig, T , R, D) )

# Define the metropolis algorithm
def metropolis(flux, wl, sig, T_0, R_0, D_0, N):

    #initialise
    T_out = []
    R_out = []
    D_out = []
    post_out=[]
    T_i = T_0
    R_i = R_0
    D_i = D_0
    N_acc = 0

    for i in range (0, N):
        # Draw trial values from normal distributions
        T_trial = np.random.normal(T_i, 800, 1)
        R_trial = np.random.normal(R_i, 1000, 1)
        D_trial = np.random.normal(D_i, 150, 1)
            
        full_post_trial = full_posterior_log(flux, wl, sig, T_trial, R_trial, D_trial)
        full_post_i = full_posterior_log(flux, wl, sig, T_i, R_i, D_i)                             
        # Selection for acceptance of trial values
        if (full_post_trial >= full_post_i):
            T_i = T_trial
            R_i = R_trial
            D_i = D_trial
            N_acc += 1

        else:
            # Accept trial value with probabilty equal to the trial/initial posterior ratio
            if ( np.random.random() >
                ( np.exp(full_post_trial - full_post_i) )  ):
                T_i = T_i
                R_i = R_i
                D_i = D_i
            
            else:
                T_i = T_trial
                R_i = R_trial
                D_i = D_trial
                N_acc += 1

        T_out.append(T_i)
        R_out.append(R_i)
        D_out.append(D_i)
        post_out.append(full_post_i)
        accept_frac = N_acc/float(N)

    return T_out, R_out, D_out, post_out, accept_frac

# Define function to make a plot of the HPD regions
def makeplot_hpd(mcmc_sample, xlabel):
    plt.figure()
    xy = sns.distplot(mcmc_sample).get_lines()[0].get_data()
    arr90 = hpd(mcmc_sample, 0.10)
    if (np.shape(arr90)[0] == 1):
        idx90 = (xy[0] > arr90[0][0]) & (xy[0] < arr90[0][1])
    else:
        idx90 = (xy[0] > arr90[0]) & (xy[0] < arr90[1])
    plt.fill_between( xy[0][np.where(idx90)], 0, xy[1][np.where(idx90)], label = '90%', facecolor='green', alpha=0.5)
    arr60 = hpd(mcmc_sample, 0.40)
    if (np.shape(arr60)[0] == 1):
        idx60 = (xy[0] > arr60[0][0]) & (xy[0] < arr60[0][1])
    else:
        idx60 = (xy[0] > arr60[0]) & (xy[0] < arr60[1])
    plt.fill_between( xy[0][np.where(idx60)], 0, xy[1][np.where(idx60)], label = '60%', facecolor='red', alpha=0.5)
    arr30 = hpd(mcmc_sample, 0.70)
    if (np.shape(arr30)[0] == 1):
        idx30 = (xy[0] > arr30[0][0]) & (xy[0] < arr30[0][1])
    else:
        idx30 = (xy[0] > arr30[0]) & (xy[0] < arr30[1])
    plt.fill_between( xy[0][np.where(idx30)], 0, xy[1][np.where(idx30)], label = '30%', facecolor='black', alpha=0.5)
    plt.legend()
    plt.xlabel(xlabel)

# Define functions to plot the individual priors
def priorT(T):
    return (np.exp(-(T/20000)) )

def priorR(R):
    return ( np.exp(-0.5*(((R - 8000)/1000)**2)) )

def priorD(D):
    return ( (D**2) * np.exp(-(D/500)) )
    

# Define the Geweke convergence test
def geweke_test(mcmc_sample):
    N = len(mcmc_sample)
    S=100
    # Split the chain into 100 segments
    samples = np.reshape(mcmc_sample, (S, (N/S)))
    # Calculate the variance and mean of each sample
    mean = []
    var = []
    for i in range(0, S):
        mean.append(np.mean(samples[i]))
        var.append(np.var(samples[i]))

    mean=np.array(mean)
    var=np.array(var)
    # Calculate the z values for sequential segments (i - i+1)
    z = []
    for i in range(0, S-1):
        z.append( (mean[i] - mean[i+1])/np.sqrt((var[i] + var[i+1])) )
    z=np.array(z)

    return z

# --------------------------------------------------------
# Define the functions necessary for autocorrelation

# Define the effective sample size
def N_eff(mcmc_sample, auto_corr):
    N=len(mcmc_sample)
    return (N / (1 + (2*np.sum(auto_corr))) )


# Define autocorrelation at a specified lag
def autocorr(x, lag=1):
    S = autocov(x, lag)
    return S[0, 1] / np.sqrt(np.prod(np.diag(S)))

# Define the autocovariance at a specified lag
def autocov(x, lag=1):
    x = np.asarray(x)
    if not lag:
        return 1
    if lag < 0:
        raise ValueError("Autocovariance lag must be a positive integer")
    return np.cov(x[:-lag], x[lag:], rowvar=0, bias=1)

# Define the autocorr for increasing lags
def sample_corr(mcmc_sample, lags):
    auto_corr=[]
    for l in lags:
        a = autocorr(mcmc_sample, l)
        if (a > 0):
            auto_corr.append(a)
        else:
            # once = 0 fluctuating around noise values
            return np.array(auto_corr)
    return np.array(auto_corr)

# --------------------------------------------------------
# Define the functions necessary for calculating the HPD
def calc_min_interval(x, alpha):
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions

    level = len(dimensions)

    if level == 1:
        return range(dimensions[0])

    indices = [[]]

    while level:

        _indices = []

        for j in range(dimensions[level - 1]):

            _indices += [[j] + i for i in indices]

        indices = _indices

        level -= 1

    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices

def hpd(x, alpha=0.05):
    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


    
