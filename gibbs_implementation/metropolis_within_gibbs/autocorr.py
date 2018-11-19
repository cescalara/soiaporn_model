import numpy as np

# Define autocorrelation at a specified lag
def autocorr(x, lag=1):
    S = autocov(x, lag)
    return S[0, 1] / np.sqrt(np.prod(np.diag(S)))

# Define the autocovariance at a specified lag
def autocov(x, lag=1):
    x = np.asarray(x)
    if not lag:
        return 1
    return np.cov(x[:-lag], x[lag:], rowvar=0, bias=1)

# Define the sample correlation for increasing lags
def sample_corr(mcmc_sample, lags):
    auto_corr=[]
    for l in lags:
        a = autocorr(mcmc_sample, l)
        if (a > 0):
            auto_corr.append(a)
        else:
            # once = 0 fluctuating around noise values
            return auto_corr
    return auto_corr

def N_eff(mcmc_sample, auto_corr):
    N=len(mcmc_sample)
    return (N / (1 + (2*np.sum(auto_corr))) )
