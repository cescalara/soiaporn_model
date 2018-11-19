from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ['get_contour_inputs', 'make_fig_5_plot']

"""
Functions to recreate Soiaporn et al. parameter posterior plots
with Stan ouput.

@author Francesca Capel
@date July 2018
"""

def get_contour_inputs(A_samples, B_samples, levels, smooth):
    """
    Get inputs to the matplotlib plt.countour() function for specified HPD levels
    for parameters A and B. Used to make a joint marginal posterior contour plot.

    :param A_samples: samples for parameter A
    :param B_samples: samples for parameter B
    :param levels: list of HPD levels
    """


    H, X, Y = np.histogram2d(A_samples, B_samples, bins = 500)
    H = gaussian_filter(H, smooth)
    
    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    return X1, Y1, H.T, V


def make_fig_5_plot(chain_a, chain_b, levels, smooth, cmap = None):
    """
    Reacreate the plot in figure 5 of Soiaporn et al.
    Input chains should be a dict with 'log10_kappa' and 'f'
    samples.

    :chain_a: samples for 17 AGN, period 1+2+3
    :chain_b: samples for 17 AGN, period 1+2
    """

    # find levels
    f_samples_a = chain_a['f']
    f_samples_b = chain_b['f']
    kappa_samples_a = chain_a['log10_kappa']
    kappa_samples_b = chain_b['log10_kappa']

    # make plot
    xa, ya, ha, va = get_contour_inputs(kappa_samples_a, f_samples_a, levels, smooth)
    xb, yb, hb, vb = get_contour_inputs(kappa_samples_b, f_samples_b, levels, smooth)

    f, ax = plt.subplots(1, 2, sharey = True, figsize = (12, 5))
    ax[0].contour(xb, yb, hb, vb, cmap = cmap)
    ax[0].set_xlim(0, 3)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_title('17 AGN, period 2+3')
    ax[0].set_xlabel('$log_{10}(\kappa)$')
    ax[0].set_ylabel('f')
    ax[1].contour(xa, ya, ha, va, cmap = cmap)
    ax[1].set_xlim(0, 3)
    ax[1].set_title('17 AGN, period 1+2+3')
    ax[1].set_xlabel('$log_{10}(\kappa)$')

    return f





















































    
