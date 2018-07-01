"""
Functions for use with recreation of the Soiaporn model 
Gibbs sampling methods.

@author Francesca Capel
@date June 2018
"""
import numpy as np

alpha_T = 20370
A = 3000

def scale_F_T(s, f, eps, w):
    sum_term = 0
    for k in range(len(eps)):
        sum_term += w[k] * eps[k] 
    denom = (1 + s) + ((1 - f) * (alpha_T / 4 * np.pi)) + (f * sum_term)
    return 1 / denom

def get_p_lam(f, eps, kappa, kappa_c, d_i, theta_i, varpi, w):
    p_lam = []

    # k = 0
    f_lam_0 = A * np.cos(theta_i)
    p_lam_0 = f_lam_0 * (1 - f)
    if (p_lam_0 < 1e-16):
        p_lam_0 = 0
    p_lam.append(p_lam_0)

    # k > 0
    for k in range(len(eps)):
        f_lam_k = np.exp(log_fik(kappa, kappa_c, d_i, varpi[k], theta_i))
        p_lam_k = (f_lam_k / eps[k]) * f * w[k]
        if (p_lam_k < 1e-16):
            p_lam_k = 0
        p_lam.append(p_lam_k)

    return p_lam / sum(p_lam)

def prob_lam_i(k, F_T, f, kappa, kappa_c, eps, w, varpi, d):
    if i == 0:
        h = (1 - f) * eps[0]
    else:
        h = f * w[k] * eps[k]
    term1 = np.exp(log_fik(kappa, kappa_c, d, varpi[i]) / eps[i])
    return term1 * h

def fik(kappa, kappa_c, d_i, varpi, theta_i):
    term1 = kappa * kappa_c / (4 * np.sinh(kappa) * np.sinh(kappa_c))
    inner = np.linalg.norm((kappa_c * d_i) + (kappa * varpi))
    term2 = np.sinh(inner) / inner
    return A * np.cos(theta_i) * term1 * term2

def log_fik(kappa, kappa_c, d_i, varpi, theta_i):

    inner = np.linalg.norm((kappa_c * d_i) + (kappa * varpi))
    
    if kappa > 100 or kappa_c > 100:
        lprob = np.log(kappa * kappa_c) - np.log(4 * np.pi * inner) + inner - (kappa + kappa_c) + np.log(2)
    else:
        lprob = np.log(kappa * kappa_c) - np.log(4 * np.pi * np.sinh(kappa) * np.sinh(kappa_c)) + log(np.sinh(inner)) - np.log(inner)

    return np.log(A * np.cos(theta_i)) + lprob


def p_f(F_T, f, eps, lam, w, N_C, a, b):
    sum_term = 0
    for k in range(len(eps)):
        sum_term += w[k] * eps[k]
    inner = -F_T * ((1 - f) * (alpha_T / 4 * np.pi) + f * sum_term)
    term1 = np.exp(inner)
    m_0 = lam.count(0)
    term2 = (1 - f)**(m_0 + b - 1) * f**(N_C - m_0 + a - 1)
    return term1 * term2

def log_p_f(F_T, f, eps, lam, w, N_C, a, b):
    sum_term = 0
    for k in range(len(eps)):
        sum_term += w[k] * eps[k]
    inner = -F_T * ((1 - f) * (alpha_T / 4 * np.pi) + f * sum_term)
    term1 = inner
    m_0 = lam.count(0)
    term2 = np.log((1 - f)**(m_0 + b - 1) * f**(N_C - m_0 + a - 1))
    return term1 + term2

def get_weights(D):

    w = []
    norm = 0
    
    for d in D:
        norm += 1 / d**2

    for d in D:
        w.append( (1 / d**2) / norm )

    return w
