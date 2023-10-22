import numpy as np
from scipy.stats import ncx2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def neg_lld(var, x):
    """
    Negative log-likelihood function of F distribution's PDF
    :param var: array (2): [degree of freedom; non-centrality parameter]
    :param x: samples
    :return:
    """
    prob = ncx2.logpdf(x, var[0], var[1])
    prob[prob == np.inf] = - np.inf  # `logpdf` doesn't distinguish +- inf.
    return - np.mean(prob)

def estimate(samples):
    """
    Max log-likelihood estimation under assumption of non-centralized chi2 distribution.
    Note: the result is largely dependent to the starting point of "degree of freedom".
    :param samples: observations
    :return: [1] array (2): [degree of freedom; non-centrality parameter]
             [2] array (2x2): Variance matrix of point estimation
    """
    result = minimize(
        fun=neg_lld,
        x0=np.array([1, 0]),
        args=(samples,),
        method='L-BFGS-B',
        bounds=[(1e-10, None), (0, None)]
    )
    return {
        'param': result.x,
        'variance': - result.hess_inv.todense(),
        'neg_lld': result.fun,
    }

def draw(samples, estimated):
    """
    Draw sample's histogram and non-centralized chi2 PDF function's line plot.
    :param samples: observations
    :param estimated: Estimated parameter
     assume samples fit the non-centralized chi2 distribution, parameters are
     [degree of freedom; non-centrality parameter]
    :return: Figure
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 6))
    ax1.hist(samples, bins=21, edgecolor='k', color='#808080')
    ax1.set_xlabel('$(\hat y - y)^2$')
    ax1.set_ylabel('Count')
    ax2 = ax1.twinx()
    hist_range = ax1.get_xlim()
    ax2_x = np.linspace(0, hist_range[1], 400)
    ax2_y_hat = ncx2.pdf(ax2_x, *estimated)
    ax2.plot(ax2_x, ax2_y_hat, color='k')
    ax2.set_ylabel('Estimated Probability Density')
    ax2.set_ylim([0, None])
    return fig

def profile_nc_neg_lld(var, df, x):
    """
    Profile
    :param var: array (1): non-centrality parameter (fixed to point estimate)
    :param df: degree of freedom
    :param x: samples
    :return: gap between profile log-likelihood and objective
    """
    prob = ncx2.logpdf(x, df, var[0])
    prob[prob == np.inf] = - np.inf  # `logpdf` doesn't distinguish +- inf.
    return - np.mean(prob)

def profile_nc_estimate(samples, df):
    result = minimize(
        fun=profile_nc_neg_lld,
        x0=np.array([0]),
        args=(df, samples),
        method='L-BFGS-B',
        bounds=[(0, None)]
    )
    return {
        'param': result.x,
        'variance': - result.hess_inv.todense(),
        'neg_lld': result.fun,
    }
