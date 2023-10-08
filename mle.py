import numpy as np
from scipy.stats import ncx2, chi2
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

def profile_df_neg_lld(var, nc, x, objective):
    """
    Profile
    :param var: array (1): [degree of freedom]
    :param nc: non-centrality parameter (fixed to point estimate)
    :param x: samples
    :param objective: the objective log-likelihood value to fit
    :return: gap between profile log-likelihood and objective
    """
    prob = ncx2.logpdf(x, var[0], nc)
    prob[prob == np.inf] = - np.inf  # `logpdf` doesn't distinguish +- inf.
    return np.abs(- np.mean(prob) - objective)

def p_df_estimate(samples, max_lld, profile_param):
    """
    Get p-value of log-likelihood Wilk's theorem based test.
     H_0: estimated_param != profile_param (caution: unusual definition)
     H_1: estimated_param == profile_param
    :param samples: data samples
    :param max_lld: max log-likelihood value under null hypothesis
    :param profile_param: parameters under alternative hypothesis (profile log-likelihood)
    :return: p-value
    """
    # L_p(df) = L_p(df*) + 1/2 * chi2(alpha, df=p)
    d = (neg_lld(profile_param, samples) - max_lld) * 2
    # diminish max_lld estimate error: profile neg_LLD < min neg_LLD (MLE)
    d = np.maximum(d, 0)
    p = 1 - chi2.cdf(d, 1)
    return p


def ci_df_estimate(samples, estimated, max_lld, alpha=0.95):
    """
    Confidence interval of parameter 'df'.
    :param max_lld: max log-likelihood value under point estimate
    :param samples: data samples
    :param estimated: point estimate of parameters
    :param alpha: confidence level
    :return: Confidence interval of parameter 'df'.
    """
    n = samples.shape[0]
    # L_p(df) = L_p(df*) + 1/2 * chi2(alpha, df=p)
    objective = max_lld + 0.5 * chi2.ppf(alpha, n) / n
    lower_bound = minimize(
        fun=profile_df_neg_lld,
        x0=np.array([1e-10]),
        args=(estimated[1], samples, objective),  # 'nc' is fixed.
        method='Nelder-Mead',
        bounds=[(1e-10, estimated[0])]
    )
    upper_bound = minimize(
        fun=profile_df_neg_lld,
        x0=np.array([estimated[0]]),
        args=(estimated[1], samples, objective),  # 'nc' is fixed.
        method='Nelder-Mead',
        bounds=[(estimated[0], None)]
    )
    return {
        'CI': (lower_bound.x[0], upper_bound.x[0]),
        'error': (lower_bound.fun, upper_bound.fun),
    }
