import numpy as np



def gev_pdf(x, mu, sigma, xi):
    """ 
    Calculate the GEV PDF
        Parameters:
    x : array-like, Values at which to evaluate the CDF.
    mu : float, Location parameter.
    sigma : float, Scale parameter.
    xi : float, Shape parameter .
    
    Returns:
    pdf : array-like
        Values of the GEV PDF evaluated at the given x points.
    """
    # Make sure sigma is positive
    assert sigma > 0, "sigma must be positive"
    
    # Calculate the value of t(x)
    t = 1 + xi * (x-mu)/sigma
    
    # Check if t(x) > 0, otherwise PDF is 0
    valid = t > 0
    pdf = np.zeros_like(x)
    
    if xi != 0:
        pdf[valid] = (1/sigma) * (t[valid]**(-1/xi-1)) * np.exp(-t[valid] ** (-1/xi))
    else:  
        pdf[valid] = (1/sigma) * (np.exp(-(x[valid]-mu)/sigma)**(xi+1)) * np.exp(-np.exp(-(x[valid]-mu)/sigma))
    
    return pdf

def gev_cdf(x, mu, sigma, xi):
    """ 
    Calculate the GEV CDF 
            Parameters:
    x : array-like, Values at which to evaluate the CDF.
    mu : float, Location parameter.
    sigma : float, Scale parameter.
    xi : float, Shape parameter .
    
    Returns:
    cdf : array-like
        Values of the GEV CDF evaluated at the given x points.
    """
    # Make sure sigma is positive
    assert sigma > 0, "sigma must be positive"
    
    # Calculate the value of t(x)
    t = 1 + xi * (x - mu) / sigma
    
    # Check if t(x) > 0, otherwise CDF is 0
    valid = t > 0

    if xi == 0: # When xi == 0, that is, the limiting distribution
        cdf = np.zeros_like(x)
        cdf[valid] = np.exp(-np.exp(-(x[valid] - mu) / sigma))
    
    # Classification discussion on the positive and negative cases of xi
    # When I first tried it without classification, it caused some unreasonable situations, 
    # for example, the CDF reached 1 and then quickly dropped to 0
    elif xi > 0: 
        cdf = np.zeros_like(x)
        cdf[valid] = np.exp(-t[valid] ** (-1 / xi))
        
    elif xi < 0:
        cdf = np.ones_like(x) # This is the only difference from above where xi>0
        cdf[valid] = np.exp(-t[valid] ** (-1 / xi))
    
    return cdf



def gev_quantile(q, mu, sigma, xi):
    """
    Calculate the quantile function (inverse CDF) of the GEV distribution.
    
        Parameters:
    q : array-like, Quantiles to evaluate (values between 0 and 1).
    mu : float, Location parameter.
    sigma : float, Scale parameter.
    xi : float, Shape parameter.
    
    Returns:
    quantile : array-like, Values of the GEV quantile (inverse CDF) corresponding to the given q values.
    
    """
    assert sigma > 0, "sigma must be positive"

    if xi != 0:  # Handle the case where xi != 0
        x = mu + (sigma / xi) * ((-np.log(q)) ** (-xi) - 1)
    else:        # Special case for xi = 0
        x = mu - sigma * np.log(-np.log(q))
    
    return x


def gev_ns_pdf(x, t, cmu, mu0, sigma, xi):
    """ 
    Calculate the non-stationary GEV PDF with μ(t) = cmu * t + mu0
        Parameters:
    x : array-like, the values at which to evaluate the PDF.
    t : float or array-like, the time variable for which the non-stationarity is introduced.
    cmu : float, the coefficient for the time-varying location parameter μ(t).
    mu0 : float, the location parameter.
    sigma : float, the scale parameter.
    xi : float, the shape parameter.

    Returns:
    pdf : array-like, the calculated PDF values for each x at time t.
    The location parameter μ(t) is modeled as a linear function of time:
        μ(t) = cmu * t + mu0
    This allows the GEV distribution to vary with time, representing non-stationarity.
    """
    # Make sure sigma is positive
    assert sigma > 0, "sigma must be positive"

    # Calculate the time-dependent location parameter μ(t)
    mu_t = cmu * t + mu0

    # Calculate the value of t(x) = 1 + xi * (x - μ(t)) / σ
    t_x = 1 + xi * (x - mu_t) / sigma

    # Check if t(x) > 0, otherwise PDF is 0
    valid = t_x > 0
    pdf = np.zeros_like(x)

    if xi != 0:
        pdf[valid] = (1 / sigma) * (t_x[valid] ** (-1 / xi - 1)) * np.exp(-t_x[valid] ** (-1 / xi))
    else:  # Special case when ξ = 0 
        pdf[valid] = (1 / sigma) * np.exp(-(x[valid] - mu_t[valid]) / sigma)**(xi+1) * np.exp(-np.exp(-(x[valid] - mu_t[valid]) / sigma))
    
    return pdf


def gev_ns_cdf(x, t, cmu, mu0, sigma, xi):
    """ 
    Calculate the non-stationary GEV CDF with μ(t) = cmu * t + mu0
        Parameters:
    x : array-like, the values at which to evaluate the CDF.
    t : float or array-like, the time variable for which the non-stationarity is introduced.
    cmu : float, the coefficient for the time-varying location parameter μ(t).
    mu0 : float, the location parameter
    sigma : float, the scale parameter.
    xi : float, the shape parameter

    Returns:
    cdf : array-like, the calculated CDF values for each x at time t.
    The location parameter μ(t) is modeled as a linear function of time:
        μ(t) = cmu * t + mu0
    This allows the GEV distribution to vary with time, representing non-stationarity.
    """
    # Make sure sigma is positive
    assert sigma > 0, "sigma must be positive"

    # Calculate the time-dependent location parameter μ(t)
    mu_t = cmu * t + mu0

    # Calculate the value of t(x) = 1 + xi * (x - μ(t)) / σ
    t_x = 1 + xi * (x - mu_t) / sigma

    # Check if t(x) > 0, otherwise CDF is 0
    valid = t_x > 0
    cdf = np.zeros_like(x)

    if xi == 0:  # Special case when ξ = 0 (limiting distribution)
        cdf[valid] = np.exp(-np.exp(-(x[valid] - mu_t[valid]) / sigma))
    elif xi > 0:  # ξ > 0
        cdf[valid] = np.exp(-t_x[valid] ** (-1 / xi))
    elif xi < 0:  # ξ < 0
        cdf = np.ones_like(x)
        cdf[valid] = np.exp(-t_x[valid] ** (-1 / xi))

    return cdf


def gev_ns_quantile(q, t, cmu, mu0, sigma, xi):
    """
    Calculate the non-stationary GEV (Generalized Extreme Value) quantile function (inverse CDF)
    
    Parameters:
    q : array-like, the quantile values (between 0 and 1) at which to evaluate the inverse CDF.
    t : float or array-like, the time variable for which the non-stationarity is introduced.
    cmu : float, the coefficient for the time-varying location parameter μ(t).
    mu0 : float, the location parameter.
    sigma : float, the scale parameter.
    xi : float, the shape parameter.

    Returns:
    quantile : array-like
        The calculated quantile values for the given q at time t.

    The location parameter μ(t) is modeled as a linear function of time:
        μ(t) = cmu * t + mu0
    This allows the GEV distribution's quantiles to vary with time, representing non-stationarity.
    """
    assert sigma > 0, "sigma must be positive"

    # Calculate the time-dependent location parameter μ(t)
    mu_t = cmu * t + mu0

    if xi != 0:  # Handle the case where ξ ≠ 0
        x = mu_t + (sigma / xi) * ((-np.log(q)) ** (-xi) - 1)
    else:  # Special case for ξ = 0
        x = mu_t - sigma * np.log(-np.log(q))
    
    return x


def neg_log_likelihood(params, x, t):
    """ 
    Negative log-likelihood for non-stationary GEV without using genextreme.
    
    Parameters:
    params : array-like, contains the GEV parameters [xi, mu0, cmu, sigma].
    x : array-like, the observed data.
    t : array-like, the time variable corresponding to each observation in x.

    Returns:
    neg_log_lik : float, the negative log-likelihood of the GEV parameters given the data.
    """
    xi, mu0, cmu, sigma = params
    mu_t = mu0 + cmu * t  # Time-varying location parameter
    
    # Ensure scale (sigma) is positive
    if sigma <= 0:
        return np.inf

    # Calculate t(x) = 1 + xi * (x - mu_t) / sigma
    t_x = 1 + xi * (x - mu_t) / sigma
    
    # Check for invalid values (where t(x) <= 0, the PDF is undefined)
    if np.any(t_x <= 0):
        return np.inf

    # Calculate the log-likelihood
    if xi != 0:
        # General case for non-zero shape parameter (xi != 0)
        log_pdf = -np.log(sigma) - (1 / xi + 1) * np.log(t_x) - t_x ** (-1 / xi)
    else:
        # Special case for zero shape parameter (Gumbel distribution)
        log_pdf = -np.log(sigma) - (x - mu_t) / sigma - np.exp(-(x - mu_t) / sigma)
    
    # Negative log-likelihood
    neg_log_lik = -np.sum(log_pdf)
    
    return neg_log_lik