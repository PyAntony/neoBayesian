from scipy.stats import norm


def getParamsFromInterval(interval: tuple, perct):
    '''
    Calculates mean and variance of a normal distribution
    from and interval of values and the percentage contained.

    Parameters:
    ----------
    interval: lower and upper boundaries.
    percentage: percentage contained within boundaries.
    '''
    zscores = {80:1.282, 85:1.440, 90:1.645, 95:1.960,
               99:2.576, 99.5:2.807}

    mean = sum(interval)/2
    std = (interval[1] - interval[0]) / (2*zscores[perct])

    return (mean, std**2)


def getLikelihood(distParams: tuple, distRange: tuple):
    '''
    Calculate likelihood of a normal random variable.

    Parameters:
    -----------
    distParams: mean and variance.
    distRange: upper and lower limits for CDF.
    '''
    mean, variance = distParams

    return (norm.cdf(distRange[1], loc=mean, scale=variance**0.5)
            - norm.cdf(distRange[0], loc=mean, scale=variance**0.5)
    )


def getPosterior(x: list or int, sample_v, prior_m, prior_v, ci=0):
    '''
    Calculates the posterior parameters (mean and variance) of
    a normal distribution after observation/s.

    NOTE: for non-informative uniform priors use 0 for 'prior_m' and 'prior_v'.

    Parameters:
    -----------
    x - observations (or means). Single observation can be passed as int.
        If you are only given the 'mean' and the number of observations 'n',
        but not the actual observations, you can pass: ['mean']*'n'.
    sample_v - variance of the sample.
    prior_m - prior mean.
    prior_v - prior variance.
    ci - confidence interval. If used, pass the number needed in the
         zscores dictionary.
    '''
    x = x if isinstance(x, list) else [x]

    mean = (((prior_m*sample_v) + (sum(x)*prior_v)) /
            (sample_v + len(x)*prior_v)
    )
    variance = ((sample_v * prior_v) /
                (sample_v + len(x)*prior_v)
    )

    if prior_m == 0 and prior_v == 0:
        mean = sum(x)/len(x)
        variance = sample_v/len(x)

    zscores = {80:1.282, 85:1.440, 90:1.645, 95:1.960, 99:2.576, 99.5:2.807}
    if ci:
        lower = round(mean - zscores[ci]*(variance**0.5), 3)
        upper = round(mean + zscores[ci]*(variance**0.5), 3)

    return (mean, variance) if not ci else (mean, variance, (lower, upper))
