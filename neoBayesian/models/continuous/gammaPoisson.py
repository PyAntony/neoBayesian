'''
Parameterization:
-----------------
a = alpha = theta + 1
'''
def getMeanVarOrThetaBeta(mean=0, variance=0, theta=0, beta=0):
    '''
    Finds parameters theta and beta or mean and variance.
    Formulas:
        beta = mean/variance
        theta = (beta * mean) - 1

    If theta and beta are passed, return the mean and variance instead.
    Formulas:
        mean = (theta + 1) / beta
        variance = (theta + 1)/beta^2
    '''
    if mean and variance:
        beta = mean/variance
        theta = (beta * mean) - 1
        return (beta, theta)

    if theta and beta:
        mean = (theta + 1) / beta
        variance = (theta + 1)/beta**2
        return (mean, variance)


def getPosterior(beta, theta, observations: list):
    '''Calculates posterior beta, theta, expected value, and variance.'''
    beta_ = beta + len(observations)
    theta_ = theta + sum(observations)
    exp =  (theta_ + 1) / beta_
    var = (theta_ + 1) / beta_**2

    return (beta_, theta_, exp, var)
