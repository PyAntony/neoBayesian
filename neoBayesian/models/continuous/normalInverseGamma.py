'''
Parameterization:
-----------------
a = alpha = t(tau) + 1 = s/2 - 1
B = beta = r/2
'''
def getMeanVarOrSR(mean=0, variance=0, r=0, s=0):
    '''
    Finds parameters r and s or mean and variance. Formulas:

    r = (2 * mean^3) + (2 * mean)
        ------------
          variance

    s = (2 * mean^2) + 6
        ------------
          variance

    If r and s are passed, return the mean and variance instead. Formulas:

    mean = r/(s-4)
    variance =      2 * r^2
                ---------------
                (s-4)^2 * (s-6)
    '''
    if mean and variance:
        r = (2 * mean**3)/variance + (2*mean)
        s = (2 * mean**2)/variance + 6
        return (r, s)

    if r and s:
        mean = r / (s - 4)
        variance = (2 * r**2) / ((s - 4)**2 * (s - 6))
        return (mean, variance)


def getPosterior(known_mean, observations: list, rs_tuple:tuple):
    '''Calculates posterior r, s, expected value, and variance.'''
    r_ = rs_tuple[0] + sum(map(lambda v: (v - known_mean)**2, observations))
    s_ = rs_tuple[1] + len(observations)
    exp = r_ / (s_ - 4)
    var = (2 * r_**2) / ((s_ - 4)**2 * (s_ - 6))

    return (r_, s_, exp, var)
