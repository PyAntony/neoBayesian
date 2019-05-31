import math


def UniformDist(first: int, last: int , steps: int, mode='pdf', verbose='no'):
    '''
    Random variable can take any value from a finite sequence with
    equal likelihood.

    Parameters:
    -----------
    first: first value in sequence.
    last: last value in sequence (inclusive).
    steps: steps of the sequence.
    mode: 'pdf' (default), 'var', 'exp', or integer 'k' to compute CDF.
    verbose: print statistics.
    '''
    my_list = list(range(first, last+1, steps))

    p = 1/len(my_list)
    mean = (my_list[0]+my_list[-1])/2
    mean_squared = sum(map(lambda val: p*(val**2), my_list))

    if verbose == 'yes':
        print(f'Length: {len(my_list)}')
        print(f'Mean: {mean}')
        print(f'Mean**2: {mean_squared}')
        print(f'Variance: {mean_squared - mean**2}\n')

    # Return ***********************************
    if mode == 'pdf':
        return p
    if mode == 'var':
        return mean_squared - mean**2
    if mode == 'exp':
        return mean
    if isinstance(mode, int):
        return (1/p*(my_list.index(mode)+1)
                if mode in my_list else
                print(f"{mode} not in sequence. Can't compute CDF.")
        )


def BinomialDist(n: int, k: int, p: float, mode='pdf', results=[]):
    '''
    Calculates probability of exactly 'k' successes in 'n' observations.

    Parameters:
    -----------
    n - number of trials.
    k - number of successful trials.
    p - probability of success of single event.
    mode - 'pdf' (default), 'exp', 'var', or
        'cdf' (adds all probabilities from 0 to k).
    '''
    if k < 0 and mode == 'cdf':
        return sum(results)

    f1 = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    f2 = (p**k)*((1-p)**(n-k))

    if mode == 'pdf':
        return f1*f2
    if mode == 'var':
        return (n*p)*(1-p)
    if mode == 'exp':
        return n*p
    if mode == 'cdf':
        return BinomialDist(n, k-1, p, mode, results+[f1*f2])


def PoissonDist(u, lower_k, upper_k='', results=[]):
    '''
    Calculates probability of 'n' number of independent events
    in a fixed time. Expected value and Variance equals 'u'.
    Formula:
        pdf = P(X=x) = (u**x)*(e**(-u))
                      -----------------
                             x!

    Parameters:
    -----------
    u - mean number of successes in the given time interval or region.
    lower_k - number of successes occurring in a given time interval.
    upper_k - if passed, probabilities are added up to [upper_k] (inclusive).
    '''
    if upper_k and (lower_k > upper_k):
        return sum(results)

    result = ((math.e**(-u))*(u**lower_k))/math.factorial(lower_k)

    if not upper_k:
        return result

    return PoissonDist(u, lower_k+1, upper_k, results+[result])


def GeometricDist(p, k, mode='pdf', results=[]):
    '''
    Calculates the probability of failing 'k' number of times until
    the first success occurs: P(X=k). Mode 'cdf' will calculate all
    probabilities from 0 to 'k' (inclusive) and add them up.
    Formulas:
        cdf = 1 - (1 - p)**(k + 1)
        survival f = (1 - p)**(k + 1)

    Parameters:
    -----------
    p - success probability.
    k - independent trials until first success.
    mode - 'exp', 'var', 'cdf' or 'pdf' (by default).
    '''
    if k < 0 and mode == 'cdf':
        return sum(results)

    result = ((1-p)**k)*p

    if mode == 'pdf':
        return result
    if mode == 'var':
        return (1-p)/(p**2)
    if mode == 'exp':
        return (1-p)/p

    # else if mode == 'cdf'
    return GeometricDist(p, k-1, mode, results+[result])


def GeometricDistParams(p='', expected_value=''):
    '''
    Uses the formula "E(x) = (1-p)/p" to find the "p" parameter
    or the expected value for the Geometric distribution.
    '''
    return (1/(1+expected_value)
            if expected_value and not p else
            (1-p)/p
            if p and not expected_value else
            print('Enter correct arguments.')
           )
