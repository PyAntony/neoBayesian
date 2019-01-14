from tabulate import tabulate


def getMeanVarOrAlphaBeta(mean=0, variance=0, alpha=0, beta=0):
    '''
    Find parameters alpha and beta or mean and variance.
    Formulas:
        alpha = (mean^2 - mean^3 - variance * mean)/variance
        beta = (alpha * (1 - mean))/mean

    If alpha and beta are passed, return the mean and variance instead.
    Formula:
        mean = alpha/(alpha + beta)
        variance =               alpha * beta
                    --------------------------------------
                    (alpha * beta)**2 * (alpha + beta + 1)
    '''
    if mean and variance:
        alpha = (mean**2 - mean**3 - variance*mean)/variance
        beta = (alpha*(1-mean))/mean
        return (alpha, beta)

    if alpha and beta:
        mean = alpha/(alpha+beta)
        variance = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
        return (mean, variance)


def getPosterior(N:int, k:int, alphabeta_tuple:tuple, verbose=True):
    '''
    Calculates posterior alpha, beta, expected value, and variance.
    expected value:
        E[p|Data] = alpha_ / (alpha_ + beta_)

    variance:
                    (alpha_ * beta_)
       -----------------------------------------
       (alpha_ + beta_ + 1) * (alpha_ + beta_)^2

    Parameters:
    -----------
    N - sample size.
    k - number of successful events.
    alphabeta_tuple - prior alpha and beta parameters.
    '''
    alpha_ = alphabeta_tuple[0] + k
    beta_ = alphabeta_tuple[1] + (N - k)
    exp =  (alpha_/(alpha_+beta_))
    var = (alpha_ * beta_) / ((alpha_+beta_+1) * (alpha_+beta_)**2)

    if verbose:
        print(f'N: {N}, k: {k}')
        print('\nPosterior:')
        print(f'\talpha-> {round(alpha_, 3)}',
              f'\n\tbeta-> {round(beta_, 3)},',
              f'\n\texpected-> {round(exp, 3)}',
              f'\n\tvariance-> {round(var, 3)}.\n')

    return (alpha_, beta_, exp, var)


def iterOverK(priors: tuple, true_p: float, cutoff: float, k=1, acc=[]):
    '''
    Calculates expected values (0 t0 100%) of a beta distribution
    increasing the number of successful events by one and
    keeping the ratio of k/N equals to the true
    probability "true_p". Function stops when last expected value
    is >= cutoff. Parameters Are updated after each observation.
    REMEMBER: given a true probability 'p', EXP value will never equals 'p'
    so the cutoff must be always less than 'true_p'.

    Parameters:
    -----------
    priors - starting alpha and beta tuple.
    true_p - the observed probability. Used to keep the k/N ratio
             over each function call.
    cutoff - expected value to stop recursion.
    k - starting number of successful events.
    '''
    a, b, exp, var = getPosterior(k/true_p, k, priors, verbose=False)
    tp = (k/true_p, k, a, b, exp, var)

    if exp >= cutoff:
        headers=['N', 'k', 'Alpha', 'Beta', 'Expected', 'Variance']
        print(tabulate(acc+[tp], headers=headers))
        return
    else:
        iterOverK((a, b), true_p, cutoff, k+1, acc + [tp])
