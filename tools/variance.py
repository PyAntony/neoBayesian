import math
import csv

import operator
from functools import reduce
import itertools as it

from tabulate import tabulate
from neoBayesian.helpers.helpers import *
from neoBayesian.models.discrete import BinomialDist


def varianceWithFormula(variances: list, exp_values: list, weights: list):
    '''
    Gets variance by applying the GROUPS formula
    (Conditional Variance Formula). Formulas:

    within: Ew[VARx[X|W]] -> expected value of the variances.
    between: VARw[Ex[X|W]] -> variance of the expected values.

    Parameters:
    -----------
    (lists' elements must be passed in order.)
    variances: list of variances.
    exp_values: list of expected values.
    weights: list of weights.

    Returns:
    --------
    Ew[VARx[X|W]] + VARw[Ex[X|W]]
    '''
    within_tuples = list(zip(variances, weights))
    between_tuples = list(zip(exp_values, weights))

    within_multiplied = list(map(multiply, within_tuples))
    within_variance = sum(within_multiplied)

    between_multiplied = list(map(multiply, between_tuples))
    mean_of_Xsquared = doubleReduction(between_tuples, 'square')
    mean_squared = sum(between_multiplied)**2
    between_variance = mean_of_Xsquared - mean_squared

    VARIANCE = within_variance + (mean_of_Xsquared - mean_squared)

    # Display results --------------------------------------------
    print('Within variance values:')
    print('-->', prepareValues(within_tuples))
    print('--> Var[X=x|W_n] * W_n:', prepareValues(within_multiplied))

    print('\nBetween expected values:')
    print('-->', prepareValues(between_tuples))
    print('--> E[X=x|W_n] * W_n:', prepareValues(between_multiplied))

    print('\nE[X]: ', round(sum(between_multiplied), 3))
    print('E[X]^2: ', round(mean_squared, 3))
    print('E[X^2]: ', round(mean_of_Xsquared, 3))
    print('---------------------------------------')
    print(f'\nWithin-group variance: {round(within_variance, 5)}')
    print(f'Between-group variance: {round(between_variance, 5)}')
    print(f'VARIANCE: {round(VARIANCE, 5)}')

    return VARIANCE


def varianceBruteForce(k: int, tuples_ls=[], dist='binomial'):
    '''
    Computes the variance by the standard method. It takes the total
    unconditional probability for each value of 'k' (0 to 'k')
    and computes:
        Var[X] = E[X^2] - E[X]^2

    (Supports Binomial distribution only)

    Parameters:
    -----------
    tuples_ls: [(Proba1, W1), (Proba2, W2), ..., (Proba.n, W.n)]
    k: sample size (X = [0 to k]).
    '''
    def iterUntilK(i=0, acc=[]):

        if i > k: return acc

        getProbabilityTuples = lambda tp: (BinomialDist(k, i, tp[0]), tp[1])
        probabilities = list(map(getProbabilityTuples, tuples_ls))

        print(f'Pr(X = k:{i}) --> ', prepareValues(probabilities))
        return iterUntilK(i+1, acc+[(i, doubleReduction(probabilities))])

    probaTuples_ls = iterUntilK()

    E_x = doubleReduction(probaTuples_ls)
    E_x2 = doubleReduction(probaTuples_ls, 'square')

    print('\n--------------')
    print('E[X]: ', round(E_x, 3))
    print('E[X]^2: ', round(E_x**2, 3))
    print('E[X^2]: ', round(E_x2, 3))
    print('\nVar[X]: ', round(E_x2 - E_x**2, 3))

    return E_x2 - E_x**2
