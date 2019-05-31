
import math
import csv

import operator
from functools import reduce
import itertools as it

from tabulate import tabulate
from neoBayesian.helpers.helpers import *


def calculatePosteriorProba(numerator: tuple, denominator: list):
    '''
    Calculates posterior probability with given conditional probabilities
    (p1, p2, ..., pn) and weights.
    Formula:
                       (p1*weight1)
        -----------------------------------------------
        (p1*weight1) + (p2*weight2) + ... (pn*weight1n)

    Remember: (p1*weight1) is added to the denominator when computed.

    Parameters:
    -----------
    numerator: tuple with conditional probability and weight probability.
    denominator: list of tuples with the other (compliment) conditional
                 probabilities and weight probabilities.

    Returns:
    --------
    posterior probability.
    '''
    result = multiply(numerator)/doubleReduction(denominator+[numerator])

    nString = prepareValues(numerator, bSign='*')
    dString = prepareValues(denominator + [numerator])

    print(' '*int(len(dString)/2 - len(nString)/2), nString)
    print('-'*len(dString))
    print(dString)
    print('\n=', round(result, 6))

    return result


def tabulateBayesianAlgorithm(*args):
    '''
    Displays tabular form of Bayesian algorithm.
    Multiple tuples can be passed with the following format:
    ('label1', 'prior1', 'likelihood1'),
    ('label2', 'prior2', 'likelihood2'), ...
    ('labeln', 'priorn', 'likelihoodn').

    Labels must be unique.
    '''
    p_TYPE = ('Type', ('Prior', 'Likelihood', 'Joint', 'Posterior'))

    keyValuePair = lambda tp: tuple([tp[0], tp[1:] + (multiply(tp[1:]), )])
    ls_tabulate = list(map(keyValuePair, args))

    marginal_proba = sum([tp[1][-1] for tp in ls_tabulate])

    kvPairWithMarginal = (
        lambda tp: tuple([tp[0], tp[1] + (tp[1][-1]/marginal_proba, )])
    )

    dcToTabulate = dict([p_TYPE] + list(map(kvPairWithMarginal, ls_tabulate)))

    print(tabulate(dcToTabulate, headers="keys"))
    print('\nMarginal Probability = sum([joint probabilities])')
    print(f'= {round(marginal_proba, 4)}')


def getTotalExpectationOrProbability(tuples_ls, *args):
    '''
    Gets total expectation or probability from multiple tuples.

    Tuples format:
        Total expectation: ('n', 'p', 'prior')
        Total probability: ('p', 'prior')
    '''
    return sum(map(multiply, args if not tuples_ls else tuples_ls))
