import math

import operator
from functools import reduce
import itertools as it

from tabulate import tabulate
from neoBayesian.helpers.helpers import *


def splitDF(rows, target):
    '''split list of dictionaries into training sets.'''
    # list of key-value pairs + 'n' and target column
    x_train = [dc for dc in rows if dc[target]]
    # list of key-value pairs
    x_test = [list(dc.items())[:-2] for dc in rows if not dc[target]]
    # unique target categories(strings)
    targets = set([dc[target] for dc in rows if dc[target]])

    return x_train, x_test, targets


def getProbaTuples(x_train, x_test, target, target_col):
    '''
    Computes the conditional probability for each set of observations
    given the target. Assumes conditional independence.
    '''
    # total number of observations from target category
    tbase = sum([int(dc['n']) for dc in x_train if dc[target_col]==target])
    # total number of observations
    total = sum([int(dc['n']) for dc in x_train])

    # get dictionary in target group if column equals value (dc[tp[0]]==tp[1])
    filterDc = lambda dc, tp: dc[tp[0]]==tp[1] and dc[target_col]==target
    def getProba(tp):
        # filter observations by each group passed and get numbers
        countGroup = [int(dc['n']) for dc in x_train if filterDc(dc, tp)]
        return sum(countGroup)/tbase

    # v_tuple: key-value pair for each predictor variable (column name-value)
    packTuple = lambda v_tuple: (v_tuple[0], v_tuple[1], getProba(v_tuple))
    # list of tuples with key, value, and conditional probability
    conditionals = list(map(packTuple, x_test))
    # intersection: multiplication of all conditionals and target probability
    inter = multiply([tp[-1] for tp in conditionals]) * (tbase/total)
    infoDC = {'tgCol':target_col, 'tgName':target, 'tgProba':tbase/total,
              'support':tbase, 'intersec':inter}

    return conditionals + [infoDC]


def display(ls):
    '''Display conditional and Intersection probabilities per target.'''
    def iterOverTarget(l):
        dc = l[-1]
        tProba = round(dc['tgProba'], 8)
        intersection = round(dc['intersec'], 8)
        head = ('Var', 'Category', 'Probability')

        print(f"\nPr(Target => {dc['tgCol']}:{dc['tgName']}):", tProba)
        print('Support:', dc['support'])
        print(f"\nPr(Observations | {dc['tgName']}):")
        print(tabulate(l[:-2], headers=head))
        print(f"\nIntersection = {intersection}")

    [iterOverTarget(l) for l in ls]
    head = ('Target', 'Proba')
    print('\n RESULTS:')
    print(tabulate(getFinalProbas(ls), tablefmt="fancy_grid", headers=head))


def getIntersections(ls):
    '''Extract intersection probabilities'''
    def extractVals(l):
        targetName = l[-2][1]
        pr = l[-1][-2]
        return (targetName, pr)

    return [extractVals(l) for l in ls]


def getFinalProbas(ls):
    '''
    Computes final probabilities. Formula:

        Pr(Observations|Target)    => Intersection probability
     -----------------------------
     sum(Pr(Observations|Targets)) => Sum of intersections
    '''
    def getIntersection(l):
        '''Get Intersection probability from infoDC.'''
        return l[-1]['intersec']

    # add all intersection probabilities
    marginal = sum([getIntersection(l) for l in ls])
    # return 1 tuple per target
    return [(l[-1]['tgName'], round(l[-1]['intersec']/marginal, 4))
            for l in ls]


def pyNaiveBayes(csv_path, target_col, verbose=False, display_n=5):
    '''
    Calculates Probability of conditional event for each target
    (--> Pr(Target Category|x-1 AND x-2 AND ...x-n )) for a single
    observation using Naive Bayes (conditional independence assumption).
    Formula:

    Pr(Target | x-1 AND x-2 AND ...x-n) =

        Pr(x-1|Target) * Pr(x-2|Target)... * Pr(x-n|Target)
        ---------------------------------------------------
        Pr(x-1|Target) * Pr(x-2|Target)... * Pr(x-n|Target)
        + Pr(x-1|Target`) * Pr(x-2|Target`)... * Pr(x-n|Target`)

    Parameters:
    -----------
    csv_path: path of csv file. Requirements:
              - Last 2 columns must be the 'target' & count 'n'
    target_col: column name with 'target' (y_train).
    '''
    # split the CSV file =>
    # x_train: list of dictionaries. Each dc represents 1 observation where
    #          'key' if the column name and 'value' the value in that row.
    # x_test: list of observations where target column is null.
    # targets: a set of each target/category in the target column.
    x_train, x_tests, targets = splitDF(csvAsDicts(csv_path), target_col)

    def getProbabilities(x_test):
        # get probability for each independent conditional event in x_test
        perTarget = lambda t: getProbaTuples(x_train, x_test, t, target_col)
        return list(map(perTarget, targets))

    allProbabilities = list(map(getProbabilities, x_tests))

    if verbose:
        title = '\n-----> OBSERVATION:'
        itDisplay = lambda n: print(title, n) or display(allProbabilities[n])
        [itDisplay(n) for n in range(display_n)]

    return [getFinalProbas(ls) for ls in allProbabilities]
