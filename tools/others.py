import itertools as it
from neoBayesian.helpers.helpers import *
from tabulate import tabulate


def probaByBruteForce(probability_map, cutoff, iterations):
    '''
    1 - Calculates the probability of a random variable 'X'
        by computing all possible associated probabilities.
    2 - Calculates the Expected value of 'X' conditional on
        probability from (1).
        Formula:
            EXP(X|event) = EXP(X and event)/Pr(event)

    Parameters:
    -----------
    probability_map: dictionary with probabilities for each outcome.
                     keys must be the actual values (integer) for each outcome.
    cutoff: filter set of outcomes >= cutoff value.
    iterations: number of repetitions for each set of outcomes.

    Returns:
    --------
    Tuple: (Probability of the event -> Pr(X=[...]),
            EXP value conditional on the event -> EXP(X|X=[...])).
    '''
    # multiply 1 sequence of events by the number of iterations
    all_outcomes = [sorted(probability_map.keys())]*iterations

    # get tuples with all possible sequences, the sum of points, and its
    # probability of occurrence
    mapAndMultiply = lambda tp: multiply((probability_map[v] for v in tp))
    complete_tuples = map(lambda tp: tuple((tp, sum(tp), mapAndMultiply(tp))),
                          # cartisian products: all possible sequences
                          it.product(*all_outcomes))

    # filter sequence of events above cutoff
    filtered_tuples = filter(lambda tp: tp[1] >= cutoff, complete_tuples)
    results = sorted(filtered_tuples, key=lambda tp: tp[1], reverse=True)

    # final results
    PROBABILITY = sum([tp[-1] for tp in results])
    EXPECTATION = sum([tp[1]*tp[-1] for tp in results])/PROBABILITY

    # rounding and grouping for displaying purposes only --------------
    # round probabilities, sort and group
    roundLast = lambda tp: tp[1:-1] + (round(tp[-1], 8),)
    results_p = sorted([roundLast(tp) for tp in results],
                   key=lambda tp: (tp[0], tp[1]),
                   reverse=True)
    grouped = [(key[0], key[1], len(list(group)))
                for key, group in
                it.groupby(results_p)]

    print(tabulate(grouped, headers=('Event Value', 'Probability', 'Count')))
    print(f'\nEVENT PROBABILITY: {round(PROBABILITY, 6)} <---')
    print(f'Conditional EXP: {round(EXPECTATION, 6)} <---')

    return (PROBABILITY, EXPECTATION)


def getTestStats(sensitivity, specificity, prevalence):
    '''
    Calculates Positive Predictive Value and Negative Predictive Value.
    Formulas:

        PPV = Pr(D+|T+):
                            Sensitivity * Prevalence
        -----------------------------------------------------------------
        (Sensitivity * Prevalence) + (1 - Specificity) * (1 - Prevalence)


        NPV = Pr(D-|T-):
                         Specificity * (1 - Prevalence)
        ---------------------------------------------------------------
        Specificity * (1 - Prevalence) + (1 - Sensitivity) * Prevalence

    Parameters:
    -----------
    Sensitivity: Pr(T+|D+)
    Specificity: Pr(T-|D-)
    Prevalence: Pr(D+)
    '''
    PPV = ((sensitivity*prevalence)/
          ((sensitivity*prevalence) + (1-specificity)*(1-prevalence))
          )
    NPV = ((specificity*(1-prevalence))/
          ((specificity*(1-prevalence)) + (1-sensitivity)*prevalence)
          )

    print(f'PPV={round(PPV, 5)}')
    print(f'NPV={round(NPV, 5)}')

    return (PPV, NPV)
