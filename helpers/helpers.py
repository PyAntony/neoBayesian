import operator
from functools import reduce
import csv


def multiply(tp: tuple or list):
    '''Reduce collection by multiplication.'''
    return reduce(operator.mul, tp, 1)


def roundy(obj: int or tuple, n=7):
    '''Round numbers or inner numbers from nested tuples.'''
    return (round(obj, n) if isinstance(obj, (float, int))
            else tuple(roundy(v, n) for v in obj)
    )


def prepareValues(tuples, wSign='*', bSign='+', fnum=3):
    '''
    Prepare list of values to be displayed by rounding numbers and
    adding signs between values.

    Parameters
    ----------
    tuples: list of values (either list of numbers or tuples of pairs)
    wSign: within tuple sign.
    bSign: between tuples sign.
    fnum: to round floats.
    '''
    innerDisp = lambda tp: f'({tp[0]} {wSign} {tp[1]})'
    outerDisp = lambda tuples: f' {bSign} '.join(map(innerDisp, tuples))
    disp = lambda ls: f' {bSign} '.join(map(lambda n: str(n), ls))

    return (outerDisp(roundy(tuples, fnum))
            if isinstance(tuples[0], tuple)
            else disp(roundy(tuples, fnum))
    )


def doubleReduction(tpList, flag=''):
    '''
    Reduce tuples by multiplication and sum the results.
    If any flag is passed, the first element of each
    tuple will be squared first.
    '''
    squared = lambda tp: (tp[0]**2, tp[1])
    ls = tpList if not flag else list(map(squared, tpList))

    return sum(map(lambda tp: reduce(operator.mul, tp, 1), ls))


def csvAsDicts(path):

    with open(path, 'r', newline='', encoding='utf-8') as ofile:
        return list(csv.DictReader(ofile))
