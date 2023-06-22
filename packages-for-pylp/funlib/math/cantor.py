from __future__ import division
import numpy as np


def cantor_number(coordinate):
    '''Return the "Cantor number" of tuples of arbitrary length.

    The Cantor number is the generalization of Cantor's mapping of rational
    numbers (2-tuples) to natural numbers for arbitrary dimensions.

    The numbers obtained this way can be used to uniquely index a positive
    integer coordinate in a volume of unknown size.
    '''

    if len(coordinate) == 1:
        return coordinate[0]

    return (
        pyramide_volume(len(coordinate), sum(coordinate)) +
        cantor_number(coordinate[:-1]))


def inv_cantor_number(c, dims=3):
    '''
    Return coordinate of a given cantor number

    P = pyramide_volume

    x+y+z < P_inv(c) < x+y+z+1

    This is true for any given set of coordinates ( at least 2 and 3 )
    Due to the fact that both the pyramide volume function
    and its inverse if monotonically increasing on the interval [0,inf)

    Inverse function for pyramide volume is nontrivial however
    i have calculated it by hand for 2 and 3 dimensions.
    i actually used wolfram to solve for the inverse function in 3 dimensions
    '''
    if dims == 1:
        return [int(c)]
    el = inv_pyramide_volume(c, dims)
    pv = int(pyramide_volume(dims, el))
    coord = inv_cantor_number(c - pv, dims - 1)
    coord.append(int(el - sum(coord)))
    return coord


def pyramide_volume(dims, edge_length):
    if edge_length == 0:
        return 0
    v = 1
    for d in range(dims):
        v *= edge_length + d
        v /= d + 1
    return int(v)


def inv_pyramide_volume(c, dims):
    el = int(floating_inv_pyramide_volume(c, dims))
    pv = pyramide_volume(dims, el)
    if c == pv or dims == 1:
        return el
    increment = 1 if pv < c else -1
    while True:
        new_el = el + increment
        pv = pyramide_volume(dims, int(new_el))
        if increment > 0 and pv > c:
            break
        elif increment < 0 and pv <= c:
            el = new_el
            break
        else:
            el = new_el
    return el


def floating_inv_pyramide_volume(c, dims):
    '''
    This is gross however I am not sure there is a simple way to generalize
    the inverse of pyramide_volume. I  have calculated it for 3 and 2 dimension
    '''
    if c == 0:  # the dims 3 function is not defined at c==0
        return 0
    if dims == 3:
        # this is an approximation to the commented
        # out function that avoids squaring c
        return int((np.sqrt(729) * c + 27 * c)**(1 / 3) / 3**(2 / 3) +
                   1/(3 * np.sqrt(729) * c + 81 * c)**(1 / 3) - 1)
    if dims == 2:
        return int(np.floor((np.sqrt(8*c + 1) - 1)/2))
    if dims == 1:
        return 0
