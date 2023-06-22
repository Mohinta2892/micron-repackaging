import numpy as np


def encode64(c, bits=None):
    """Encodes a list c into an int64 by progressively shifting
    the elements of a given list

    Parameters:
    c (list): coordinate which will be encoded into a unique int64
    bits (list or None): Amount of bits to represent each value, if
        given, should be the same length as c. If not given, encode64
        will evenly distribute the values across the 64 available bits

    Returns:
    id64 (int): integer representation of the given list.
    """
    id64 = 0
    cum_shift = 0
    if bits is None:
        # If bits are not given we will evenly
        # space the bits
        bits = [64 // len(c) for _ in c]
    for d in range(len(c)):
        id64 |= c[d] << cum_shift
        cum_shift += bits[d]
    return id64


def decode64(id64, dims, bits=None):
    """Decodes a value which was encoded using encode64

    Parameters:
    id64 (int): int64 generated from encode64
    dims (int): number of dimensions in the coordinate
    bits (list or None): bits per item in the list when encoded.

    Returns:
    coord (list): coordinate encoded by this integer
    """
    if bits is None:
        # If bits are not given we will evenly
        # space the bits
        bits = [64 // dims for _ in range(dims)]
    coord = []
    for d in range(dims):
        mask = (1 << bits[d]) - 1
        coord.append(np.uint64(id64 & mask))
        id64 = id64 >> bits[d]
    return coord
