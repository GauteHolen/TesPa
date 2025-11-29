from numba import njit, prange
import numpy as np


@njit
def periodic_array(r, L):
    """Apply periodic boundary conditions to position r within box of size L."""
    for i in prange(3):
        if r[i] < 0:
            r[i] += L[i]
        elif r[i] >= L[i]:
            r[i] -= L[i]
    return r


@njit
def reflective_array(r,v,lower,upper):
    """Apply reflective boundary conditions to position r within box of size L."""
    for i in prange(3):
        if r[i] < lower:
            r[i] = -r[i]
            v[i] = -v[i]
        elif r[i] >= upper[i]:
            r[i] = 2*upper - r[i]
            v[i] = -v[i]
    return r, v


@njit(inline='always')
def periodic(r,v,lower,upper):
    """Apply periodic boundary conditions to position r within box of size L."""
    if r < lower:
        #print("Applying periodic BC")
        r = upper - 1e-6
    elif r >= upper:
        #print("Applying periodic BC")
        r = lower + 1e-6
    return r, v


@njit(inline='always')
def reflective(r,v,lower,upper):
    """Apply reflective boundary conditions to position r within box of size L."""
    v_coeff = 0.5
    if r < lower:
        r = 2 * lower - r
        v = -v * v_coeff
        #print("Applying reflective BC")
    elif r >= upper:
        r = 2*upper - r
        v = -v * v_coeff
        #print("Applying reflective BC")
    return r, v