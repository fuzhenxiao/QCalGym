import warnings
import numpy as np
from scipy.linalg import expm as scipy_expm

def apply_propagator(propagator, states):
    propagator, states = (np.asarray(propagator), np.asarray(states))
    if propagator.ndim != 2 or states.ndim not in (1, 2):
        raise ValueError('Expected a matrix acting on a vector or columns of vectors')
    if not np.isfinite(propagator).all() or not np.isfinite(states).all():
        raise ValueError('Propagation input contains non-finite values')
    contraction = 'ij,j->i' if states.ndim == 1 else 'ij,jk->ik'
    result = np.einsum(contraction, propagator, states, optimize=False)
    if not np.isfinite(result).all():
        raise FloatingPointError('Propagation produced non-finite values')
    return result

def expm(matrix):
    matrix = np.asarray(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError('Matrix exponential input contains non-finite values')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='(divide by zero|overflow|invalid value) encountered in matmul', module='scipy\\.linalg\\._matfuncs$')
        result = scipy_expm(matrix)
    if not np.isfinite(result).all():
        raise FloatingPointError('Matrix exponential produced non-finite values')
    return result
