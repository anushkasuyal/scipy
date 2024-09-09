import numpy as np
from numpy import asarray
from scipy.sparse import (issparse, coo_array)
from scipy.sparse._sputils import is_pydata_spmatrix, convert_pydata_sparse_to_scipy
from scipy.linalg import LinAlgError
from scipy.sparse.linalg._dsolve.linsolve import spsolve
import math

def tensorsolve(A, b, axes=None):
    is_pydata_sparse = is_pydata_spmatrix(b)
    pydata_sparse_cls = b.__class__ if is_pydata_sparse else None
    A = convert_pydata_sparse_to_scipy(A)
    b = convert_pydata_sparse_to_scipy(b)

    try:
        A.shape
    except AttributeError:
        A = asarray(A)

    an = A.ndim

    if axes is not None:
        allaxes = list(range(0, an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        A = A.transpose(allaxes)

    try:
        b.shape
    except AttributeError:
        b = asarray(b)
        
    if math.prod(A.shape[b.ndim:]) != math.prod(A.shape[:b.ndim]):
        raise LinAlgError(
            "Input arrays must satisfy the requirement \
            prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )

    if A.ndim<=2:
        return spsolve(A, b)

    if A.shape[:b.ndim] != b.shape:
        raise LinAlgError(
            "A must have shape = b.shape + Q"
        )

    # if n-D tensorsolve
    left_shape = A.shape[:b.ndim]
    left_coords = A.coords[:b.ndim]
    
    right_shape = A.shape[b.ndim:] 
    right_coords = A.coords[b.ndim:]
    
    # Ravel the coordinates into 1D
    left_raveled_coords = np.ravel_multi_index(left_coords, left_shape)
    right_raveled_coords = np.ravel_multi_index(right_coords, right_shape)

    a_2d_coords = np.vstack((left_raveled_coords, right_raveled_coords))
    
    if not issparse(b):
        b = asarray(b)
        # b_2d_coords = np.ravel_multi_index(b.coords, b.shape)
    
    nD_res_shape = A.shape[b.ndim:]

    A = coo_array((A.data, a_2d_coords), (math.prod(b.shape), math.prod(b.shape)))
    b = b.reshape(-1) # or ravel for np array?

    x = spsolve(A, b)

    assert(isinstance(x, np.ndarray))
    return x.reshape(nD_res_shape)
