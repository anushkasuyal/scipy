""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_array', 'coo_matrix', 'isspmatrix_coo']

import math
from warnings import warn

import numpy as np

from .._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._sparsetools import (coo_tocsr, coo_todense, coo_todense_nd,
                           coo_matvec, coo_matvec_nd, coo_matmat_dense)
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast_char, to_native, isshape, getdtype,
                       getdata, downcast_intp_index, get_index_dtype,
                       check_shape, check_reshape_kwargs, isscalarlike, isdense)

import operator


class _coo_base(_data_matrix, _minmax_mixin):
    _format = 'coo'
    _allow_nd = range(1, 65)

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _data_matrix.__init__(self, arg1, maxprint=maxprint)
        if not copy:
            copy = copy_if_needed

        if isinstance(arg1, tuple):
            if isshape(arg1, allow_nd=self._allow_nd):
                self._shape = check_shape(arg1, allow_nd=self._allow_nd)
                idx_dtype = self._get_index_dtype(maxval=max(self._shape))
                data_dtype = getdtype(dtype, default=float)
                self.coords = tuple(np.array([], dtype=idx_dtype)
                                     for _ in range(len(self._shape)))
                self.data = np.array([], dtype=data_dtype)
                self.has_canonical_format = True
            else:
                try:
                    obj, coords = arg1
                except (TypeError, ValueError) as e:
                    raise TypeError('invalid input format') from e

                if shape is None:
                    if any(len(idx) == 0 for idx in coords):
                        raise ValueError('cannot infer dimensions from zero '
                                         'sized index arrays')
                    shape = tuple(operator.index(np.max(idx)) + 1
                                  for idx in coords)
                self._shape = check_shape(shape, allow_nd=self._allow_nd)
                idx_dtype = self._get_index_dtype(coords,
                                                  maxval=max(self.shape),
                                                  check_contents=True)
                self.coords = tuple(np.array(idx, copy=copy, dtype=idx_dtype)
                                     for idx in coords)
                self.data = getdata(obj, copy=copy, dtype=dtype)
                self.has_canonical_format = False
        else:
            if issparse(arg1):
                if arg1.format == self.format and copy:
                    self.coords = tuple(idx.copy() for idx in arg1.coords)
                    self.data = arg1.data.copy()
                    self._shape = check_shape(arg1.shape, allow_nd=self._allow_nd)
                    self.has_canonical_format = arg1.has_canonical_format
                else:
                    coo = arg1.tocoo()
                    self.coords = tuple(coo.coords)
                    self.data = coo.data
                    self._shape = check_shape(coo.shape, allow_nd=self._allow_nd)
                    self.has_canonical_format = False
            else:
                # dense argument
                M = np.asarray(arg1)
                if not isinstance(self, sparray):
                    M = np.atleast_2d(M)
                    if M.ndim != 2:
                        raise TypeError(f'expected 2D array or matrix, not {M.ndim}D')

                self._shape = check_shape(M.shape, allow_nd=self._allow_nd)
                if shape is not None:
                    if check_shape(shape, allow_nd=self._allow_nd) != self._shape:
                        message = f'inconsistent shapes: {shape} != {self._shape}'
                        raise ValueError(message)

                index_dtype = self._get_index_dtype(maxval=max(self._shape))
                coords = M.nonzero()
                self.coords = tuple(idx.astype(index_dtype, copy=False)
                                     for idx in coords)
                self.data = M[coords]
                self.has_canonical_format = True

        if len(self._shape) > 2:
            self.coords = tuple(idx.astype(np.int64, copy=False) for idx in self.coords)

        if dtype is not None:
            newdtype = getdtype(dtype)
            self.data = self.data.astype(newdtype, copy=False)

        self._check()

    @property
    def row(self):
        if self.ndim > 1:
            return self.coords[-2]
        result = np.zeros_like(self.col)
        result.setflags(write=False)
        return result


    @row.setter
    def row(self, new_row):
        if self.ndim < 2:
            raise ValueError('cannot set row attribute of a 1-dimensional sparse array')
        new_row = np.asarray(new_row, dtype=self.coords[-2].dtype)
        self.coords = self.coords[:-2] + (new_row,) + self.coords[-1:]

    @property
    def col(self):
        return self.coords[-1]

    @col.setter
    def col(self, new_col):
        new_col = np.asarray(new_col, dtype=self.coords[-1].dtype)
        self.coords = self.coords[:-1] + (new_col,)

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape, allow_nd=self._allow_nd)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        # When reducing the number of dimensions, we need to be careful about
        # index overflow. This is why we can't simply call
        # `np.ravel_multi_index()` followed by `np.unravel_index()` here.
        flat_coords = _ravel_coords(self.coords, self.shape, order=order)
        if len(shape) == 2:
            if order == 'C':
                new_coords = divmod(flat_coords, shape[1])
            else:
                new_coords = divmod(flat_coords, shape[0])[::-1]
        else:
            new_coords = np.unravel_index(flat_coords, shape, order=order)

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of `new_coords` regardless.
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return self.__class__((new_data, new_coords), shape=shape, copy=False)

    reshape.__doc__ = _spbase.reshape.__doc__

    def _getnnz(self, axis=None):
        if axis is None or (axis == 0 and self.ndim == 1):
            nnz = len(self.data)
            if any(len(idx) != nnz for idx in self.coords):
                raise ValueError('all index and data arrays must have the '
                                 'same length')

            if self.data.ndim != 1 or any(idx.ndim != 1 for idx in self.coords):
                raise ValueError('coordinates and data arrays must be 1-D')

            return int(nnz)

        if axis < 0:
            axis += self.ndim
        if axis >= self.ndim:
            raise ValueError('axis out of bounds')

        return np.bincount(downcast_intp_index(self.coords[1 - axis]),
                           minlength=self.shape[1 - axis])

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        self.sum_duplicates()
        if axis is None:
            return np.count_nonzero(self.data)

        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            raise ValueError('axis out of bounds')
        mask = self.data != 0
        coord = self.coords[1 - axis][mask]
        return np.bincount(downcast_intp_index(coord), minlength=self.shape[1 - axis])

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def _check(self):
        """ Checks data structure for consistency """
        if self.ndim != len(self.coords):
            raise ValueError('mismatching number of index arrays for shape; '
                             f'got {len(self.coords)}, expected {self.ndim}')

        # index arrays should have integer data types
        for i, idx in enumerate(self.coords):
            if idx.dtype.kind != 'i':
                warn(f'index array {i} has non-integer dtype ({idx.dtype.name})',
                     stacklevel=3)

        idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.shape))
        self.coords = tuple(np.asarray(idx, dtype=idx_dtype)
                             for idx in self.coords)
        self.data = to_native(self.data)

        if self.nnz > 0:
            for i, idx in enumerate(self.coords):
                if idx.max() >= self.shape[i]:
                    raise ValueError(f'axis {i} index {idx.max()} exceeds '
                                     f'matrix dimension {self.shape[i]}')
                if idx.min() < 0:
                    raise ValueError(f'negative axis {i} index: {idx.min()}')

    def transpose(self, axes=None, copy=False):
        if axes is None:
            axes = range(self.ndim)[::-1]
        elif isinstance(self, sparray):
            if not hasattr(axes, "__len__") or len(axes) != self.ndim:
                raise ValueError("axes don't match matrix dimensions")
            if len(set(axes)) != self.ndim:
                raise ValueError("repeated axis in transpose")
        elif axes != (1, 0):
            raise ValueError("Sparse matrices do not support an 'axes' "
                             "parameter because swapping dimensions is the "
                             "only logical permutation.")

        permuted_shape = tuple(self._shape[i] for i in axes)
        permuted_coords = tuple(self.coords[i] for i in axes)
        return self.__class__((self.data, permuted_coords),
                              shape=permuted_shape, copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def resize(self, *shape) -> None:
        shape = check_shape(shape, allow_nd=self._allow_nd)
        if self.ndim > 2:
            raise ValueError("only 1-D or 2-D input accepted")
        if len(shape) > 2:
            raise ValueError("shape argument must be 1-D or 2-D")
        # Check for added dimensions.
        if len(shape) > self.ndim:
            flat_coords = _ravel_coords(self.coords, self.shape)
            max_size = math.prod(shape)
            self.coords = np.unravel_index(flat_coords[:max_size], shape)
            self.data = self.data[:max_size]
            self._shape = shape
            return

        # Check for removed dimensions.
        if len(shape) < self.ndim:
            tmp_shape = (
                self._shape[:len(shape) - 1]  # Original shape without last axis
                + (-1,)  # Last axis is used to flatten the array
                + (1,) * (self.ndim - len(shape))  # Pad with ones
            )
            tmp = self.reshape(tmp_shape)
            self.coords = tmp.coords[:len(shape)]
            self._shape = tmp.shape[:len(shape)]

        # Handle truncation of existing dimensions.
        is_truncating = any(old > new for old, new in zip(self.shape, shape))
        if is_truncating:
            mask = np.logical_and.reduce([
                idx < size for idx, size in zip(self.coords, shape)
            ])
            if not mask.all():
                self.coords = tuple(idx[mask] for idx in self.coords)
                self.data = self.data[mask]

        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    def toarray(self, order=None, out=None):
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and not B.flags.c_contiguous:
            raise ValueError("Output array must be C or F contiguous")
        # This handles both 0D and 1D cases correctly regardless of the
        # original shape.
        if self.ndim == 1:
            coo_todense_nd(np.array([1]), self.nnz, self.ndim,
                           self.coords[0], self.data, B.ravel('A'), fortran)
        elif self.ndim == 2:
            M, N = self.shape
            coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                        B.ravel('A'), fortran)
        else:
            if fortran:
                strides = np.append(1, np.cumprod(self.shape[:-1]))
            else:
                strides = np.append(np.cumprod(self.shape[1:][::-1])[::-1], 1)
            coords = np.concatenate(self.coords)
            coo_todense_nd(strides, self.nnz, self.ndim,
                           coords, self.data, B.ravel('A'), fortran)
        # Note: reshape() doesn't copy here, but does return a new array (view).
        return B.reshape(self.shape)

    toarray.__doc__ = _spbase.toarray.__doc__

    def tocsc(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.ndim != 2:
            raise ValueError(f'Cannot convert. CSC format must be 2D. Got {self.ndim}D')
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)
        else:
            from ._csc import csc_array
            indptr, indices, data, shape = self._coo_to_compressed(csc_array._swap)

            x = self._csc_container((data, indices, indptr), shape=shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.ndim > 2:
            raise ValueError(f'Cannot convert. CSR must be 1D or 2D. Got {self.ndim}D')
        if self.nnz == 0:
            return self._csr_container(self.shape, dtype=self.dtype)
        else:
            from ._csr import csr_array
            arrays = self._coo_to_compressed(csr_array._swap, copy=copy)
            indptr, indices, data, shape = arrays

            x = self._csr_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def _coo_to_compressed(self, swap, copy=False):
        """convert (shape, coords, data) to (indptr, indices, data, shape)"""
        M, N = swap(self._shape_as_2d)
        # convert idx_dtype intc to int32 for pythran.
        # tested in scipy/optimize/tests/test__numdiff.py::test_group_columns
        idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.nnz, N))

        if self.ndim == 1:
            indices = self.coords[0].copy() if copy else self.coords[0]
            nnz = len(indices)
            indptr = np.array([0, nnz], dtype=idx_dtype)
            data = self.data.copy() if copy else self.data
            return indptr, indices, data, self.shape

        # ndim == 2
        major, minor = swap(self.coords)
        nnz = len(major)
        major = major.astype(idx_dtype, copy=False)
        minor = minor.astype(idx_dtype, copy=False)

        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty_like(minor, dtype=idx_dtype)
        data = np.empty_like(self.data, dtype=self.dtype)

        coo_tocsr(M, N, nnz, major, minor, self.data, indptr, indices, data)
        return indptr, indices, data, self.shape

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todia(self, copy=False):
        if self.ndim != 2:
            raise ValueError(f'Cannot convert. DIA format must be 2D. Got {self.ndim}D')
        self.sum_duplicates()
        ks = self.col - self.row  # the diagonal for each nonzero
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # probably undesired, should todia() have a maxdiags parameter?
            warn(f"Constructing a DIA matrix with {len(diags)} diagonals "
                 "is inefficient",
                 SparseEfficiencyWarning, stacklevel=2)

        #initialize and fill in data array
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        return self._dia_container((data, diags), shape=self.shape)

    todia.__doc__ = _spbase.todia.__doc__

    def todok(self, copy=False):
        if self.ndim > 2:
            raise ValueError(f'Cannot convert. DOK must be 1D or 2D. Got {self.ndim}D')
        self.sum_duplicates()
        dok = self._dok_container(self.shape, dtype=self.dtype)
        # ensure that 1d coordinates are not tuples
        if self.ndim == 1:
            coords = self.coords[0]
        else:
            coords = zip(*self.coords)

        dok._dict = dict(zip(coords, self.data))
        return dok

    todok.__doc__ = _spbase.todok.__doc__

    def diagonal(self, k=0):
        if self.ndim != 2:
            raise ValueError("diagonal requires two dimensions")
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            inds = tuple(idx[diag_mask] for idx in self.coords)
            (row, _), data = self._sum_duplicates(inds, self.data[diag_mask])
        diag[row + min(k, 0)] = data

        return diag

    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        if self.ndim != 2:
            raise ValueError("setting a diagonal requires two dimensions")
        M, N = self.shape
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # Update the internal structure.
        self.coords = (np.concatenate((self.row[keep], new_row)),
                       np.concatenate((self.col[keep], new_col)))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data. By default the index arrays are copied.
        """
        if copy:
            coords = tuple(idx.copy() for idx in self.coords)
        else:
            coords = self.coords
        return self.__class__((data, coords), shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self) -> None:
        """Eliminate duplicate entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.coords, self.data)
        self.coords, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, coords, data):
        # Assumes coords not in canonical format.
        if len(data) == 0:
            return coords, data
        # Sort coords w.r.t. rows, then cols. This corresponds to C-order,
        # which we rely on for argmin/argmax to return the first index in the
        # same way that numpy does (in the case of ties).
        order = np.lexsort(coords[::-1])
        coords = tuple(idx[order] for idx in coords)
        data = data[order]
        unique_mask = np.logical_or.reduce([
            idx[1:] != idx[:-1] for idx in coords
        ])
        unique_mask = np.append(True, unique_mask)
        coords = tuple(idx[unique_mask] for idx in coords)
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return coords, data

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.coords = tuple(idx[mask] for idx in self.coords)

    #######################
    # Arithmetic handlers #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        if self.ndim == 1:
            coo_todense_nd(np.array([1]), self.nnz, self.ndim,
                           self.coords[0], self.data, result.ravel('A'), fortran)
        elif self.ndim == 2:
            M, N = self._shape_as_2d
            coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                        result.ravel('A'), fortran)
        else:
            if fortran:
                strides = np.append(1, np.cumprod(self.shape[:-1]))
            else:
                strides = np.append(np.cumprod(self.shape[1:][::-1])[::-1], 1)
            coords = np.concatenate(self.coords)
            coo_todense_nd(strides, self.nnz, self.ndim,
                           coords, self.data, result.ravel('A'), fortran)
        return self._container(result, copy=False)


    def _add_sparse(self, other):
        if self.ndim < 3:
            return self.tocsr()._add_sparse(other)

        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        other = self.__class__(other)
        new_data = np.concatenate((self.data, other.data))
        new_coords = tuple(np.concatenate((self.coords, other.coords), axis=1))
        A = self.__class__((new_data, new_coords), shape=self.shape)
        return A


    def _sub_sparse(self, other):
        if self.ndim < 3:
            return self.tocsr()._sub_sparse(other)

        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        other = self.__class__(other)
        new_data = np.concatenate((self.data, -other.data))
        new_coords = tuple(np.concatenate((self.coords, other.coords), axis=1))
        A = coo_array((new_data, new_coords), shape=self.shape)
        return A


    def _matmul_vector(self, other):
        if self.ndim > 2:
            result = np.zeros(math.prod(self.shape[:-1]),
                              dtype=upcast_char(self.dtype.char, other.dtype.char))
            shape = np.array(self.shape)
            strides = np.append(np.cumprod(shape[:-1][::-1])[::-1][1:], 1)
            coords = np.concatenate(self.coords)
            coo_matvec_nd(self.nnz, len(self.shape), strides, coords, self.data,
                          other, result)

            result = result.reshape(self.shape[:-1])
            return result

        # self.ndim <= 2
        result_shape = self.shape[0] if self.ndim > 1 else 1
        result = np.zeros(result_shape,
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        if self.ndim == 2:
            col = self.col
            row = self.row
        elif self.ndim == 1:
            col = self.coords[0]
            row = np.zeros_like(col)
        else:
            raise NotImplementedError(
                f"coo_matvec not implemented for ndim={self.ndim}")

        coo_matvec(self.nnz, row, col, self.data, other, result)
        # Array semantics return a scalar here, not a single-element array.
        if isinstance(self, sparray) and result_shape == 1:
            return result[0]
        return result


    def _matmul_dispatch(self, other):
        if self.ndim < 3:
            return _spbase._matmul_dispatch(self, other)

        N = self.shape[-1]
        err_prefix = "matmul: dimension mismatch with signature"
        if other.__class__ is np.ndarray:
            if other.shape == (N,):
                return self._matmul_vector(other)
            if other.shape == (N, 1):
                result = self._matmul_vector(other.ravel())
                return result.reshape(*self.shape[:-1], 1)
            if other.ndim == 1:
                msg = f"{err_prefix} (n,k={N}),(k={other.shape[0]},)->(n,)"
                raise ValueError(msg)
            msg = "n-D matrix-matrix multiplication not implemented for ndim>2"
            raise NotImplementedError(msg)

        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        if issparse(other):
            raise NotImplementedError("sparse-sparse matmul not implemented for ndim>2")

        # If it's a list or whatever, treat it like an array
        other_a = np.asanyarray(other)

        if other_a.ndim == 0 and other_a.dtype == np.object_:
            # Not interpretable as an array; return NotImplemented so that
            # other's __rmatmul__ can kick in if that's implemented.
            return NotImplemented

        try:
            other.shape
        except AttributeError:
            other = other_a

        if other.ndim == 1 or other.ndim == 2 and other.shape[1] == 1:
            # dense row or column vector
            if other.shape[0] != N:
                raise ValueError(
                    f"{err_prefix} (n,k={N}),(k={other.shape[0]},1?)->(n,1?)"
                )

            result = self._matmul_vector(np.ravel(other))

            if isinstance(other, np.matrix):
                result = self._ascontainer(result)

            if other.ndim == 2 and other.shape[1] == 1:
                # If 'other' was an (nx1) column vector, reshape the result
                result = result.reshape((*self.shape[:-1], 1))

            return result

        else:
            # dense nD array or matrix ("multivector") not implemented
            return NotImplemented


    def _matmul_multivector(self, other):
        result_dtype = upcast_char(self.dtype.char, other.dtype.char)
        if self.ndim == 2:
            result_shape = (self.shape[0], other.shape[1])
            col = self.col
            row = self.row
        elif self.ndim == 1:
            result_shape = (other.shape[1],)
            col = self.coords[0]
            row = np.zeros_like(col)
        else:
            raise NotImplementedError(
                f"coo_matmat_dense not implemented for ndim={self.ndim}")

        result = np.zeros(result_shape, dtype=result_dtype)
        coo_matmat_dense(self.nnz, other.shape[-1], row, col,
                         self.data, other.ravel('C'), result)
        return result.view(type=type(other))


############# dot
    def dot(a, b):
        if a.ndim < 3 and (np.isscalar(b) or b.ndim<3):
            return _data_matrix.dot(a, b)
        if isdense(b):
            return a._dense_dot(b)
        else:
            if not (isinstance(a, (coo_array, coo_matrix)) or np.isscalar(a)):
                raise TypeError("Input a must be a COO sparse array or a scalar")
            if not (isinstance(b, (coo_array, coo_matrix)) or np.isscalar(b)):
                raise TypeError("Input b must be a COO sparse array or a scalar")

            # Handle scalar multiplication
            if np.isscalar(a) or np.isscalar(b):
                return a * b

            # Handle inner product of vectors (1-D arrays)
            if a.ndim == 1 and b.ndim == 1:
                if a.shape[0] != b.shape[0]:
                    raise ValueError(f"shapes {a.shape} and {b.shape} are not aligned for inner product")
                return a @ b
            
            # Handle matrix multiplication (2-D arrays)
            if a.ndim == 2 and b.ndim == 2:
                if a.shape[1] != b.shape[0]:
                    raise ValueError(f"shapes {a.shape} and {b.shape} are not aligned for matrix multiplication")
                return a @ b
            
            return a._sparse_dot(b)

    
    def _sparse_dot(a, b):
        a_is_1d = False
        b_is_1d = False

        # reshape to 2-D if a or b is 1-D
        if a.ndim == 1:
            a = a.reshape((1, a.shape[0])) # prepend 1 to shape
            a_is_1d = True

        if b.ndim == 1:
            b = b.reshape((b.shape[0], 1)) # append 1 to shape
            b_is_1d = True

        if a.shape[-1] != b.shape[-2]:
                raise ValueError(f"shapes {a.shape} and {b.shape} are not aligned for n-D dot")
        
        a_2d, b_2d, og_shape_a, og_shape_b = _prepare_sparse_tensors_for_tensordot(a, b, axes_a=[a.ndim-1], axes_b=[b.ndim-2], tensordot=False)
        
        prod = (a_2d @ b_2d).tocoo() # routes via 2-D CSR

        # Combine the shapes of the non-reduced axes
        combined_shape = og_shape_a + og_shape_b

        # Unravel the 2D coordinates to get multi-dimensional coordinates
        unraveled_coords1 = np.unravel_index(np.array(prod.coords[0]), og_shape_a)
        unraveled_coords2 = np.unravel_index(np.array(prod.coords[1]), og_shape_b)

        nd_coords1 = np.array(unraveled_coords1)
        nd_coords2 = np.array(unraveled_coords2)

        prod_arr = coo_array((prod.data, np.concatenate((nd_coords1, nd_coords2))), combined_shape)
        
        # reshape back if a or b were originally 1-D
        if a_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[1:])
        if b_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[:-1])

        return prod_arr


    def tensordot(a, b, axes=2):
        # Check if axes is an integer
        if isinstance(axes, int):
            # Ensure the integer is valid
            if axes < 1 or axes > min(a.ndim, b.ndim):
                raise ValueError("axes integer is out of bounds for input arrays")
            
        # Check if axes is a list/tuple of length 2
        elif isinstance(axes, (tuple, list)):
            if len(axes) != 2:
                raise ValueError("axes must be a tuple/list of length 2")
            
            a_axes, b_axes = axes
            if len(a_axes) != len(b_axes):
                raise ValueError("axes lists/tuples must be of the same length")
            
            if any(ax >= a.ndim or ax < -a.ndim for ax in a_axes) or \
               any(ax >= b.ndim or ax < -b.ndim for ax in b_axes):
                raise ValueError("axes indices are out of bounds for input arrays")
            
            if any(a.shape[ax] != b.shape[bx] for ax, bx in zip(a_axes, b_axes)):
                raise ValueError("sizes of the corresponding axes must match")

        else:
            raise TypeError("axes must be an integer or a tuple/list of integers")
        
        # Determine the axes for contraction
        axes_a, axes_b = _determine_default_axes(len(a.shape), len(b.shape), axes)
        
        # Convert tuple to list if necessary
        if isinstance(axes_a, tuple):
            axes_a = list(axes_a)
        if isinstance(axes_b, tuple):
            axes_b = list(axes_b)

        # Adjust negative axes for a
        axes_a = [axis + a.ndim if axis < 0 else axis for axis in axes_a]
        # Adjust negative axes for b
        axes_b = [axis + b.ndim if axis < 0 else axis for axis in axes_b]
        
        if isdense(b):
            return a._dense_tensordot(b)
        else:
            return a._sparse_tensordot(b, axes_a, axes_b)


    def _sparse_tensordot(a, b, axes_a, axes_b):
        # Prepare the tensors for tensordot operation
        a_2d, b_2d, og_shape_a, og_shape_b = _prepare_sparse_tensors_for_tensordot(a, b, axes_a, axes_b, tensordot=True)

        # Perform matrix multiplication (routed via 2-D CSR)
        prod = (a_2d @ b_2d).tocoo()

        # Combine the shapes of the non-contracted axes
        combined_shape = og_shape_a + og_shape_b
        
        # Unravel the 2D coordinates to get multi-dimensional coordinates
        if og_shape_a:
            unraveled_coords1 = np.unravel_index(prod.row, og_shape_a)
            nd_coords1 = np.array(unraveled_coords1)

        if og_shape_b:
            unraveled_coords2 = np.unravel_index(prod.col, og_shape_b)
            nd_coords2 = np.array(unraveled_coords2)

        if og_shape_a and og_shape_b:
            coords = np.concatenate((nd_coords1, nd_coords2))
        elif og_shape_a:
            coords = nd_coords1
        elif og_shape_b:
            coords = nd_coords2
        else:
            coords = np.array([])
        
        if coords.shape == (0,): # if result is scalar
            return prod.data[0]
        
        # Construct the resulting COO array with combined coordinates and shape
        prod_arr = coo_array((prod.data, tuple(coords)), shape=combined_shape)
        
        return prod_arr


def _determine_default_axes(ndim_a, ndim_b, axes):
    """
    Determine the default axes for tensordot-like operation.

    Parameters:
    -----------
    ndim_a : int
        Number of dimensions of the first array.
    ndim_b : int
        Number of dimensions of the second array.
    axes : int
        Number of axes to match for the dot product.

    Returns:
    --------
    axes_a : list of int
        The axes of the first array to be reduced.
    axes_b : list of int
        The axes of the second array to be reduced.
    """
    if isinstance(axes, int):
        axes_a = list(range(ndim_a - axes, ndim_a))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
    
    return axes_a, axes_b


def _ravel_non_reduced_axes(coords, shape, axes):

    ndim = len(shape)
    non_reduced_axes = [ax for ax in range(ndim) if ax not in axes]

    if not non_reduced_axes:
        return np.zeros((1, len(coords[0])), dtype=int)  # Return an array with one row
    
    # Extract the shape of the non-reduced axes
    non_reduced_shape = [shape[ax] for ax in non_reduced_axes]
    
    # Extract the coordinates of the non-reduced axes
    coords = np.array(coords)
    non_reduced_coords = coords[non_reduced_axes, :]
    
    # Ravel the coordinates into 1D
    raveled_coords = np.ravel_multi_index(non_reduced_coords, non_reduced_shape)
    
    return raveled_coords


def _prepare_sparse_tensors_for_tensordot(a, b, axes_a, axes_b, tensordot):
    ndim_a = len(a.shape)
    ndim_b = len(b.shape)
    
    # Ravel non-reduced axes coordinates
    a_raveled_coords = _ravel_non_reduced_axes(a.coords, a.shape, axes_a)
    b_raveled_coords = _ravel_non_reduced_axes(b.coords, b.shape, axes_b)

    # Get the shape of the non-reduced axes
    original_shape_a = tuple(a.shape[ax] for ax in range(ndim_a) if ax not in axes_a)
    original_shape_b = tuple(b.shape[ax] for ax in range(ndim_b) if ax not in axes_b)
    
# Create 2D coords arrays
    if original_shape_a:
        ravel_coords_shape_a = (math.prod(original_shape_a), math.prod([a.shape[ax] for ax in axes_a]))
    else:
        ravel_coords_shape_a = (1, math.prod([a.shape[ax] for ax in axes_a]))  # Single row for no shape case

    if original_shape_b:
        ravel_coords_shape_b = (math.prod([b.shape[ax] for ax in axes_b]), math.prod(original_shape_b))
    else:
        ravel_coords_shape_b = (math.prod([b.shape[ax] for ax in axes_b]), 1)  # Single column for no shape case
    
    if tensordot:
        a_2d_coords = np.vstack((a_raveled_coords, np.ravel_multi_index(np.array(a.coords)[axes_a, :], [a.shape[ax] for ax in axes_a])))
        b_2d_coords = np.vstack((np.ravel_multi_index(np.array(b.coords)[axes_b, :], tuple([b.shape[ax] for ax in axes_b])), b_raveled_coords))
    
    else: # if dot
        a_2d_coords = np.vstack((a_raveled_coords, a.coords[axes_a[0]]))
        b_2d_coords = np.vstack((b.coords[axes_b[0]], b_raveled_coords))

    return coo_array((a.data, a_2d_coords), ravel_coords_shape_a), coo_array((b.data, b_2d_coords),
                                                ravel_coords_shape_b), original_shape_a, original_shape_b


def _ravel_coords(coords, shape, order='C'):
    """Like np.ravel_multi_index, but avoids some overflow issues."""
    if len(coords) == 1:
        return coords[0]
    # Handle overflow as in https://github.com/scipy/scipy/pull/9132
    if len(coords) == 2:
        nrows, ncols = shape
        row, col = coords
        if order == 'C':
            maxval = (ncols * max(0, nrows - 1) + max(0, ncols - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(ncols, row, dtype=idx_dtype) + col
        elif order == 'F':
            maxval = (nrows * max(0, ncols - 1) + max(0, nrows - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(nrows, col, dtype=idx_dtype) + row
        else:
            raise ValueError("'order' must be 'C' or 'F'")
    return np.ravel_multi_index(coords, shape, order=order)


def isspmatrix_coo(x):
    """Is `x` of coo_matrix type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if `x` is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_array, coo_matrix, csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(coo_matrix([[5]]))
    True
    >>> isspmatrix_coo(coo_array([[5]]))
    False
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_matrix)


# This namespace class separates array from matrix with isinstance
class coo_array(_coo_base, sparray):
    """
    A sparse array in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_array(D)
            where D is an ndarray

        coo_array(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_array(shape, [dtype])
            to construct an empty sparse array with shape `shape`
            dtype is optional, defaulting to dtype='d'.

        coo_array((data, coords), [shape])
            to construct from existing data and index arrays:
                1. data[:]       the entries of the sparse array, in any order
                2. coords[i][:]  the axis-i coordinates of the data entries

            Where ``A[coords] = data``, and coords is a tuple of index arrays.
            When shape is not specified, it is inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the sparse array
    shape : tuple of integers
        Shape of the sparse array
    ndim : int
        Number of dimensions of the sparse array
    nnz
    size
    data
        COO format data array of the sparse array
    coords
        COO format tuple of index arrays
    has_canonical_format : bool
        Whether the matrix has sorted coordinates and no duplicates
    format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse arrays
        - Once a COO array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty sparse array
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> coo_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a sparse array using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_array((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a sparse array with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_array((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """


class coo_matrix(spmatrix, _coo_base):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            where D is a 2-D ndarray

        coo_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix
    has_canonical_format : bool
        Whether the matrix has sorted indices and no duplicates
    format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a COO matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """

    def __setstate__(self, state):
        if 'coords' not in state:
            # For retro-compatibility with the previous attributes
            # storing nnz coordinates for 2D COO matrix.
            state['coords'] = (state.pop('row'), state.pop('col'))
        self.__dict__.update(state)
