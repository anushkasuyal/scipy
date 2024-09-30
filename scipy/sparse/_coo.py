""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_array', 'coo_matrix', 'isspmatrix_coo']

import math
from warnings import warn

import numpy as np

from .._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._sparsetools import (coo_tocsr, coo_todense, coo_todense_nd,
                           coo_matvec, coo_matvec_nd, coo_matmat_dense,
                           coo_matmat_dense_nd)
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast_char, to_native, isshape, getdtype,
                       getdata, downcast_intp_index, get_index_dtype,
                       check_shape, check_reshape_kwargs, isscalarlike, isdense,
                       is_pydata_spmatrix)

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
            
        else: # dim>2
            flat_indices = np.ravel_multi_index(self.coords, self.shape)
            B = np.zeros(self.shape, dtype=self.dtype)
            np.add.at(B.ravel(), flat_indices, self.data)
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
        data = np.add.reduceat(data, downcast_intp_index(unique_inds), dtype=self.dtype)
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
        if self.shape != other.shape:
            bshape = np.broadcast_shapes(self.shape, other.shape)
            self = self._broadcast_to(bshape)
            other = np.broadcast_to(other, bshape)
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        if self.ndim < 3 and len(other.shape) < 3:
            M, N = self._shape_as_2d
            coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                        result.ravel('A'), fortran)
        else:
            result_shape = self.shape
            self = self.reshape(1, -1)
            other = other.reshape(1, -1)
            result = _spbase._add_dense(self, other)
            # reshape back to n-D
            result = result.reshape(result_shape)
        return self._container(result, copy=False)


    def _add_sparse(self, other):
        if self.ndim < 3 and other.ndim < 3:
            return self.tocsr()._add_sparse(other)
        
        other = self.__class__(other)
        if other.shape != self.shape:
            # This will raise an error if the shapes are not broadcastable
            bshape = np.broadcast_shapes(self.shape, other.shape)
            self = self._broadcast_to(bshape)
            other = other._broadcast_to(bshape)

        new_data = np.concatenate((self.data, other.data))
        new_coords = tuple(np.concatenate((self.coords, other.coords), axis=1))
        A = self.__class__((new_data, new_coords), shape=self.shape)
        return A

    def _sub_dense(self, other):
        if self.shape != other.shape:
            bshape = np.broadcast_shapes(self.shape, other.shape)
            self = self._broadcast_to(bshape)
            other = np.broadcast_to(other, bshape)

        return self.todense() - other

    def _sub_sparse(self, other):
        if self.ndim < 3 and other.ndim < 3:
            return self.tocsr()._sub_sparse(other)

        other = self.__class__(other)
        if other.shape != self.shape:
            # This will raise an error if the shapes are not broadcastable
            bshape = np.broadcast_shapes(self.shape, other.shape)
            self = self._broadcast_to(bshape)
            other = other._broadcast_to(bshape)

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


    def _rmatmul_dispatch(self, other):
        if isscalarlike(other):
            return self._mul_scalar(other)
        else:
            # Don't use asarray unless we have to
            try:
                o_ndim = other.ndim
                perm = tuple(range(o_ndim)[:-2]) + tuple(range(o_ndim)[-2:][::-1])
                tr = other.transpose(perm)
            except AttributeError:
                o_arr = np.asarray(other)
                o_ndim = o_arr.ndim
                perm = tuple(range(o_ndim)[:-2]) + tuple(range(o_ndim)[-2:][::-1])
                tr = o_arr.transpose(perm)
            
            s_ndim = self.ndim
            perm = tuple(range(s_ndim)[:-2]) + tuple(range(s_ndim)[-2:][::-1])
            ret = self.transpose(perm)._matmul_dispatch(tr)
            if ret is NotImplemented:
                return NotImplemented
            
            if s_ndim == 1 or o_ndim == 1:
                perm = range(ret.ndim)
            else:
                perm = tuple(range(ret.ndim)[:-2]) + tuple(range(ret.ndim)[-2:][::-1])
            return ret.transpose(perm)
        

    def _matmul_dispatch(self, other):
        if isscalarlike(other):
            return self.multiply(other)
        
        if not (issparse(other) or isdense(other)):
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

        if self.ndim < 3 and other.ndim < 3:
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
            if other.shape[-2] == N:
                # check for batch dimensions compatibility
                batch_shape_A = self.shape[:-2]
                batch_shape_B = other.shape[:-2]
                if batch_shape_A != batch_shape_B:
                    try:
                        # This will raise an error if the shapes are not broadcastable
                        np.broadcast_shapes(batch_shape_A, batch_shape_B)
                    except ValueError:
                        raise ValueError("Batch dimensions are not broadcastable")
            
                return self._matmul_multivector(other)
            else:
                raise ValueError(
                    f"{err_prefix} (n,..,k={N}),(k={other.shape[-2]},..,m)->(n,..,m)"
                )
        
            
        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        if issparse(other):
            self_is_1d = self.ndim == 1
            other_is_1d = other.ndim == 1

            # reshape to 2-D if self or other is 1-D
            if self_is_1d:
                self = self.reshape(self._shape_as_2d) # prepend 1 to shape

            if other_is_1d:
                other = other.reshape((other.shape[0], 1)) # append 1 to shape

            # Check if the inner dimensions match for matrix multiplication
            if N != other.shape[-2]:
                raise ValueError(
                    f"{err_prefix} (n,..,k={N}),(k={other.shape[-2]},..,m)->(n,..,m)"
                )
            
            # If A or B has more than 2 dimensions, check for
            # batch dimensions compatibility
            if self.ndim > 2 or other.ndim > 2:
                batch_shape_A = self.shape[:-2]
                batch_shape_B = other.shape[:-2]
                if batch_shape_A != batch_shape_B:
                    try:
                        # This will raise an error if the shapes are not broadcastable
                        np.broadcast_shapes(batch_shape_A, batch_shape_B)
                    except ValueError:
                        raise ValueError("Batch dimensions are not broadcastable")
            
            result = self._matmul_sparse(other)

            # reshape back if a or b were originally 1-D
            if self_is_1d:
                # if self was originally 1-D, reshape result accordingly
                result = result.reshape(tuple(result.shape[:-2]) +
                                        tuple(result.shape[-1:]))
            if other_is_1d:
                result = result.reshape(result.shape[:-1])
            return result


    def _matmul_multivector(self, other):
        result_dtype = upcast_char(self.dtype.char, other.dtype.char)
        if self.ndim >= 3 or other.ndim >= 3:
            # if self has shape (N,), reshape to (1,N)
            self_is_1d = False
            if self.ndim == 1:
                self_is_1d = True
                self = self.reshape(1, self.shape[0])
            broadcast_shape = np.broadcast_shapes(self.shape[:-2], other.shape[:-2])
            self_shape = broadcast_shape + self.shape[-2:]
            other_shape = broadcast_shape + other.shape[-2:]
            
            self = self._broadcast_to(self_shape)
            other = np.broadcast_to(other, other_shape)
            result_shape = broadcast_shape + self.shape[-2:-1] + other.shape[-1:]
            result = np.zeros(result_shape, dtype=result_dtype)
            coo_matmat_dense_nd(self.nnz, len(self.shape), other.shape[-1],
                                np.array(other_shape), np.array(result_shape),
                                np.concatenate(self.coords),
                                self.data, other.ravel('C'), result)
            
            # if self was originally 1-D, reshape result accordingly
            if self_is_1d:
                result = result.reshape(tuple(result_shape[:-2]) +
                                        tuple(result_shape[-1:]))
            return result
        
        if self.ndim == 2:
            result_shape = (self.shape[0], other.shape[1])
            col = self.col
            row = self.row
        elif self.ndim == 1:
            result_shape = (other.shape[1],)
            col = self.coords[0]
            row = np.zeros_like(col)
        result = np.zeros(result_shape, dtype=result_dtype)
        coo_matmat_dense(self.nnz, other.shape[-1], row, col,
                         self.data, other.ravel('C'), result)
        return result.view(type=type(other))


    def dot(self, other):
        if not (issparse(other) or isdense(other) or isscalarlike(other)):
            # If it's a list or whatever, treat it like an array
            o_array = np.asanyarray(other)

            if o_array.ndim == 0 and o_array.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = o_array

        if self.ndim < 3 and (np.isscalar(other) or other.ndim<3):
            return _spbase.dot(self, other)
        if isdense(other):
            return self._dense_dot(other)
        else:
            # Handle scalar multiplication
            if np.isscalar(other):
                return self * other
            
            if other.format != "coo":
                raise TypeError("input must be a COO matrix/array")

            # Handle inner product of vectors (1-D arrays)
            if self.ndim == 1 and other.ndim == 1:
                if self.shape[0] != other.shape[0]:
                    raise ValueError(f"shapes {self.shape} and {other.shape}"
                                     " are not aligned for inner product")
                return self @ other
            
            # Handle matrix multiplication (2-D arrays)
            if self.ndim == 2 and other.ndim == 2:
                if self.shape[1] != other.shape[0]:
                    raise ValueError(f"shapes {self.shape} and {other.shape}"
                                     " are not aligned for matmul")
                return self @ other
            
            return self._sparse_dot(other)

    
    def _sparse_dot(self, other):
        self_is_1d = self.ndim == 1
        other_is_1d = other.ndim == 1

        # reshape to 2-D if self or other is 1-D
        if self_is_1d:
            self = self.reshape(self._shape_as_2d)  # prepend 1 to shape
        if other_is_1d:
            other = other.reshape((other.shape[0], 1))  # append 1 to shape

        if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for n-D dot")
        
        # Prepare the tensors for dot operation
        # Ravel non-reduced axes coordinates
        self_raveled_coords = _ravel_non_reduced_axes(self.coords,
                                                      self.shape, [self.ndim-1])
        other_raveled_coords = _ravel_non_reduced_axes(other.coords,
                                                       other.shape, [other.ndim-2])

        # Get the shape of the non-reduced axes
        self_nonreduced_shape = self.shape[:-1]
        other_nonreduced_shape = other.shape[:-2] + other.shape[-1:]
        
        # Create 2D coords arrays
        ravel_coords_shape_self = (math.prod(self_nonreduced_shape), self.shape[-1])
        ravel_coords_shape_other = (other.shape[-2], math.prod(other_nonreduced_shape))
        
        self_2d_coords = (self_raveled_coords, self.coords[-1])
        other_2d_coords = (other.coords[-2], other_raveled_coords)

        self_2d = coo_array((self.data, self_2d_coords), ravel_coords_shape_self)
        other_2d = coo_array((other.data, other_2d_coords), ravel_coords_shape_other)
        
        prod = (self_2d @ other_2d).tocoo() # routes via 2-D CSR

        # Combine the shapes of the non-reduced axes
        combined_shape = self_nonreduced_shape + other_nonreduced_shape

        # Unravel the 2D coordinates to get multi-dimensional coordinates
        shapes = (self_nonreduced_shape, other_nonreduced_shape)
        iter_cs = zip(prod.coords, shapes)
        prod_coords = sum((np.unravel_index(c, s) for c, s in iter_cs), start=())

        prod_arr = coo_array((prod.data, prod_coords), combined_shape)
        
        # reshape back if a or b were originally 1-D
        if self_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[1:])
        if other_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[:-1])

        return prod_arr
    
    def _dense_dot(self, other):
        self_is_1d = self.ndim == 1
        other_is_1d = other.ndim == 1

        # reshape to 2-D if self or other is 1-D
        if self_is_1d:
            self = self.reshape(self._shape_as_2d)  # prepend 1 to shape
        if other_is_1d:
            other = other.reshape((other.shape[0], 1))  # append 1 to shape

        if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for n-D dot")

        new_shape_self = self.shape[:-1] + (1,) * (len(other.shape) - 1) \
            + self.shape[-1:]
        new_shape_other = (1,) * (len(self.shape) - 1) + other.shape

        result_shape = self.shape[:-1] + other.shape[:-2] + other.shape[-1:]
        result = self.reshape(new_shape_self) @ other.reshape(new_shape_other)
        prod_arr = result.reshape(result_shape)

        # reshape back if a or b were originally 1-D
        if self_is_1d:
            prod_arr = prod_arr.reshape(result_shape[1:])
        if other_is_1d:
            prod_arr = prod_arr.reshape(result_shape[:-1])

        return prod_arr

    def tensordot(self, other, axes=2):
        if not isdense(other) and not issparse(other):
            # If it's a list or whatever, treat it like an array
            other_array = np.asanyarray(other)

            if other_array.ndim == 0 and other_array.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_array

        axes_self, axes_other = _process_axes(self.ndim, other.ndim, axes)

        # Adjust negative axes for self
        axes_self = [axis + self.ndim if axis < 0 else axis for axis in axes_self]
        # Adjust negative axes for other
        axes_other = [axis + other.ndim if axis < 0 else axis for axis in axes_other]

        # Check for shape compatibility along specified axes
        if any(self.shape[ax] != other.shape[bx]
               for ax, bx in zip(axes_self, axes_other)):
            raise ValueError("sizes of the corresponding axes must match")
        
        if isdense(other):
            return self._dense_tensordot(other, axes_self, axes_other)
        else:
            return self._sparse_tensordot(other, axes_self, axes_other)


    def _sparse_tensordot(self, other, axes_self, axes_other):
        ndim_self = len(self.shape)
        ndim_other = len(other.shape)

        # Prepare the tensors for tensordot operation       
        # Ravel non-reduced axes coordinates
        self_non_red_coords = _ravel_non_reduced_axes(self.coords, self.shape,
                                                      axes_self)
        self_reduced_coords = np.ravel_multi_index(
            [self.coords[ax] for ax in axes_self], [self.shape[ax] for ax in axes_self])
        other_non_red_coords = _ravel_non_reduced_axes(other.coords, other.shape,
                                                       axes_other)
        other_reduced_coords = np.ravel_multi_index(
            [other.coords[a] for a in axes_other], [other.shape[a] for a in axes_other]
        )
        # Get the shape of the non-reduced axes
        self_nonreduced_shape = tuple(self.shape[ax] for ax in range(ndim_self)
                              if ax not in axes_self)
        other_nonreduced_shape = tuple(other.shape[ax] for ax in range(ndim_other)
                               if ax not in axes_other)
        
        # Create 2D coords arrays
        ravel_coords_shape_self = (math.prod(self_nonreduced_shape),
                                math.prod([self.shape[ax] for ax in axes_self]))
        ravel_coords_shape_other = (math.prod([other.shape[ax] for ax in axes_other]),
                                    math.prod(other_nonreduced_shape))

        self_2d_coords = (self_non_red_coords, self_reduced_coords)
        other_2d_coords = (other_reduced_coords, other_non_red_coords)

        self_2d = coo_array((self.data, self_2d_coords), ravel_coords_shape_self)
        other_2d = coo_array((other.data, other_2d_coords), ravel_coords_shape_other)

        # Perform matrix multiplication (routed via 2-D CSR)
        prod = (self_2d @ other_2d).tocoo()

        # Combine the shapes of the non-contracted axes
        combined_shape = self_nonreduced_shape + other_nonreduced_shape

        # Unravel the 2D coordinates to get multi-dimensional coordinates
        iter_cs = zip(prod.coords, (self_nonreduced_shape, other_nonreduced_shape))
        coords = sum((np.unravel_index(c, s) for c, s in iter_cs if s), start=())
 

        if coords == ():  # if result is scalar
            return sum(prod.data)
            
        # Construct the resulting COO array with combined coordinates and shape
        return coo_array((prod.data, coords), shape=combined_shape)


    def _dense_tensordot(self, other, axes_self, axes_other):
        ndim_self = len(self.shape)
        ndim_other = len(other.shape)

        non_reduced_axes_self = [ax for ax in range(ndim_self) if ax not in axes_self]
        reduced_shape_self = [self.shape[s] for s in axes_self]
        non_reduced_shape_self = [self.shape[s] for s in non_reduced_axes_self]

        non_reduced_axes_other = [ax for ax in range(ndim_other)
                                  if ax not in axes_other]
        reduced_shape_other = [other.shape[s] for s in axes_other]
        non_reduced_shape_other = [other.shape[s] for s in non_reduced_axes_other]

        permute_self = non_reduced_axes_self + axes_self
        permute_other = non_reduced_axes_other[:-1] + axes_other \
            + non_reduced_axes_other[-1:]
        self = self.transpose(permute_self)
        other = np.transpose(other, permute_other)

        reshape_self = (*non_reduced_shape_self, math.prod(reduced_shape_self))
        reshape_other = (*non_reduced_shape_other[:-1], math.prod(reduced_shape_other),
                        *non_reduced_shape_other[-1:])

        prod_arr = self.reshape(reshape_self).dot(other.reshape(reshape_other))
        return prod_arr


    def _matmul_sparse(self, other):
        """
        Perform sparse-sparse matrix multiplication for two n-D COO arrays.
        The method converts input n-D arrays to 2-D block array format,
        uses csr_matmat to multiply them, and then converts the
        result back to n-D COO array.
        
        Parameters:
        self (COO): The first n-D sparse array in COO format.
        other (COO): The second n-D sparse array in COO format.
        
        Returns:
        prod (COO): The resulting n-D sparse array after multiplication.
        """
        if self.ndim < 3 and other.ndim < 3:
            return _spbase._matmul_sparse(self, other)

        # Get the shapes of self and other
        self_shape = self.shape
        other_shape = other.shape
        
        # Determine the new shape to broadcast self and other
        broadcast_shape = np.broadcast_shapes(self_shape[:-2], other_shape[:-2])
        self_new_shape = tuple(broadcast_shape) + self_shape[-2:]
        other_new_shape = tuple(broadcast_shape) + other_shape[-2:]

        self_broadcasted = self._broadcast_to(self_new_shape)
        other_broadcasted = other._broadcast_to(other_new_shape)
        
        # Convert n-D COO arrays to 2-D block diagonal arrays
        self_block_diag = _block_diag(self_broadcasted)
        other_block_diag = _block_diag(other_broadcasted)
        
        # Use csr_matmat to perform sparse matrix multiplication
        prod_block_diag = (self_block_diag @ other_block_diag).tocoo()
        
        # Convert the 2-D block diagonal array back to n-D
        prod = _extract_block_diag(prod_block_diag, shape=(*broadcast_shape,
                                                     self.shape[-2], other.shape[-1]))
        
        return prod


    def _broadcast_to(self, new_shape, copy=False):
        if self.shape == new_shape:
            return self.copy() if copy else self
        
        old_shape = self.shape

        # Check if the new shape is compatible for broadcasting
        if len(new_shape) < len(old_shape):
            raise ValueError("New shape must have at least as many dimensions"
                             " as the current shape")
        
        # Add leading ones to shape to ensure same length as `new_shape` 
        shape = (1,) * (len(new_shape) - len(old_shape)) + tuple(old_shape)
        
        # Ensure the old shape can be broadcast to the new shape
        if any((o != 1 and o != n) for o, n in zip(shape, new_shape)):
            raise ValueError(f"current shape {old_shape} cannot be "
                             "broadcast to new shape {new_shape}")

        # Reshape the COO array to match the new dimensions
        self = self.reshape(shape)

        coords = self.coords
        new_data = self.data
        new_coords = coords[-1:]  # Copy last coordinate to start
        cum_repeat = 1 # Cumulative repeat factor for broadcasting
        
        if shape[-1] != new_shape[-1]: # broadcasting the n-th (col) dimension
            repeat_count = new_shape[-1]
            cum_repeat *= repeat_count
            new_data = np.tile(new_data, repeat_count)
            new_dim = np.repeat(np.arange(0, repeat_count), self.nnz)
            new_coords = (new_dim,)
        
        for i in range(-2, -(len(shape)+1), -1):
            if shape[i] != new_shape[i]:
                repeat_count = new_shape[i] # number of times to repeat data, coords
                cum_repeat *= repeat_count # update cumulative repeat factor
                nnz = len(new_data) # Number of non-zero elements so far

                # Tile data and coordinates to match the new repeat count
                new_data = np.tile(new_data, repeat_count)
                new_coords = tuple(np.tile(new_coords[i+1:], repeat_count))

                # Create new dimensions and stack them
                new_dim = np.repeat(np.arange(0, repeat_count), nnz)
                new_coords = (new_dim,) + new_coords
            else:
                # If no broadcasting needed, tile the coordinates
                new_dim = np.tile(coords[i], cum_repeat)
                new_coords = (new_dim,) + new_coords
                
        return coo_array((new_data, new_coords), new_shape)


    def _process_arrays_for_comparison(self, other, op_name):
        if is_pydata_spmatrix(other):
            return NotImplemented
        
        if not (issparse(other) or isdense(other) or isscalarlike(other)):
            # If it's a list or whatever, treat it like an array
            other_a = np.asanyarray(other)

            if other_a.ndim == 0 and other_a.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_a

        if self.ndim < 3 and (isscalarlike(other) or other.ndim < 3):
            return getattr(_data_matrix, op_name)(self.tocsr(), other)
        
        # Scalar other.
        if isscalarlike(other):
            result_shape = self.shape
            self = self.reshape(1,-1)
            result = getattr(_data_matrix, op_name)(self.tocsr(), other)
            if isinstance(result, sparray):
                result = result.tocoo()
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(result_shape)
            return result

        elif isdense(other) or issparse(other):
            if self.shape != other.shape:
                # This will raise an error if the shapes are not broadcastable
                broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
                # Broadcasting the arrays if they have different shapes
                # that are compatible for broadcasting
                self = self._broadcast_to(broadcast_shape)
                if isdense(other):
                    other = np.broadcast_to(other, broadcast_shape)
                else:
                    other = other._broadcast_to(broadcast_shape)

            result_shape = self.shape

            # reshaping n-D arrays to 2-D arrays
            self = self.reshape(1,-1)
            other = other.reshape(1,-1)
            
            # routing via 2-D CSR
            result = getattr(_data_matrix, op_name)(self.tocsr(), other)

            if isinstance(result, sparray):
                result = result.tocoo()

            # reshaping back to n-D if output is 2-D boolean array
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(result_shape)

            return result
        else:
            return NotImplemented

    def __eq__(self, other):
        return self._process_arrays_for_comparison(other, '__eq__')
    
    def __ne__(self, other):
        return self._process_arrays_for_comparison(other, '__ne__')
                                       
    def __lt__(self, other):
        return self._process_arrays_for_comparison(other, '__lt__')

    def __gt__(self, other):
        return self._process_arrays_for_comparison(other, '__gt__')

    def __le__(self, other):
        return self._process_arrays_for_comparison(other, '__le__')

    def __ge__(self, other):
        return self._process_arrays_for_comparison(other, '__ge__')
    
    def multiply(self, other):
        """Point-wise multiplication by another array/matrix."""

        if isscalarlike(other):
            return self._mul_scalar(other)
        
        if not (issparse(other) or isdense(other)):
            # If it's a list or whatever, treat it like an array
            other_a = np.asanyarray(other)

            if other_a.ndim == 0 and other_a.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_a

        if self.ndim < 3 and other.ndim < 3:
            result = _data_matrix.multiply(self.tocsr(), other)
            if isinstance(result, sparray):
                result = result.tocoo()
            return result
            
        if issparse(other):
            if self.shape == other.shape: # no broadcasting required
                result_shape = self.shape

                # reshape n-D COO to 2-D
                self = self.reshape(1, -1)
                other = other.reshape(1, -1)

                # route via 2-D CSR multiply
                result = _data_matrix.multiply(self.tocsr(), other)

                # convert to COO format before reshaping to n-D
                # this is important because other formats don't support n-D
                if isinstance(result, sparray):
                    result = result.tocoo()

                # reshape back to n-D
                if isinstance(result, (np.ndarray, sparray)):
                    result = result.reshape(result_shape)

                return result
            
            bshape = np.broadcast_shapes(self.shape, other.shape)
            
            # single element
            if math.prod(other.shape) == 1:
                result = self._mul_scalar(other.toarray().ravel()[0])
                return result.reshape(bshape)
            if math.prod(self.shape) == 1:
                result = other._mul_scalar(self.toarray().ravel()[0])
                result = result.reshape(bshape)
                return result
            
            # different but broadcastable shapes
            self = self._broadcast_to(bshape)
            other = other._broadcast_to(bshape)
            # reshape to 2-D
            self = self.reshape(1, -1)
            other = other.reshape(1, -1)
            # route via 2-D CSR multiply
            result = _data_matrix.multiply(self.tocsr(), other)
            # convert to COO
            if isinstance(result, sparray):
                result = result.tocoo()

            # reshape back to n-D
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(bshape)

            return result
        
        else: # if other is dense
        # no broadcasting required if same shapes,
        # just reshape to 2-D, route via CSR, convert to COO,
        # and reshape back
            if self.shape == other.shape:
                result_shape = self.shape
                self = self.reshape(1, -1)
                other = other.reshape(1, -1)
                result = _data_matrix.multiply(self.tocsr(), other)
                # convert to COO
                if isinstance(result, sparray):
                    result = result.tocoo()
                # reshape back to n-D
                if isinstance(result, (np.ndarray, sparray)):
                    result = result.reshape(result_shape)
                return result
            
            # This will raise an error if the shapes are not broadcastable
            bshape = np.broadcast_shapes(self.shape, other.shape)

            # single element
            if math.prod(other.shape) == 1:
                result = self._mul_scalar(other.ravel()[0])
                result = result.reshape(bshape)
                return result
            
            # if self is a single element and other is dense,
            # use np.multiply and reshape
            if math.prod(self.shape) == 1:
                result = np.multiply(self.toarray().ravel()[0], other)
                result = result.reshape(bshape)
                return result
            
            # different but broadcastable shapes
            self = self._broadcast_to(bshape)
            other = np.broadcast_to(other, bshape)
            self = self.reshape(1, -1)
            other = other.reshape(1, -1)
            result = _data_matrix.multiply(self.tocsr(), other)
            # convert to COO
            if isinstance(result, sparray):
                result = result.tocoo()
            # reshape back to n-D
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(bshape)
            return result
    
    def _divide(self, other, true_divide=False, rdivide=False):
        """Point-wise division by another array/matrix."""
        # Scalar other.
        if isscalarlike(other):
            if rdivide:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())

            if true_divide and np.can_cast(self.dtype, np.float64):
                return self.astype(np.float64)._mul_scalar(1./other)
            
        if not (issparse(other) or isdense(other)):
            # If it's a list or whatever, treat it like an array
            other_a = np.asanyarray(other)

            if other_a.ndim == 0 and other_a.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_a

        if self.ndim < 3 and other.ndim < 3:
            return _data_matrix._divide(self.tocsr(), other, true_divide, rdivide)

        else:
            return NotImplemented

    def mean(self, axis=None, dtype=None, out=None):
        if axis == ():
            return self.toarray()
        
        if self.ndim < 3:
            result = _spbase.mean(self, axis, dtype, out) 
            return result

        axis = _validateaxes(axis, self.ndim, self.shape)
        
        if out is not None:
            out_shape = out.shape

        if axis is None:
            res_dtype = self.dtype.type
            integral = (np.issubdtype(self.dtype, np.integer) or
                        np.issubdtype(self.dtype, np.bool_))

            # output dtype
            if dtype is None:
                if integral:
                    res_dtype = np.float64
            else:
                res_dtype = np.dtype(dtype).type

            # intermediate dtype for summation
            inter_dtype = np.float64 if integral else res_dtype
            inter_self = self.astype(inter_dtype)

            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")
            ret = (inter_self / math.prod(self.shape)).sum(dtype=res_dtype, out=out)
            result_shape = ret.shape
        else:
            result_shape = tuple(self.shape[ax] for ax in range(self.ndim)
                                 if ax not in axis)

        if out is not None and out_shape != result_shape:
            raise ValueError("dimensions do not match")
        
        if axis is not None:
            non_axis_coords = _ravel_non_reduced_axes(self.coords, self.shape, axis)
            axis_coords = np.ravel_multi_index(np.array(self.coords)[axis, :],
                                               [self.shape[ax] for ax in axis])
            coords_2d = np.vstack((non_axis_coords, axis_coords))

            shape_2d = (math.prod(result_shape), math.prod([self.shape[ax]
                                                            for ax in axis]))

            if out is not None:
                out = out.reshape(math.prod(result_shape))

            self = coo_array((self.data, coords_2d), shape_2d)
            ret_flattened = _spbase.mean(self, axis=1, dtype=dtype, out=out)
            ret = ret_flattened.reshape(result_shape)      
            if out is not None:
                out = out.reshape(result_shape) 

        return ret


    def sum(self, axis=None, dtype=None, out=None):
        if axis == ():
            ret = self.todense()
            if out is not None:
                if out.shape != self.shape:
                    raise ValueError("dimensions do not match")
                out[...] = ret
            return ret
        
        if self.ndim < 3:
            result = _spbase.sum(self, axis, dtype, out)
            return result

        axis = _validateaxes(axis, self.ndim, self.shape)
        
        if out is not None:
            out_shape = out.shape
        
        if axis is None:
            self = self.reshape(-1)
            ret = _spbase.sum(self, axis, dtype, out)
            result_shape = ret.shape
        else:
            result_shape = tuple(self.shape[ax] for ax in range(self.ndim)
                                 if ax not in axis)

        if out is not None and out_shape != result_shape:
            raise ValueError("dimensions do not match")
        
        if axis is not None:
            non_axis_coords = _ravel_non_reduced_axes(self.coords, self.shape, axis)
            axis_coords = np.ravel_multi_index(np.array(self.coords)[axis, :],
                                               [self.shape[ax] for ax in axis])
            coords_2d = np.vstack((non_axis_coords, axis_coords))

            shape_2d = (math.prod(result_shape), math.prod([self.shape[ax]
                                                            for ax in axis]))
            
            if out is not None:
                out = out.reshape(math.prod(result_shape))

            self = coo_array((self.data, coords_2d), shape_2d)
            ret_flattened = _spbase.sum(self, axis=1, dtype=dtype, out=out)
            ret = ret_flattened.reshape(result_shape)

            if out is not None:
                out = out.reshape(result_shape)      
        
        return ret


    def _maximum_minimum_coo(self, other, op_name):
        if not (issparse(other) or isdense(other) or isscalarlike(other)):
            # If it's a list or whatever, treat it like an array
            other_a = np.asanyarray(other)

            if other_a.ndim == 0 and other_a.dtype == np.object_:
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_a

        if self.ndim < 3 and (isscalarlike(other) or other.ndim < 3):
            return  getattr(_data_matrix, op_name)(self.tocsr(), other)
        
         # Scalar other
        if isscalarlike(other):
            result_shape = self.shape
            self = self.reshape(1, -1)
            result =  getattr(_data_matrix, op_name)(self.tocsr(), other)
            # convert to COO
            if isinstance(result, sparray):
                result = result.tocoo()
            # reshape back to n-D
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(result_shape)
            return result

        elif isdense(other) or issparse(other):
            if self.shape != other.shape:
                # This will raise an error if the shapes are not broadcastable
                broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
                
                # Broadcasting the arrays if they have different shapes
                # that are compatible for broadcasting
                self = self._broadcast_to(broadcast_shape)
                if isdense(other):
                    other = np.broadcast_to(other, broadcast_shape)
                else:
                    other = other._broadcast_to(broadcast_shape)

            result_shape = self.shape

            # reshaping n-D arrays to 2-D arrays
            self = self.reshape(1,-1)
            other = other.reshape(1,-1)
            
            # routing via 2-D CSR
            result =  getattr(_data_matrix, op_name)(self.tocsr(), other)

            if isinstance(result, sparray):
                result = result.tocoo()

            # reshaping back to n-D if output is 2-D boolean array
            if isinstance(result, (np.ndarray, sparray)):
                result = result.reshape(result_shape)

            return result
        
        else:
            return NotImplemented

    def _find_max_or_min(self, axis, out, _max_or_min, _max_or_min_axis, explicit):
        zero = self.dtype.type(0)
        
        if axis is None:
            return _max_or_min((*self.data, zero))
        
        if not isinstance(axis, (int, tuple)):
            raise ValueError("'axis' should be int/tuple of ints")

        if axis == ():
            return self.copy()
        
        if type(axis) is int:
            axis = [axis]
        
        if any(ax >= self.ndim or ax < -self.ndim for ax in axis):
            raise ValueError("axis out of range")
        
        axis = [ax if ax>=0 else ax+self.ndim for ax in axis]

        if any(self.shape[d] == 0 for d in axis):
            raise ValueError("zero-size array to reduction operation")

        # Check for duplicates
        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        if len(axis) == self.ndim:
            return _max_or_min((*self.data, zero))
        
        non_axis_coords = _ravel_non_reduced_axes(self.coords, self.shape, axis)
        axis_coords = np.ravel_multi_index(np.array(self.coords)[axis, :],
                                           [self.shape[ax] for ax in axis])
        coords_2d = np.vstack((non_axis_coords, axis_coords))

        result_shape = tuple(self.shape[ax] for ax in range(self.ndim)
                             if ax not in axis)
        shape_2d = (math.prod(result_shape), math.prod([self.shape[ax] for ax in axis]))

        self = coo_array((self.data, coords_2d), shape_2d)
        res = (self._min_or_max(1, out, _max_or_min_axis, explicit))
        unraveled_coords = np.concatenate(np.unravel_index(res.coords, result_shape))
        
        return (coo_array((res.data, unraveled_coords), result_shape))


    def max(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.max(self, axis, out, explicit=explicit)
        
        return self._find_max_or_min(axis, out, np.max, np.maximum, explicit)
    
    def min(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.min(self, axis, out, explicit=explicit)
        
        return self._find_max_or_min(axis, out, np.min, np.minimum, explicit)
    
    def nanmax(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.nanmax(self, axis, out, explicit=explicit)
        
        return self._find_max_or_min(axis, out, np.nanmax, np.fmax, explicit)

    def nanmin(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.nanmin(self, axis, out, explicit=explicit)
        
        return self._find_max_or_min(axis, out, np.nanmin, np.fmin, explicit)

    def _find_arg_max_or_min(self, axis, out, _max_or_min, _max_or_min_axis, explicit):
        if axis is None:
            flat = self.reshape(-1)
            return flat._arg_min_or_max(0, out, _max_or_min, _max_or_min_axis, explicit)

        if not isinstance(axis, int):
            raise ValueError("'axis' should be int or None")
        
        if axis >= self.ndim or axis < -self.ndim:
            raise ValueError("axis out of range")
        
        axis = axis if axis>=0 else axis+self.ndim
 
        non_reduced_axes = [ax for ax in range(self.ndim) if ax != axis]
        non_reduced_shape = [self.shape[ax] for ax in non_reduced_axes]
        
        non_axis_coords = np.ravel_multi_index(
            np.array(self.coords)[non_reduced_axes, :], non_reduced_shape)
        
        axis_coords = np.ravel_multi_index(tuple([np.array(self.coords)[axis, :]]),
                                           tuple([self.shape[axis]]))
        coords_2d = np.vstack((non_axis_coords, axis_coords))

        result_shape = tuple(non_reduced_shape)
        shape_2d = (math.prod(result_shape), self.shape[axis])

        self = coo_array((self.data, coords_2d), shape_2d)
        res_flattened = self._arg_min_or_max(1, out, _max_or_min, _max_or_min_axis,
                                             explicit)
        res = res_flattened.reshape(result_shape)      
        return res
    
    def argmax(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.argmax(self, axis, out, explicit=explicit)
        return self._find_arg_max_or_min(axis, out, np.argmax, np.greater, explicit)
    
    def argmin(self, axis=None, out=None, *, explicit=False):
        if self.ndim<3:
            return _minmax_mixin.argmin(self, axis, out, explicit=explicit)
        return self._find_arg_max_or_min(axis, out, np.argmin, np.less, explicit)

    def maximum(self, other):
        """Element-wise maximum between this and another array/matrix."""
        return self._maximum_minimum_coo(other, 'maximum')
       

    def minimum(self, other):
        """Element-wise minimum between this and another array/matrix."""
        return self._maximum_minimum_coo(other, 'minimum')


    def diagonalnd(self, axis1=0, axis2=1, offset=0):
        x, y = self.shape[axis1], self.shape[axis2]
        diag_size = min(x, y)
        diag_shape = [*(self.shape[i] for i in range(len(self.shape))
                        if i!=axis1 and i!=axis2), diag_size]
        if offset <= -x or offset >= y:
            diag_shape[-1] = 0
            return coo_array(np.empty(tuple(diag_shape), dtype=self.data.dtype))
        diag_shape[-1] = min(x + min(offset, 0), y - max(offset, 0)) 
        # diag = np.zeros(diag_shape, dtype=self.dtype)
        diag_mask = (self.coords[axis1] + offset) == self.coords[axis2]
        new_data = self.data[diag_mask]
        inds = [idx[diag_mask] for idx in self.coords]
        # min shape out of both axes
        if offset>=0:
            ax = axis1
        else:
            ax = axis2
        inds = tuple(inds[i] for i in range(len(self.shape))
                     if i!=axis2 and i!=axis1) + tuple(inds[ax:ax+1])
        
        diag = coo_array((new_data, inds), diag_shape)
        return diag


def kron(a, b):
    # Ensure the arrays have the same number of dimensions
    ndim = max(a.ndim, b.ndim)
    a_shape = (1,) * (ndim - a.ndim) + a.shape
    b_shape = (1,) * (ndim - b.ndim) + b.shape
    
    # Expand the coordinates of a and b to match ndim, by reshaping
    a = a.reshape(a_shape)
    b = b.reshape(b_shape)
    
    # Compute the new shape
    new_shape = tuple(a_shape[i] * b_shape[i] for i in range(ndim))

    # Use broadcasting to compute the new coordinates
    a_coords_expanded = np.expand_dims(a.coords, axis=-1)  # shape (ndim, nnz_a, 1)
    b_coords_expanded = np.expand_dims(b.coords, axis=-2)  # shape (ndim, 1, nnz_b)

    new_coords = a_coords_expanded * np.array(b_shape).reshape(-1, 1, 1) + b_coords_expanded
    new_coords = new_coords.reshape(ndim, -1)

    # Compute the new data array using outer product
    new_data = np.outer(a.data, b.data).ravel()

    # Return the product array
    return coo_array((new_data, new_coords), shape=new_shape)


def vstack(arrays):
    # Ensure the input is a tuple of COO arrays
    if not isinstance(arrays, tuple):
        raise TypeError("Input must be a tuple of COO arrays/matrices.")
    if not all(isinstance(a, (coo_array, coo_matrix)) for a in arrays):
        raise TypeError("All elements of the tuple must be in COO format")

    # Ensure there is at least one array
    if len(arrays) == 0:
        raise ValueError("Input tuple must contain at least one array.")
    
    # Get the shape of the first array
    first_shape = arrays[0].shape
    
    # Ensure all arrays have the same shape along all but the first axis
    for a in arrays:
        if a.shape[1:] != first_shape[1:]:
            raise ValueError("All arrays must have the same shape along all but the first axis.")

    # Concatenate data from all arrays
    data = np.concatenate([a.data for a in arrays])
    
    # Concatenating coordinates
    coords = np.concatenate([a.coords for a in arrays], axis=1)
    first_dim_sizes = [a.shape[0] for a in arrays]
    first_dim_offsets = np.cumsum([0] + first_dim_sizes[:-1])
    
    # Adjust the first coordinate
    coords[0] += np.repeat(first_dim_offsets, [a.nnz for a in arrays])

    # New shape after stacking
    new_shape = (sum(first_dim_sizes),) + first_shape[1:]

    return coo_array((data, coords), shape=new_shape)


def hstack(arrays):
    # Ensure the input is a tuple of COO arrays
    if not isinstance(arrays, tuple):
        raise TypeError("Input must be a tuple of COO arrays/matrices.")
    if not all(isinstance(a, (coo_array, coo_matrix)) for a in arrays):
        raise TypeError("All elements of the tuple must be in COO format")

    # Ensure there is at least one array
    if len(arrays) == 0:
        raise ValueError("Input tuple must contain at least one array.")

    # Get the shape of the first array
    first_shape = arrays[0].shape

    # Ensure all arrays have the same shape along all but the second axis
    for a in arrays:
        if a.shape[0] != first_shape[0] or a.shape[2:] != first_shape[2:]:
            raise ValueError("All arrays must have the same shape along all but the second axis.")
    
    # Concatenate data from all arrays
    data = np.concatenate([a.data for a in arrays])

    # Concatenating coordinates
    coords = np.concatenate([a.coords for a in arrays], axis=1)
    second_dim_sizes = [a.shape[1] for a in arrays]
    second_dim_offsets = np.cumsum([0] + second_dim_sizes[:-1])

    # Adjust the second coordinate
    coords[1] += np.repeat(second_dim_offsets, [a.nnz for a in arrays])

    # New shape after stacking
    new_shape = (first_shape[0], sum(second_dim_sizes)) + first_shape[2:]

    return coo_array((data, coords), shape=new_shape)


def block_diag(arrays):
    # Ensure the input is a tuple of COO arrays
    if not isinstance(arrays, tuple):
        raise TypeError("Input must be a tuple of COO arrays/matrices.")
    if not all(isinstance(a, (coo_array, coo_matrix)) for a in arrays):
        raise TypeError("All elements of the tuple must be in COO format")
    
    # Ensure there is at least one array
    if len(arrays) == 0:
        raise ValueError("Input tuple must contain at least one array.")
    
    # Get the number of dimensions from the first array
    num_dims = arrays[0].ndim
    
    # Ensure all arrays have the same number of dimensions
    for a in arrays:
        if a.ndim != num_dims:
            raise ValueError("All arrays must have the same number of dimensions.")
    
    # Calculate the total nnz and result shape
    total_nnz = sum(a.nnz for a in arrays)
    result_shape = np.zeros(num_dims, dtype=int)

    for a in arrays:
        result_shape += np.array(a.shape)
    
    # Preallocate arrays for data and coordinates
    result_data = np.empty(total_nnz, dtype=arrays[0].data.dtype)
    result_coords = np.empty((num_dims, total_nnz), dtype=int)
    
    # Offset trackers for each dimension
    dim_offsets = np.zeros(num_dims, dtype=int)
    current_nnz = 0
    
    for a in arrays:
        nnz = a.nnz
        # Populate the result array
        result_data[current_nnz:current_nnz + nnz] = a.data
        
        # Calculate new coordinates with offsets
        for dim in range(num_dims):
            result_coords[dim, current_nnz:current_nnz + nnz] = a.coords[dim] + dim_offsets[dim]
        
        # Update offsets
        dim_offsets += np.array(a.shape)
        current_nnz += nnz
    
    # Return the block diagonal coo_array
    return coo_array((result_data, tuple(result_coords)), shape=tuple(result_shape))


def _block_diag(self):
    """
    Converts an N-D COO array into a 2-D COO array in block diagonal form.

    Parameters:
    self (coo_array): An N-Dimensional COO sparse array.

    Returns:
    coo_array: A 2-Dimensional COO sparse array in block diagonal form.
    """
    if self.ndim<2:
        raise ValueError("array must have atleast dim=2")
    num_blocks = math.prod(self.shape[:-2])
    n_col = self.shape[-1]
    n_row = self.shape[-2]
    res_arr = self.reshape((num_blocks, n_row, n_col))
    new_coords = np.empty((2, self.nnz), dtype = int)
    for axis in [1, 2]:
        new_coords[axis - 1] = res_arr.coords[axis] +\
            (res_arr.coords[0] * res_arr.shape[axis])

    new_shape = (num_blocks * n_row, num_blocks * n_col)
    return coo_array((self.data, tuple(new_coords)), shape=new_shape)


def _extract_block_diag(self, shape):
    n_row, n_col = shape[-2], shape[-1]

    # Extract data and coordinates from the block diagonal COO array
    data = self.data
    row, col = self.row, self.col

    # Initialize new coordinates array
    new_coords = np.empty((len(shape), self.nnz), dtype=int)
    
    # Calculate within-block indices
    new_coords[-2] = row % n_row
    new_coords[-1] = col % n_col

    # Calculate coordinates for higher dimensions
    temp_block_idx = row // n_row
    for i in range(len(shape) - 3, -1, -1):
        size = shape[i]
        new_coords[i] = temp_block_idx % size
        temp_block_idx = temp_block_idx // size

    # Create the new COO array with the original n-D shape
    return coo_array((data, tuple(new_coords)), shape=shape)


def _process_axes(ndim_a, ndim_b, axes):
    if isinstance(axes, int):
        if axes < 1 or axes > min(ndim_a, ndim_b):
            raise ValueError("axes integer is out of bounds for input arrays")
        axes_a = list(range(ndim_a - axes, ndim_a))
        axes_b = list(range(axes))
    elif isinstance(axes, (tuple, list)):
        if len(axes) != 2:
            raise ValueError("axes must be a tuple/list of length 2")
        axes_a, axes_b = axes
        if len(axes_a) != len(axes_b):
            raise ValueError("axes lists/tuples must be of the same length")
        if any(ax >= ndim_a or ax < -ndim_a for ax in axes_a) or \
           any(bx >= ndim_b or bx < -ndim_b for bx in axes_b):
            raise ValueError("axes indices are out of bounds for input arrays")
    else:
        raise TypeError("axes must be an integer or a tuple/list of integers")
    
    return list(axes_a), list(axes_b)

def _ravel_non_reduced_axes(coords, shape, axes):
    ndim = len(shape)
    non_reduced_axes = [ax for ax in range(ndim) if ax not in axes]

    if not non_reduced_axes:
        return np.zeros((len(coords[0])), dtype=int)  # Return an array with one row
    
    # Extract the shape of the non-reduced axes
    non_reduced_shape = [shape[ax] for ax in non_reduced_axes]
    
    # Extract the coordinates of the non-reduced axes
    non_reduced_coords = tuple(coords[idx] for idx in non_reduced_axes)
    
    # Ravel the coordinates into 1D
    raveled_coords = np.ravel_multi_index(non_reduced_coords, non_reduced_shape)
    
    return raveled_coords


def _validateaxes(axis, ndim, shape):
    if axis is not None:
        if not isinstance(axis, (int, tuple)):
            raise ValueError("'axis' should be int/tuple of ints")
        
        if type(axis) is int:
            axis = [axis]
            
        if len(axis)>ndim:
            raise ValueError("axis tuple has too many elements")
        
        if any(ax >= ndim or ax < -ndim for ax in axis):
            raise ValueError("axis out of range")
        
        axis = [ax if ax>=0 else ax+ndim for ax in axis]

        if any(shape[d] == 0 for d in axis):
            raise ValueError("zero-size array to reduction operation")

        # Check for duplicates
        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        if len(axis) == ndim:
            axis = None
        return axis

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
