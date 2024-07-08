import numpy as np
from numpy.testing import assert_equal
import pytest
from scipy.sparse import coo_array, random_array


def test_shape_constructor():
    np.random.seed(12)
    n, m, p = np.random.randint(low=0, high= 20), np.random.randint(low=0, high= 20), np.random.randint(low=0, high= 20), 
    
    empty1d = coo_array((n,))
    assert empty1d.shape == (n,)
    assert_equal(empty1d.toarray(), np.zeros((n,)))

    empty2d = coo_array((n, m))
    assert empty2d.shape == (n, m)
    assert_equal(empty2d.toarray(), np.zeros((n, m)))
    
    empty3d = coo_array((n, m, p))
    assert empty3d.shape == (n, m, p)
    assert_equal(empty3d.toarray(), np.zeros((n, m, p)))

    with pytest.raises(TypeError, match='invalid input format'):
        coo_array((3, 2, 2, 2))


def test_dense_constructor():
    # constructing 1d array
    res1d = coo_array([1, 2, 3])
    assert res1d.shape == (3,)
    assert_equal(res1d.toarray(), np.array([1, 2, 3]))

    # constructing 2d array
    res2d = coo_array([[1, 2, 3], [4, 5, 6]])
    assert res2d.shape == (2, 3)
    assert_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))

    # storing inf as element of sparse array
    inf_2d = coo_array([[1, 2, np.inf], [4, 5, 6]])
    assert inf_2d.shape == (2, 3)
    assert_equal(inf_2d.toarray(), np.array([[1, 2, np.inf], [4, 5, 6]]))
    assert(inf_2d.toarray()[0][2] == np.inf)

    # constructing 3d array
    res3d = coo_array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    assert res3d.shape == (2, 2, 3)
    assert_equal(res3d.toarray(), np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))

    # storing nan as element of sparse array
    nan_3d = coo_array([[[1, np.nan]], [[3, 4]], [[5, 6]]])
    assert nan_3d.shape == (3, 1, 2)
    assert_equal(nan_3d.toarray(), np.array([[[1, np.nan]], [[3, 4]], [[5, 6]]]))

    with pytest.raises(ValueError, match='shape must be a 1- or 2- or 3-tuple'):
        coo_array([[[[3]]], [[[4]]]])


def test_dense_constructor_with_shape():
    res1d = coo_array([1, 2, 3], shape=(3,))
    assert res1d.shape == (3,)
    assert_equal(res1d.toarray(), np.array([1, 2, 3]))

    res2d = coo_array([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
    assert res2d.shape == (2, 3)
    assert_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))

    res3d = coo_array([[[3]], [[4]]], shape=(2, 1, 1))
    assert res3d.shape == (2, 1, 1)
    assert_equal(res3d.toarray(), np.array([[[3]], [[4]]]))

    with pytest.raises(ValueError, match='shape must be a 1- or 2- or 3-tuple'):
        coo_array([[[[3]]], [[[4]]]], shape=(2, 1, 1, 1))


def test_dense_constructor_with_inconsistent_shape():
    with pytest.raises(ValueError, match='inconsistent shapes'):
        coo_array([1, 2, 3], shape=(4,))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        coo_array([1, 2, 3], shape=(3, 1))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        coo_array([[1, 2, 3]], shape=(3,))

    with pytest.raises(ValueError, match='inconsistent shapes'):
        coo_array([[[3]], [[4]]], shape=(1, 1, 1))

    with pytest.raises(ValueError,
                       match='axis 0 index 2 exceeds matrix dimension 2'):
        coo_array(([1], ([2],)), shape=(2,))

    with pytest.raises(ValueError,
                       match='axis 1 index 3 exceeds matrix dimension 3'):
        coo_array(([1,3], ([0, 1], [0, 3], [1, 1])), shape=(2, 3, 2))

    with pytest.raises(ValueError, match='negative axis 0 index: -1'):
        coo_array(([1], ([-1],)))

    with pytest.raises(ValueError, match='negative axis 2 index: -1'):
        coo_array(([1], ([0], [2], [-1])))


def test_1d_sparse_constructor():
    np.random.seed(12)
    n = np.random.randint(low=0, high= 20)

    empty1d = coo_array((n,))
    res = coo_array(empty1d)
    assert res.shape == (n,)
    assert_equal(res.toarray(), np.zeros((n,)))


def test_1d_tuple_constructor():
    res = coo_array(([9,8], ([1,2],)))
    assert res.shape == (3,)
    assert_equal(res.toarray(), np.array([0, 9, 8]))


def test_1d_tuple_constructor_with_shape():
    res = coo_array(([9,8], ([1,2],)), shape=(4,))
    assert res.shape == (4,)
    assert_equal(res.toarray(), np.array([0, 9, 8, 0]))


def test_2d_sparse_constructor():
    np.random.seed(12)
    n, m = np.random.randint(low=0, high= 20), np.random.randint(low=0, high= 20)
    
    empty2d = coo_array((n, m))
    res = coo_array(empty2d)
    assert res.shape == (n, m)
    assert_equal(res.toarray(), np.zeros((n, m)))


def test_2d_tuple_constructor():
    res = coo_array(([9,8,7], ([0,1,2],[0,1,0])))
    assert res.shape == (3,2)
    assert_equal(res.toarray(), np.array([[9, 0], [0, 8], [7, 0]]))


def test_2d_tuple_constructor_with_shape():
    res = coo_array(([9,8,7], ([0,1,2],[0,1,0])), shape=(3,3))
    assert res.shape == (3,3)
    assert_equal(res.toarray(), np.array([[9, 0, 0], [0, 8, 0], [7, 0, 0]]))

    # arrays with a dimension of size 0
    with pytest.raises(ValueError, match='exceeds matrix dimension'):
        dim0_arr = coo_array(([9,8], ([1,2],[1,0])), shape=(4,0))

    emptyarr = coo_array(([], ([],[])), shape=(4,0))
    assert_equal(emptyarr.toarray(), np.empty((4,0)))


def test_3d_sparse_constructor():
    np.random.seed(12)
    n, m, p = np.random.randint(low=0, high= 20), np.random.randint(low=0, high= 20), np.random.randint(low=0, high= 20), 
    empty3d = coo_array((n, m, p))
    res = coo_array(empty3d)
    assert res.shape == (n, m, p)
    assert_equal(res.toarray(), np.zeros((n, m, p)))


def test_3d_tuple_constructor():
    res = coo_array(([9,8,7], ([0,1,2],[0,1,0])))
    assert res.shape == (3,2)
    assert_equal(res.toarray(), np.array([[9, 0], [0, 8], [7, 0]]))


def test_3d_tuple_constructor_with_shape():
    res = coo_array(([9,8,7], ([1,0,1],[0,1,2],[0,1,0])), shape=(2,3,2))
    assert res.shape == (2,3,2)
    assert_equal(res.toarray(), np.array([[[0, 0,], [0, 8], [0, 0]], [[9, 0], [0, 0], [7, 0]]]))


def test_non_subscriptability():
    coo_2d = coo_array((2, 2))

    with pytest.raises(TypeError,
                        match="'coo_array' object does not support item assignment"):
        coo_2d[0, 0] = 1

    with pytest.raises(TypeError,
                       match="'coo_array' object is not subscriptable"):
        coo_2d[0, :]


def test_reshape_1d():
    # reshaping 1d sparse arrays
    arr1d = coo_array([1, 0, 3])
    assert arr1d.shape == (3,)

    col_vec = arr1d.reshape((3, 1))
    assert col_vec.shape == (3, 1)
    assert_equal(col_vec.toarray(), np.array([[1], [0], [3]]))

    row_vec = arr1d.reshape((1, 3))
    assert row_vec.shape == (1, 3)
    assert_equal(row_vec.toarray(), np.array([[1, 0, 3]]))

    # attempting invalid reshape
    with pytest.raises(ValueError, match="cannot reshape array"):
        arr1d.reshape((3,3))
    
    # attempting reshape with a size 0 dimension
    with pytest.raises(ValueError, match="cannot reshape array"):
        arr1d.reshape((3,0))
    

def test_reshape_2d():
    # reshaping 2d sparse arrays
    arr2d = coo_array([[1, 2, 0], [0, 0, 3]])
    assert arr2d.shape == (2, 3)
    
    # 2d to 2d
    to2darr = arr2d.reshape((3,2))
    assert to2darr.shape == (3,2)
    assert_equal(to2darr.toarray(), np.array([[1, 2], [0, 0], [0, 3]]))
    
    # 2d to 1d
    to1darr = arr2d.reshape((6,))
    assert to1darr.shape == (6,)
    assert_equal(to1darr.toarray(), np.array([1, 2, 0, 0, 0, 3]))
    
    # 2d to 3d
    to3darr = arr2d.reshape((2, 3, 1))
    assert to3darr.shape == (2, 3, 1)
    assert_equal(to3darr.toarray(), np.array([[[1], [2], [0]], [[0], [0], [3]]]))

    # attempting invalid reshape
    with pytest.raises(ValueError, match="cannot reshape array"):
        arr2d.reshape((1,3))
    

def test_reshape_3d():
    # reshaping 3d sparse arrays
    arr3d = coo_array([[[1, 2, 0], [0, 0, 3]],[[4, 0, 0], [0, 5, 6]]])
    assert arr3d.shape == (2, 2, 3)

    # 3d to 3d
    to3darr = arr3d.reshape((4,3,1))
    assert to3darr.shape == (4,3,1)
    assert_equal(to3darr.toarray(),
                np.array([[[1], [2], [0]], [[0], [0], [3]],
                [[4], [0], [0]], [[0], [5], [6]]]))
    
    # 3d to 2d
    to2darr = arr3d.reshape((6,2))
    assert to2darr.shape == (6,2)
    assert_equal(to2darr.toarray(), np.array([[1, 2], [0, 0], [0, 3], [4, 0], [0, 0], [5,6]]))

    # 3d to 1d
    to1darr = arr3d.reshape((12,))
    assert to1darr.shape == (12,)
    assert_equal(to1darr.toarray(), np.array([1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 5, 6]))

    # attempting invalid reshape
    with pytest.raises(ValueError, match="cannot reshape array"):
        arr3d.reshape((11,1))
    

def test_nnz():
    np.random.seed = 12
    n,m,p = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)

    arr1d = random_array((n,), density=0.6, random_state=rng, dtype=int)
    assert arr1d.shape == (n,)
    assert arr1d.nnz == np.count_nonzero(arr1d.toarray())

    arr2d = random_array((n, m), density=0.6, random_state=rng, dtype=int)
    assert arr2d.shape == (n, m)
    assert arr2d.nnz == np.count_nonzero(arr2d.toarray())

    arr3d = random_array((n, m, p), density=0.6, random_state=rng, dtype=int)
    assert arr3d.shape == (n, m, p)
    assert arr3d.nnz == np.count_nonzero(arr3d.toarray())


def test_transpose():
    arr1d = coo_array([1, 0, 3]).T
    assert arr1d.shape == (3,)
    assert_equal(arr1d.toarray(), np.array([1, 0, 3]))

    arr2d = coo_array([[1, 2, 0], [0, 0, 3]]).T
    assert arr2d.shape == (3, 2)
    assert_equal(arr2d.toarray(), np.array([[1, 0], [2, 0], [0, 3]]))

    arr3d = coo_array([[[1, 2, 0], [0, 0, 3]],[[4, 0, 0], [0, 5, 6]]]).T
    assert arr3d.shape == (3, 2, 2)
    assert_equal(arr3d.toarray(), np.array([[[1, 4], [0, 0]], [[2, 0], [0, 5]], [[0, 0], [3, 6]]]))


def test_transpose_with_axis():
    arr1d = coo_array([1, 0, 3]).transpose(axes=(0,))
    assert arr1d.shape == (3,)
    assert_equal(arr1d.toarray(), np.array([1, 0, 3]))

    arr2d = coo_array([[1, 2, 0], [0, 0, 3]]).transpose(axes=(0, 1))
    assert arr2d.shape == (2, 3)
    assert_equal(arr2d.toarray(), np.array([[1, 2, 0], [0, 0, 3]]))

    arr3d = coo_array([[[1, 2, 0], [0, 0, 3]],[[4, 0, 0], [0, 5, 6]]]).transpose(axes=(1,2,0))
    assert arr3d.shape == (2, 3, 2)
    assert_equal(arr3d.toarray(), np.array([[[1, 4], [2, 0], [0, 0]], [[0, 0], [0, 5], [3, 6]]]))

    with pytest.raises(ValueError, match="axes don't match matrix dimensions"):
        coo_array([1, 0, 3]).transpose(axes=(0, 1))

    with pytest.raises(ValueError, match="repeated axis in transpose"):
        coo_array([[1, 2, 0], [0, 0, 3]]).transpose(axes=(1, 1))


def test_1d_depth_row_col():
    res = coo_array([1, -2, -3])
    assert_equal(res.col, np.array([0, 1, 2]))
    assert_equal(res.row, np.zeros_like(res.col))
    assert_equal(res.depth, np.zeros_like(res.col))
    assert res.row.dtype == res.col.dtype
    assert res.depth.dtype == res.col.dtype
    assert res.row.flags.writeable is False
    assert res.depth.flags.writeable is False

    res.col = [1, 2, 3]
    assert len(res.coords) == 1
    assert_equal(res.col, np.array([1, 2, 3]))
    assert res.row.dtype == res.col.dtype
    assert res.depth.dtype == res.col.dtype

    with pytest.raises(ValueError, match="cannot set row attribute of a 1-dimensional sparse array"):
        res.row = [1, 2, 3]
    
    with pytest.raises(ValueError, match="cannot set depth attribute of a 1-dimensional sparse array"):
        res.depth = [1, 2, 3]



def test_2d_depth_row_col():
    res = coo_array([[1, -2], [0, 4]])
    assert_equal(res.col, np.array([0, 1, 1]))
    assert_equal(res.row, np.array([0, 0, 1]))
    assert_equal(res.depth, np.zeros_like(res.col))
    assert res.row.dtype == res.col.dtype
    assert res.depth.dtype == res.col.dtype
    assert res.depth.flags.writeable is False

    res.row = [3, 0]
    assert len(res.coords) == 2
    assert_equal(res.row, np.array([3, 0]))
    assert res.row.dtype == res.col.dtype
    assert res.depth.dtype == res.col.dtype
    
    with pytest.raises(ValueError, match="cannot set depth attribute of a 2-dimensional sparse array"):
        res.depth = [1, 3]


def test_1d_toformats():
    res = coo_array([1, -2, -3])
    for f in [res.tobsr, res.tocsc, res.todia, res.tolil]:
        with pytest.raises(ValueError, match='Cannot convert'):
            f()
    for f in [res.tocoo, res.tocsr, res.todok]:
        assert_equal(f().toarray(), res.toarray())


# @pytest.mark.parametrize('arg', [1, 2, 4, 5, 8])
# def test_1d_resize(arg: int):
#     den = np.array([1, -2, -3])
#     res = coo_array(den)
#     den.resize(arg, refcheck=False)
#     res.resize(arg)
#     assert res.shape == den.shape
#     assert_equal(res.toarray(), den)


# @pytest.mark.parametrize('arg', zip([1, 2, 3, 4], [1, 2, 3, 4]))
# def test_1d_to_2d_resize(arg: tuple[int, int]):
#     den = np.array([1, 0, 3])
#     res = coo_array(den)

#     den.resize(arg, refcheck=False)
#     res.resize(arg)
#     assert res.shape == den.shape
#     assert_equal(res.toarray(), den)

# @pytest.mark.parametrize('arg', zip([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
# def test_1d_to_3d_resize(arg: tuple[int, int, int]):
#     den = np.array([1, 0, 3])
#     res = coo_array(den)

#     den.resize(arg, refcheck=False)
#     res.resize(arg)
#     assert res.shape == den.shape
#     assert_equal(res.toarray(), den)


# @pytest.mark.parametrize('arg', [1, 4, 6, 8])
# def test_2d_to_1d_resize(arg: int):
#     den = np.array([[1, 0, 3], [4, 0, 0]])
#     res = coo_array(den)
#     den.resize(arg, refcheck=False)
#     res.resize(arg)
#     assert res.shape == den.shape
#     assert_equal(res.toarray(), den)


# @pytest.mark.parametrize('arg', zip([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))
# def test_2d_to_3d_resize(arg: tuple[int, int, int]):
#     den = np.array([[1, 0, 3], [4, 5, 7]])
#     res = coo_array(den)
#     den.resize(arg, refcheck=False)
#     res.resize(arg)
#     assert res.shape == den.shape
#     assert_equal(res.toarray(), den)


def test_sum_duplicates():
    # 1d case
    arr1d = coo_array(([2, 2, 2], ([1, 0, 1],)))
    assert arr1d.nnz == 3
    assert_equal(arr1d.toarray(), np.array([2, 4]))
    arr1d.sum_duplicates()
    assert arr1d.nnz == 2
    assert_equal(arr1d.toarray(), np.array([2, 4]))

    # 2d case
    arr2d = coo_array(([1, 2, 3, 4], ([0, 0, 1, 1], [1, 1, 0, 1])))
    assert arr2d.nnz == 4
    assert_equal(arr2d.toarray(), np.array([[0, 3], [3, 4]]))
    arr2d.sum_duplicates()
    assert arr2d.nnz == 3
    assert_equal(arr2d.toarray(), np.array([[0, 3], [3, 4]]))

    # 3d case
    arr3d = coo_array(([2, 3, 7], ([1, 0, 1], [0, 2, 0], [1, 2, 1])))
    assert arr3d.nnz == 3
    assert_equal(arr3d.toarray(), np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                        [[0, 9, 0], [0, 0, 0], [0, 0, 0]]]))
    arr3d.sum_duplicates()
    assert arr3d.nnz == 2
    assert_equal(arr3d.toarray(), np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 3]],
                        [[0, 9, 0], [0, 0, 0], [0, 0, 0]]]))
    
    # when there are no duplicates
    arr_nodups = coo_array(([1, 2, 3, 4], ([0, 0, 1, 1], [0, 1, 0, 1])))
    assert arr_nodups.nnz == 4
    arr_nodups.sum_duplicates()
    assert arr_nodups.nnz == 4


def test_eliminate_zeros_1d():
    arr1d = coo_array(([0, 0, 1], ([1, 0, 1],)))
    assert arr1d.nnz == 3
    assert arr1d.count_nonzero() == 1
    assert_equal(arr1d.toarray(), np.array([0, 1]))
    arr1d.eliminate_zeros()
    assert arr1d.nnz == 1
    assert arr1d.count_nonzero() == 1
    assert_equal(arr1d.toarray(), np.array([0, 1]))
    assert_equal(arr1d.col, np.array([1]))
    assert_equal(arr1d.row, np.array([0]))
    assert_equal(arr1d.depth, np.array([0]))


def test_eliminate_zeros_2d():
    # for 2d sparse arrays
    arr2d_a = coo_array(([1, 0, 3], ([0, 1, 1], [0, 1, 2])))
    assert arr2d_a.nnz == 3
    assert arr2d_a.count_nonzero() == 2
    assert_equal(arr2d_a.toarray(), np.array([[1, 0, 0], [0, 0, 3]]))
    arr2d_a.eliminate_zeros()
    assert arr2d_a.nnz == 2
    assert arr2d_a.count_nonzero() == 2
    assert_equal(arr2d_a.toarray(), np.array([[1, 0, 0], [0, 0, 3]]))
    assert_equal(arr2d_a.col, np.array([0, 2]))
    assert_equal(arr2d_a.row, np.array([0, 1]))
    assert_equal(arr2d_a.depth, np.array([0, 0]))

    # for 2d sparse arrays (when the 0 data element is the only element in the last row and last column)
    arr2d_b = coo_array(([1, 3, 0], ([0, 1, 1], [0, 1, 2])))
    assert arr2d_b.nnz == 3
    assert arr2d_b.count_nonzero() == 2
    assert_equal(arr2d_b.toarray(), np.array([[1, 0, 0], [0, 3, 0]]))
    arr2d_b.eliminate_zeros()
    assert arr2d_b.nnz == 2
    assert arr2d_b.count_nonzero() == 2
    assert_equal(arr2d_b.toarray(), np.array([[1, 0, 0], [0, 3, 0]]))
    assert_equal(arr2d_b.col, np.array([0, 1]))
    assert_equal(arr2d_b.row, np.array([0, 1]))
    assert_equal(arr2d_b.depth, np.array([0, 0]))

def test_eliminate_zeros_3d():
    # for 3d sparse arrays
    arr3d = coo_array(([1, 0, 0, 4], ([0, 1, 1, 2], [0, 1, 0, 1], [1, 1, 2, 0])))
    assert arr3d.nnz == 4
    assert arr3d.count_nonzero() == 2
    assert_equal(arr3d.toarray(), np.array([[[0, 1, 0], [0, 0, 0]], 
                                        [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [4, 0, 0]]]))
    arr3d.eliminate_zeros()
    assert arr3d.nnz == 2
    assert arr3d.count_nonzero() == 2
    assert_equal(arr3d.toarray(), np.array([[[0, 1, 0], [0, 0, 0]],
                                        [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [4, 0, 0]]]))
    assert_equal(arr3d.col, np.array([1, 0]))
    assert_equal(arr3d.row, np.array([0, 1]))
    assert_equal(arr3d.depth, np.array([0, 2]))

    # for 3d sparse arrays when all elements of data array are 0
    arr3d = coo_array(([0, 0, 0, 0], ([0, 1, 1, 2], [0, 1, 0, 1], [1, 1, 2, 0])))
    assert arr3d.nnz == 4
    assert arr3d.count_nonzero() == 0
    arr3d.eliminate_zeros()
    assert arr3d.nnz == 0
    assert arr3d.count_nonzero() == 0
    assert_equal(arr3d.col, np.array([]))
    assert_equal(arr3d.row, np.array([]))
    assert_equal(arr3d.depth, np.array([]))


def test_1d_add_dense():
    np.random.seed = 12
    n = np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x + den_y
    res = sp_x + den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)

    sp_v = random_array((n,), density=0.6, random_state=rng, dtype=int)
    sp_w = random_array((n+1,), density=0.6, random_state=rng, dtype=int)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        res = sp_v._add_dense(sp_w.toarray()) # why isn't  coo_array(den_c) + den_d producing the required error?


def test_2d_add_dense():
    np.random.seed = 12
    n,m = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x + den_y
    res = sp_x + den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)

    sp_v = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    sp_w = random_array((n,m+1), density=0.6, random_state=rng, dtype=int)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        res = sp_v._add_dense(sp_w.toarray())


def test_3d_add_dense():
    np.random.seed = 12
    n,m,p = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x + den_y
    res = sp_x + den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)

    sp_v = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    sp_w = random_array((n,m+1,p+2), density=0.6, random_state=rng, dtype=int)
    with pytest.raises(ValueError, match="Incompatible shapes"):
        res = sp_v._add_dense(sp_w.toarray())


def test_1d_add_sparse():
    np.random.seed = 12
    n = np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x + den_y
    # this routes through CSR format
    sparse_sum = sp_x + sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_2d_add_sparse():
    np.random.seed = 12
    n,m = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x + den_y
    # this routes through CSR format
    sparse_sum = sp_x + sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_3d_add_sparse():
    np.random.seed = 12
    n,m,p = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x + den_y
    sparse_sum = sp_x + sp_y
    assert_equal(dense_sum, sparse_sum.toarray())

    # addition of sparse arrays with an inf element
    den_a = np.array([[[0], [np.inf]], [[-3], [0]]])
    den_b = np.array([[[0], [1]], [[2], [3]]])
    dense_sum = den_a + den_b
    sparse_sum = coo_array(den_a) + coo_array(den_b)
    assert_equal(dense_sum, sparse_sum.toarray())


def test_1d_sub_dense():
    np.random.seed = 12
    n = np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x - den_y
    res = sp_x - den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)


def test_2d_sub_dense():
    np.random.seed = 12
    n,m = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x - den_y
    res = sp_x - den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)

def test_3d_sub_dense():
    np.random.seed = 12
    n,m,p = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x - den_y
    res = sp_x - den_y
    assert type(res) == type(exp)
    assert_equal(res, exp)
        

def test_1d_sub_sparse():
    np.random.seed = 12
    n = np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x - den_y
    sparse_sum = sp_x - sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_2d_sub_sparse():
    np.random.seed = 12
    n,m = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x - den_y
    sparse_sum = sp_x - sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_3d_sub_sparse():
    np.random.seed = 12
    n,m,p = np.random.randint(low=0, high=20), np.random.randint(low=0, high=20), np.random.randint(low=0, high=20)
    rng = np.random.RandomState(23409823)
    sp_x = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((n,m,p), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x - den_y
    sparse_sum = sp_x - sp_y
    assert_equal(dense_sum, sparse_sum.toarray())

    
    # subtraction of sparse arrays with a nan element
    den_a = np.array([[[0], [np.nan]], [[-3], [0]]])
    den_b = np.array([[[0], [1]], [[2], [3]]])
    dense_sum = den_a - den_b
    sparse_sum = coo_array(den_a) - coo_array(den_b)
    assert_equal(dense_sum, sparse_sum.toarray())


def test_1d_matmul_vector():
    den_a = np.array([0, -2, -3, 0])
    den_b = np.array([0, 1, 2, 3])
    exp = den_a @ den_b
    res = coo_array(den_a) @ den_b
    assert np.ndim(res) == 0
    assert_equal(res, exp)


def test_1d_matmul_multivector():
    den = np.array([0, -2, -3, 0])
    other = np.array([[0, 1, 2, 3], [3, 2, 1, 0]]).T
    exp = den @ other
    res = coo_array(den) @ other
    assert type(res) == type(exp)
    assert_equal(res, exp)


def test_2d_matmul_multivector():
    den = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    arr2d = coo_array(den)
    exp = den @ den.T
    res = arr2d @ arr2d.T
    assert_equal(res.toarray(), exp)


def test_1d_diagonal():
    den = np.array([0, -2, -3, 0])
    with pytest.raises(ValueError, match='diagonal requires two dimensions'):
        coo_array(den).diagonal()
