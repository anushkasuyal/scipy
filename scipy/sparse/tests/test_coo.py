import numpy as np
from numpy.testing import assert_equal
import pytest
from scipy.sparse import coo_array, random_array


def test_shape_constructor():
    empty1d = coo_array((3,))
    assert empty1d.shape == (3,)
    assert_equal(empty1d.toarray(), np.zeros((3,)))

    empty2d = coo_array((3, 2))
    assert empty2d.shape == (3, 2)
    assert_equal(empty2d.toarray(), np.zeros((3, 2)))

    emptynd = coo_array((2,3,4,6,7))
    assert emptynd.shape == (2,3,4,6,7)
    assert_equal(emptynd.toarray(), np.zeros((2,3,4,6,7)))


def test_dense_constructor():
    # 1d
    res1d = coo_array([1, 2, 3])
    assert res1d.shape == (3,)
    assert_equal(res1d.toarray(), np.array([1, 2, 3]))

    # 2d
    res2d = coo_array([[1, 2, 3], [4, 5, 6]])
    assert res2d.shape == (2, 3)
    assert_equal(res2d.toarray(), np.array([[1, 2, 3], [4, 5, 6]]))

    # 4d
    arr4d = np.array([[[[3, 7], [1, 0]], [[6, 5], [9, 2]]],
                      [[[4, 3], [2, 8]], [[7, 5], [1, 6]]],
                      [[[0, 9], [4, 3]], [[2, 1], [7, 8]]]])
    res4d = coo_array(arr4d)
    assert res4d.shape == (3, 2, 2, 2)
    assert_equal(res4d.toarray(), arr4d)

    # 9d
    np.random.seed(12)
    arr9d = np.random.randn(2,3,4,7,6,5,3,2,4)
    res9d = coo_array(arr9d)
    assert res9d.shape == (2,3,4,7,6,5,3,2,4)
    assert_equal(res9d.toarray(), arr9d)

    # storing nan as element of sparse array
    nan_3d = coo_array([[[1, np.nan]], [[3, 4]], [[5, 6]]])
    assert nan_3d.shape == (3, 1, 2)
    assert_equal(nan_3d.toarray(), np.array([[[1, np.nan]], [[3, 4]], [[5, 6]]]))


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

    np.random.seed(12)
    arr7d = np.random.randn(2,4,1,6,5,3,2)
    res7d = coo_array((arr7d), shape=(2,4,1,6,5,3,2))
    assert res7d.shape == (2,4,1,6,5,3,2)
    assert_equal(res7d.toarray(), arr7d)


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


@pytest.mark.parametrize('shape', [(0,), (1,), (2,), (4,), (7,), (12,),
                                   (0,0), (2,0), (3,3), (4,7), (8,6),
                                   (0,0,0), (3,6,2), (3,9,4,5,2,1,6),
                                   (4,4,4,4,4), (5,10,3,13), (1,0,0,3),])
def test_sparse_constructor(shape):
    empty_arr = coo_array(shape)
    res = coo_array(empty_arr)
    assert res.shape == (shape)
    assert_equal(res.toarray(), np.zeros(shape))


@pytest.mark.parametrize('shape', [(0,), (1,), (2,), (4,), (7,), (12,),
                                   (0,0), (2,0), (3,3), (4,7), (8,6),
                                   (0,0,0), (3,6,2), (3,9,4,5,2,1,6),
                                   (4,4,4,4,4), (5,10,3,13), (1,0,0,3),])
def test_tuple_constructor(shape):
    np.random.seed(12)
    arr = np.random.randn(*shape)
    res = coo_array(arr)
    assert res.shape == shape
    assert_equal(res.toarray(), arr)


@pytest.mark.parametrize('shape', [(0,), (1,), (2,), (4,), (7,), (12,),
                                   (0,0), (2,0), (3,3), (4,7), (8,6),
                                   (0,0,0), (3,6,2), (3,9,4,5,2,1,6),
                                   (4,4,4,4,4), (5,10,3,13), (1,0,0,3),])
def test_tuple_constructor_with_shape(shape):
    np.random.seed(12)
    arr = np.random.randn(*shape)
    res = coo_array(arr, shape=shape)
    assert res.shape == shape
    assert_equal(res.toarray(), arr)
    

def test_tuple_constructor_for_dim_size_zero():
    # arrays with a dimension of size 0
    with pytest.raises(ValueError, match='exceeds matrix dimension'):
        coo_array(([9,8], ([1,2],[1,0])), shape=(4,0))

    emptyarr = coo_array(([], ([],[])), shape=(4,0))
    assert_equal(emptyarr.toarray(), np.empty((4,0)))


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


@pytest.mark.parametrize(('shape', 'new_shape'), [((4,9,6,5), (3,6,15,4)),
                                                  ((4,9,6,5), (36,30)),
                                                  ((4,9,6,5), (1080,)),
                                                  ((4,9,6,5), (2,3,2,2,3,5,3)),])
def test_reshape_nd(shape, new_shape):
    # reshaping a 4d sparse array
    rng = np.random.default_rng(23409823)

    arr4d = random_array(shape, density=0.6, random_state=rng, dtype=int)
    assert arr4d.shape == shape
    den4d = arr4d.toarray()

    exp_arr = den4d.reshape(new_shape)
    res_arr = arr4d.reshape(new_shape)
    assert res_arr.shape == new_shape
    assert_equal(res_arr.toarray(), exp_arr)


def test_reshape_invalid():
    # attempting invalid reshape
    arr = coo_array([1,2,3,4,5])
    with pytest.raises(ValueError, match="cannot reshape array"):
        arr.reshape((3,2))


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4,5),])
def test_nnz(shape):
    rng = np.random.default_rng(23409823)

    arr = random_array(shape, density=0.6, random_state=rng, dtype=int)
    assert arr.shape == (shape)
    assert arr.nnz == np.count_nonzero(arr.toarray())


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4,5),])
def test_transpose(shape):
    rng = np.random.default_rng(23409823)

    arr = random_array(shape, density=0.6, random_state=rng, dtype=int)
    assert arr.shape == (shape)
    exp_arr = arr.toarray().T
    trans_arr = arr.transpose()
    assert trans_arr.shape == shape[::-1]
    assert_equal(exp_arr, trans_arr.toarray())


@pytest.mark.parametrize(('shape', 'axis_perm'), [((3,), (0,)), ((2,3), (0,1)),
                                                  ((2,4,3,6,5,3), (1,2,0,5,3,4)),])
def test_transpose_with_axis(shape, axis_perm):
    rng = np.random.default_rng(23409823)

    arr = random_array(shape, density=0.6, random_state=rng, dtype=int)
    trans_arr = arr.transpose(axes=axis_perm)
    assert_equal(trans_arr.toarray(), np.transpose(arr.toarray(), axes=axis_perm))


def test_transpose_with_inconsistent_axis():
    with pytest.raises(ValueError, match="axes don't match matrix dimensions"):
        coo_array([1, 0, 3]).transpose(axes=(0, 1))

    with pytest.raises(ValueError, match="repeated axis in transpose"):
        coo_array([[1, 2, 0], [0, 0, 3]]).transpose(axes=(1, 1))


def test_1d_row_and_col():
    res = coo_array([1, -2, -3])
    assert_equal(res.col, np.array([0, 1, 2]))
    assert_equal(res.row, np.zeros_like(res.col))
    assert res.row.dtype == res.col.dtype
    assert res.row.flags.writeable is False

    res.col = [1, 2, 3]
    assert len(res.coords) == 1
    assert_equal(res.col, np.array([1, 2, 3]))
    assert res.row.dtype == res.col.dtype

    with pytest.raises(ValueError, match="cannot set row attribute"):
        res.row = [1, 2, 3]


def test_1d_toformats():
    res = coo_array([1, -2, -3])
    for f in [res.tobsr, res.tocsc, res.todia, res.tolil]:
        with pytest.raises(ValueError, match='Cannot convert'):
            f()
    for f in [res.tocoo, res.tocsr, res.todok]:
        assert_equal(f().toarray(), res.toarray())


@pytest.mark.parametrize('arg', [1, 2, 4, 5, 8,])
def test_1d_resize(arg: int):
    den = np.array([1, -2, -3])
    res = coo_array(den)
    den.resize(arg, refcheck=False)
    res.resize(arg)
    assert res.shape == den.shape
    assert_equal(res.toarray(), den)


@pytest.mark.parametrize('arg', zip([1, 2, 3, 4], [1, 2, 3, 4]))
def test_1d_to_2d_resize(arg: tuple[int, int]):
    den = np.array([1, 0, 3])
    res = coo_array(den)

    den.resize(arg, refcheck=False)
    res.resize(arg)
    assert res.shape == den.shape
    assert_equal(res.toarray(), den)


@pytest.mark.parametrize('arg', [1, 4, 6, 8,])
def test_2d_to_1d_resize(arg: int):
    den = np.array([[1, 0, 3], [4, 0, 0]])
    res = coo_array(den)
    den.resize(arg, refcheck=False)
    res.resize(arg)
    assert res.shape == den.shape
    assert_equal(res.toarray(), den)


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

    # 4d case
    arr4d = coo_array(([2, 3, 7], ([1, 0, 1], [0, 2, 0], [1, 2, 1], [1, 0, 1])))
    assert arr4d.nnz == 3
    expected = np.array(  # noqa: E501
        [[[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [3, 0]]],
         [[[0, 0], [0, 9], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]]
    )
    assert_equal(arr4d.toarray(), expected)
    arr4d.sum_duplicates()
    assert arr4d.nnz == 2
    assert_equal(arr4d.toarray(), expected)
    
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

    # for 2d sparse arrays (when the 0 data element is the only
    # element in the last row and last column)
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


def test_eliminate_zeros_nd():
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

    # for a 5d sparse array when all elements of data array are 0
    coords = ([0, 1, 1, 2], [0, 1, 0, 1], [1, 1, 2, 0], [0, 0, 2, 3], [1, 0, 0, 2])
    arr5d = coo_array(([0, 0, 0, 0], coords))
    assert arr5d.nnz == 4
    assert arr5d.count_nonzero() == 0
    arr5d.eliminate_zeros()
    assert arr5d.nnz == 0
    assert arr5d.count_nonzero() == 0
    assert_equal(arr5d.col, np.array([]))
    assert_equal(arr5d.row, np.array([]))
    assert_equal(arr5d.coords, ([], [], [], [], []))


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4),])
def test_add_dense(shape):
    rng = np.random.default_rng(23409823)
    sp_x = random_array(shape, density=0.6, random_state=rng, dtype=int)
    sp_y = random_array(shape, density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x + den_y
    res = sp_x + den_y
    assert type(res) is type(exp)
    assert_equal(res, exp)


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4),])
def test_add_sparse(shape):
    rng = np.random.default_rng(23409823)
    sp_x = random_array((shape), density=0.6, random_state=rng, dtype=int)
    sp_y = random_array((shape), density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x + den_y
    sparse_sum = sp_x + sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_add_sparse_with_inf():
    # addition of sparse arrays with an inf element
    den_a = np.array([[[0], [np.inf]], [[-3], [0]]])
    den_b = np.array([[[0], [1]], [[2], [3]]])
    dense_sum = den_a + den_b
    sparse_sum = coo_array(den_a) + coo_array(den_b)
    assert_equal(dense_sum, sparse_sum.toarray())


@pytest.mark.parametrize(('a_shape', 'b_shape'), [((7,), (12,)), ((6,4), (6,5)),
                                                  ((5,9,3,2), (9,5,2,3)),])
def test_add_sparse_with_inconsistent_shapes(a_shape, b_shape): 
    rng = np.random.default_rng(23409823)
    
    arr_a = random_array((a_shape), density=0.6, random_state=rng, dtype=int)
    arr_b = random_array((b_shape), density=0.6, random_state=rng, dtype=int)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        arr_a + arr_b


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4),])
def test_sub_dense(shape):
    rng = np.random.default_rng(23409823)
    sp_x = random_array(shape, density=0.6, random_state=rng, dtype=int)
    sp_y = random_array(shape, density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x - den_y
    res = sp_x - den_y
    assert type(res) is type(exp)
    assert_equal(res, exp)


@pytest.mark.parametrize('shape', [(0,), (1,), (3,), (7,), (0,0), (5,12),
                                   (8,7,3), (7,9,3,2,4),])
def test_sub_sparse(shape):
    rng = np.random.default_rng(23409823)

    sp_x = random_array(shape, density=0.6, random_state=rng, dtype=int)
    sp_y = random_array(shape, density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    
    dense_sum = den_x - den_y
    sparse_sum = sp_x - sp_y
    assert_equal(dense_sum, sparse_sum.toarray())


def test_sub_sparse_with_nan():
    # subtraction of sparse arrays with a nan element
    den_a = np.array([[[0], [np.nan]], [[-3], [0]]])
    den_b = np.array([[[0], [1]], [[2], [3]]])
    dense_sum = den_a - den_b
    sparse_sum = coo_array(den_a) - coo_array(den_b)
    assert_equal(dense_sum, sparse_sum.toarray())


@pytest.mark.parametrize(('a_shape', 'b_shape'), [((7,), (12,)), ((6,4), (6,5)),
                                                  ((5,9,3,2), (9,5,2,3)),])
def test_sub_sparse_with_inconsistent_shapes(a_shape, b_shape): 
    rng = np.random.default_rng(23409823)
    
    arr_a = random_array((a_shape), density=0.6, random_state=rng, dtype=int)
    arr_b = random_array((b_shape), density=0.6, random_state=rng, dtype=int)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        arr_a - arr_b



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
    assert type(res) is type(exp)
    assert_equal(res, exp)


def test_2d_matmul_multivector():
    den = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    arr2d = coo_array(den)
    exp = den @ den.T
    res = arr2d @ arr2d.T
    assert_equal(res.toarray(), exp)


mat_vec_shapes = [
    ((2, 3, 4, 5), (5,)), 
    ((0, 0), (0,)), 
    ((2, 3, 4, 7, 8), (8,)),
    ((4, 4, 2, 0), (0,)),
    ((6, 5, 3, 2, 4), (4, 1)), 
    ((2, 5), (5, 1)),
    ((3,), (3, 1)),
]
@pytest.mark.parametrize(('mat_shape', 'vec_shape'), mat_vec_shapes)
def test_nd_matmul_vector(mat_shape, vec_shape):
    rng = np.random.default_rng(23409823)

    sp_x = random_array(mat_shape, density=0.6, random_state=rng, dtype=int)
    sp_y = random_array(vec_shape, density=0.6, random_state=rng, dtype=int)
    den_x, den_y = sp_x.toarray(), sp_y.toarray()
    exp = den_x @ den_y
    res = sp_x @ den_y
    assert_equal(res,exp)


def test_1d_diagonal():
    den = np.array([0, -2, -3, 0])
    with pytest.raises(ValueError, match='diagonal requires two dimensions'):
        coo_array(den).diagonal()


def test_dot():
    # Example Usage
    a_coords = np.array([[0, 1, 2], [1, 0, 1], [0, 2, 1], [0,1,2]])  # Example coordinates for a 3D COO array
    a_data = np.array([1, 2, 3])
    a_shape = (3, 3, 4,3)
    
    b_coords = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1], [2,1,1]])  # Example coordinates for another 3D COO array
    b_data = np.array([4, 5, 6])
    b_shape = (3, 3, 3,4)
    
    a = coo_array(np.random.randint(0, 4, size=(2, 2, 3, 5)), (2, 2, 3, 5))
    b = coo_array(np.random.randint(0, 4, size=( 2, 5, 2)), (2, 5, 2))
    
    axes_a = [3]
    axes_b = [1]
    x = (np.tensordot(a.toarray(), b.toarray(), axes=[axes_a,axes_b]))
    print(x)
    print(coo_array(x))
    dotprod = a.dot(b, axes_a, axes_b)

    assert_equal(x, dotprod.toarray())
