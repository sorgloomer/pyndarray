
def test(aa):
    assert aa.array([1, 2, 3]).shape == (3,)
    assert aa.array([[1, 2, 3]]).shape == (1, 3)
    assert aa.array([[1], [2], [3]]).shape == (3, 1)
    assert aa.array([[1, 2], [3, 4], [5, 6]]).shape == (3, 2)
    assert aa.array([[[1], [3], [5]], [[1], [3], [5]]]).shape == (2, 3, 1)

    mx1 = aa.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
    ])
    mx2 = mx1[1:2, :]
    assert mx2.shape == (1, 4)

    assert aa.full((2, 3, 4, 5, 6), fill_value=0)[:, 1:, ...].shape == (2, 2, 4, 5, 6)
    assert aa.full((2, 3, 4, 5, 6), fill_value=0)[0, ..., 0].shape == (3, 4, 5)
    print(aa.full((2, 3, 4, 5, 6), fill_value=0)[[0], [0, 2], ..., 1].shape)
    assert aa.full((2, 3, 4, 5, 6), fill_value=0)[[0], [0, 2], ..., 1].shape == (1, 2, 4, 5)
    assert aa.array([1, 2, 3, 4, 5, 6, 7])[1:][1:][0] == 3


if __name__ == "__main__":
    import pyndarray as aa
    assert aa.striding.compute_row_continuous_dense_strides([2, 3, 4]) == [12, 4, 1]
    assert aa.striding.compute_column_continuous_dense_strides([2, 3, 4]) == [1, 2, 6]
    test(aa)
    import numpy as np
    test(np)
