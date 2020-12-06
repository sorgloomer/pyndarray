from collections import abc


def _unfold_iterables(data, parent_indices):
    if isinstance(data, abc.Iterable):
        axis = list(data)
        axis = [_unfold_iterables(a, parent_indices + (i,)) for i, a in enumerate(axis)]
        shapes, datas = zip(*axis)
        shape_tail = list(set(shapes))
        if len(shape_tail) != 1:
            raise ValueError(f"jagged array has wrong lengths at indices {parent_indices}")
        return (len(axis),) + shape_tail[0], datas
    return (), data


def unfold_nested_iterable_shape_and_lists(data):
    return _unfold_iterables(data, ())


def getitem(data, indices):
    return _getitem(data, indices, 0)


def _getitem(data, indices, dim_cursor):
    if dim_cursor >= len(indices):
        return data
    return _getitem(data[indices[dim_cursor]], indices, dim_cursor + 1)


def test_unfold():
    actual = jagged.unfold_nested_iterable_shape_and_lists([range(3)]*2)
    expect = [[0, 1, 2], [0, 1, 2]]
    assert expect == actual
