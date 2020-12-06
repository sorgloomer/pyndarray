from pyndarray import jagged


def test_unfold():
    actual = jagged.unfold_nested_iterable_shape_and_lists([range(3)]*2)
    expect = [[0, 1, 2], [0, 1, 2]]
    assert expect == actual
