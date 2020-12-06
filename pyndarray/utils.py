def product(items):
    r = 1
    for i in items:
        r *= i
    return r


def is_all_integer(items):
    return all(isinstance(i, int) for i in items)


def assert_shape_param(shape):
    shape = assert_int_tuple_param(shape)
    assert len(shape) >= 0
    return shape


def assert_int_tuple_param(items):
    assert type(items) == tuple
    assert is_all_integer(items)
    return items


def assert_int_list_param(items):
    assert type(items) == list
    assert is_all_integer(items)
    return items


def coerect_shape_param(shape):
    return assert_shape_param(shape if type(shape) == tuple else tuple(shape))


def fill_array(buffer, value):
    for i in range(len(buffer)):
        buffer[i] = value
