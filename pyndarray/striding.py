def compute_row_continuous_dense_strides(shape):
    dims = len(shape)
    result = [1] * dims
    for i in range(1, dims):
        result[i] = result[i - 1] * shape[dims - i]
    result.reverse()
    return result


def compute_column_continuous_dense_strides(shape):
    dims = len(shape)
    result = [1] * dims
    for i in range(1, dims):
        result[i] = result[i - 1] * shape[i - 1]
    return result


def compute_dense_strides(shape, order=None):
    fn = _ORDERS.get(order)
    if fn is None:
        raise ValueError(f"stride order must be one of f{list(_ORDERS)}")
    return fn(shape)


_ORDERS = {
    None: compute_row_continuous_dense_strides,
    'row': compute_row_continuous_dense_strides,
    'C': compute_row_continuous_dense_strides,
    'column': compute_column_continuous_dense_strides,
    'F': compute_column_continuous_dense_strides,
}
