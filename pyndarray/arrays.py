from typing import Union, Literal

from pyndarray import jagged
from pyndarray.array_class import Array
from pyndarray.dtypes import DType


def zeros(shape, dtype=None):
    return full(shape=shape, fill_value=0, dtype=dtype)


def full(shape, fill_value=None, dtype=None):
    return Array.allocate(
        shape=shape,
        fill_value=fill_value,
        dtype=dtype,
    )


def array(data, shape=None, dtype=None):
    computed_shape, computed_data = jagged.unfold_nested_iterable_shape_and_lists(data)
    if shape is not None and computed_shape != shape:
        raise ValueError("given jagged array shape did not match given shape")
    result = ndarray(shape=computed_shape, dtype=dtype)
    for ijk in result.coords():
        item = jagged.getitem(computed_data, ijk)
        result.set_cell(ijk, item)
    return result


def ndarray(
        shape,
        buffer=None,
        offset=None,
        strides=None,
        order: Union[Literal['C'], Literal['F'], None] = None,
        dtype: DType = None,
):
    return Array.allocate(
        shape=shape,
        buffer=buffer,
        offset=offset,
        strides=strides,
        order=order,
        dtype=dtype,
    )
