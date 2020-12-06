import dataclasses
from typing import Tuple, List, Any, Sequence, MutableSequence, Union, Literal, Mapping

from pyndarray import utils, dtypes, striding
from pyndarray.array_methods import ArrayMethodsMixin
from pyndarray.dtypes import DType
from pyndarray.slicing import Slicer


class ArrayIndexingMixin:
    shape: Tuple[int]

    def __getitem__(self, key):
        indices = _param_indices(key)
        return self._get_item_sweet(indices)

    def __setitem__(self, key, value):
        indices = _param_indices(key)
        return self._set_item_sweet(indices, value)

    def get_slice(self, indices: Tuple[any]):
        raise NotImplemented

    def set_slice(self, indices: Tuple[any], value):
        raise NotImplemented

    def get_cell(self, indices: Tuple[any]):
        raise NotImplemented

    def set_cell(self, indices: Tuple[any], value):
        raise NotImplemented

    def _get_item_sweet(self, indices: Tuple[int]):
        if utils.is_all_integer(indices):
            return self.get_cell(indices)
        return self.get_slice(indices)

    def _set_item_sweet(self, indices: Tuple[int], value):
        if utils.is_all_integer(indices):
            return self.set_cell(indices, value)
        return self.set_slice(indices, value)


@dataclasses.dataclass(frozen=True)
class Array(ArrayIndexingMixin, ArrayMethodsMixin):
    slicer = Slicer
    shape: Tuple[int]
    dims: int
    buffer: MutableSequence[Any]
    offset: int
    strides: List[int]
    dtype: DType
    indexers: Tuple

    @classmethod
    def factory(cls, shape: Tuple[int], buffer, offset: int, strides: Tuple[int], dtype, indexers: Tuple):
        shape = utils.assert_shape_param(shape)
        dims = len(shape)
        strides = utils.assert_int_tuple_param(strides)
        assert isinstance(dtype, DType)
        assert len(strides) == dims
        assert type(offset) == int
        assert offset >= 0
        max_buffer_index = utils.product(shape) + offset
        assert len(buffer) >= max_buffer_index
        assert type(indexers) == tuple
        assert len(indexers) == dims
        return cls(
            shape=shape,
            dims=dims,
            buffer=buffer,
            strides=strides,
            offset=offset,
            indexers=indexers,
            dtype=dtype,
        )

    @classmethod
    def allocate(
            cls,
            shape,
            buffer=None,
            offset=None,
            strides=None,
            order: Union[Literal['C'], Literal['F'], None] = None,
            fill_value=None,
            dtype: DType = None,
    ):
        shape = utils.coerect_shape_param(shape)
        if offset is None:
            offset = 0
        used_buffer_length = utils.product(shape) + offset
        if dtype is None:
            dtype = dtypes.object
        if buffer is None:
            buffer = dtype.create_buffer(used_buffer_length, fill_value)
        assert len(buffer) >= used_buffer_length + offset
        if strides is None:
            strides = striding.compute_dense_strides(shape, order=order)
        strides = tuple(strides)
        assert utils.is_all_integer(strides)
        assert len(strides) == len(shape)
        indexers = tuple(range(a) for a in shape)
        return cls.factory(
            shape=shape,
            buffer=buffer,
            offset=offset,
            strides=strides,
            indexers=indexers,
            dtype=dtype,
        )

    def _item_buffer_index(self, indices: Tuple[int]):
        assert len(indices) == len(self.strides)
        return sum(stride * indexer[index] for stride, indexer, index in zip(self.strides, self.indexers, indices))

    def get_cell(self, indices: Tuple[int]):
        buffer_index = self._item_buffer_index(indices)
        return self.buffer[buffer_index]

    def set_cell(self, indices: Tuple[int], value):
        buffer_index = self._item_buffer_index(indices)
        self.buffer[buffer_index] = value

    def get_slice(self, indices: Tuple[any]):
        slicer = self.slicer(self, indices)
        return self.factory(
            shape=slicer.new_shape,
            offset=slicer.new_offset,
            strides=slicer.new_strides,
            buffer=self.buffer,
            dtype=self.dtype,
            indexers=slicer.new_indexers,
        )

    def set_slice(self, indices: Tuple[any], value):
        _array_set(self.get_slice(indices), value)


def _array_set(target, source):
    if target.shape != source.shape:
        raise ValueError(f"wrong shape given for array set, given {source.shape}, required {target.shape}")
    for ijk in target.coords():
        target.set_cell(ijk, source.get_cell(ijk))


def _param_indices(key):
    if isinstance(key, tuple):
        return key
    return key,


def _dot(xs, ys):
    return sum(x * y for x, y in zip(xs, ys))
