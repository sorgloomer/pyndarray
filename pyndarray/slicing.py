from collections import abc

newaxis = None


class IndexType:
    axis_use: int
    axis_new: int
    axis_ellipsis: int

    @staticmethod
    def get_by_value(value):
        result = _TYPE_MAP.get(type(value), None)
        if result is None:
            raise TypeError(f"type(index) must be one of {list(_TYPE_MAP)}, but was {type(value)}")
        return result

    @staticmethod
    def apply(builder, index):
        raise NotImplemented


class IndexSlice(IndexType):
    axis_use = 1
    axis_new = 0
    axis_ellipsis = 0

    @staticmethod
    def apply(slicer, index):
        sliced_indexer = _slice_indexer(slicer.old_indexers[slicer.dim_cursor], index)
        slicer.new_indexers.append(sliced_indexer)
        slicer.new_strides.append(slicer.old_strides[slicer.dim_cursor])
        slicer.dim_cursor += 1


class IndexNumber(IndexType):
    axis_use = 1
    axis_new = 0
    axis_ellipsis = 0

    @staticmethod
    def apply(slicer, index):
        sliced_indexer = _slice_indexer(slicer.old_indexers[slicer.dim_cursor], index)
        slicer.new_offset += slicer.old_strides[slicer.dim_cursor] * sliced_indexer
        slicer.dim_cursor += 1


class IndexNewAxis(IndexType):
    axis_use = 0
    axis_new = 1
    axis_ellipsis = 0

    @staticmethod
    def apply(slicer, index):
        slicer.new_indexers.append(range(1))
        slicer.new_strides.append(0)


class IndexEllipsis(IndexType):
    axis_use = 0
    axis_new = 0
    axis_ellipsis = 1

    @staticmethod
    def apply(slicer, index):
        for _ in range(slicer.ellipsis_length):
            slicer.new_indexers.append(slicer.old_indexers[slicer.dim_cursor])
            slicer.new_strides.append(slicer.old_strides[slicer.dim_cursor])
            slicer.dim_cursor += 1


def _slice_indexer(indexer, myslice: slice):
    if isinstance(myslice, slice) or isinstance(myslice, int):
        return indexer[myslice]
    if isinstance(myslice, abc.Iterable):
        return [indexer[s] for s in myslice]
    raise TypeError("indices must be one of int, slice, list")


class Slicer:
    def __init__(self, source, indices):
        self.old_dims = source.dims
        self.old_strides = source.strides
        self.old_indexers = source.indexers
        self.new_offset = source.offset
        self.new_strides = []
        self.new_indexers = []
        self.indices = indices
        self.dim_cursor = 0
        self.index_types = [IndexType.get_by_value(i) for i in indices]
        self.ellipsis_length = self.analyze_ellipsis_length()
        self.new_shape = self.compute_new_indices_return_shape()
        self.new_strides = tuple(self.new_strides)
        self.new_indexers = tuple(self.new_indexers)

    def compute_new_indices_return_shape(self):
        for index_type, index in zip(self.index_types, self.indices):
            index_type.apply(self, index)
        assert self.dim_cursor == self.old_dims
        return tuple(len(indexer) for indexer in self.new_indexers)

    def analyze_ellipsis_length(self):
        use_dims = 0
        ellipsis_count = 0
        for index_type in self.index_types:
            use_dims += index_type.axis_use
            ellipsis_count += index_type.axis_ellipsis

        if ellipsis_count > 1:
            raise ValueError("a slice must contain at most one ellipsis")
        ellipsis_length = self.old_dims - use_dims
        if ellipsis_length < 0:
            raise ValueError(f"too much indices, {-ellipsis_length} more than needed")
        if ellipsis_count == 0 and ellipsis_length != 0:
            raise ValueError(f"too few indices, {ellipsis_length} fewer than needed")
        return ellipsis_length


_TYPE_MAP = {
    int: IndexNumber,
    list: IndexSlice,
    slice: IndexSlice,
    tuple: IndexSlice,
    type(Ellipsis): IndexEllipsis,
    type(newaxis): IndexNewAxis,
}
