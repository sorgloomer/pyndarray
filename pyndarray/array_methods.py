import itertools


def coords(self):
    return itertools.product(*[range(w) for w in self.shape])


def copy(self, dtype=None):
    if dtype is None:
        dtype = self.dtype
    result = self.allocate(shape=self.shape, dtype=dtype)
    for ijk in coords(self):
        result[ijk] = self[ijk]


def to_list(self):
    shape_head = self.shape[0]
    if len(self.shape) <= 1:
        return [self.get_cell(i) for i in range(shape_head)]
    return [
        to_list(self.get_slice((i, ...)))
        for i in range(shape_head)
    ]


class ArrayMethodsMixin:
    coords = coords
    copy = copy
    to_list = to_list
