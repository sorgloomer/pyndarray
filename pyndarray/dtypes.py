class SparseBuffer:
    def __init__(self, length, default):
        self.default = default
        self.length = length
        self.indexer = range(self.length)
        self.data = dict()

    def __setitem__(self, key, value):
        key = self.indexer[key]
        if value != self.default:
            self.data[key] = value
        else:
            self.data.pop(key)

    def __getitem__(self, key):
        key = self.indexer[key]
        return self.data.get(key, self.default)

    def __len__(self):
        return self.length


def make_typed_array(typecode, length, fill_value=None):
    import array
    result = array.array(typecode, length)
    if fill_value is not None:
        for i in range(length):
            result[i] = fill_value
    return result


class DType:
    def create_buffer(self, length, fill_value=None):
        raise NotImplemented


class DTypeInt32(DType):
    def create_buffer(self, length, fill_value=None):
        return make_typed_array(typecode='i', length=length, fill_value=fill_value)


class DTypeObject(DType):
    def create_buffer(self, length, fill_value=None):
        return [fill_value] * length


class DTypeObjectSparse(DType):
    def create_buffer(self, length, fill_value=None):
        return SparseBuffer(length=length, default=fill_value)


int32 = DTypeInt32()
object = DTypeObject()
sparse = DTypeObjectSparse()
