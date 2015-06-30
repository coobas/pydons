from pydons import MatStruct, FileBrowser, LazyDataset
import numpy as np
import tempfile
import os


DATADIR = os.path.join(os.path.dirname(__file__), 'data')


def test_hdf5():
    d = MatStruct()

    field_a = np.random.rand(3, 2)
    field_b = np.random.rand(20, 50, 3)

    d.field_a = field_a
    d.group_b = MatStruct()
    d.group_b.field_b = field_b

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)

        dd = FileBrowser(tmpf.name)

        assert np.all(dd.field_a[:] == field_a)
        assert np.all(dd.group_b.field_b[:] == field_b)


def test_lazy_limits():
    d = MatStruct()

    field_a = np.random.rand(10)
    field_b = np.random.rand(1000)
    field_c = np.random.rand(10000)

    d.field_a = field_a
    d.field_b = field_b
    d.field_c = field_c

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)

        dd = FileBrowser(tmpf.name, lazy_min_size=field_a.size+1,
                         lazy_max_size=field_c.size-1)

        assert(dd.field_a._data is not None)
        assert(dd.field_b._data is None)
        assert(dd.field_c._data is None)

        assert(dd.field_a.size == field_a.size)
        assert(dd.field_a.shape == field_a.shape)
        assert(dd.field_b.size == field_b.size)
        assert(dd.field_b.shape == field_b.shape)
        assert(dd.field_c.size == field_c.size)
        assert(dd.field_c.shape == field_c.shape)

        assert np.all(dd.field_a == field_a)
        assert np.all(dd.field_a[:] == field_a)
        assert np.all(dd.field_b[:] == field_b[:])
        assert np.all(dd.field_c[:] == field_c)

        assert(dd.field_a._data is not None)
        assert(dd.field_b._data is not None)
        assert(dd.field_c._data is None)

        assert np.all(dd.field_b[10:20] == field_b[10:20])
        assert np.all(dd.field_c[100:1000] == field_c[100:1000])


def test_squeeze():
    fb = FileBrowser(os.path.join(DATADIR, 'test_data.h5'), squeeze=True)

    assert fb.one.shape == ()
    assert fb.ones.shape == (2, )
    assert fb.ones3d.shape == (2, 3)


def test_transpose():
    fb = FileBrowser(os.path.join(DATADIR, 'test_data.h5'), transpose=True)

    assert fb.one.shape == (1, )
    assert fb.ones.shape == (2, )
    assert fb.ones3d.shape == (3, 2, 1)


def test_global_cache():
    d = MatStruct()

    field_a = np.random.rand(10)
    field_b = np.random.rand(1000)
    field_c = np.random.rand(10000)

    d.field_a = field_a
    d.field_aa = field_a
    d.field_b = field_b
    d.field_c = field_c

    LazyDataset.MAX_CACHE_SIZE = field_b.size
    LazyDataset._clear_cache()

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)

        dd = FileBrowser(tmpf.name, lazy_min_size=field_a.size,
                         lazy_max_size=LazyDataset.MAX_CACHE_SIZE)

        # test if small size arrays are cached
        cache_size = sum(f.size for f in dd.values() if f.size <= field_a.size)
        assert all(f._LazyDataset__global_cache for f in dd.values() if f.size <= field_a.size)
        assert all(~f._LazyDataset__global_cache for f in dd.values() if f.size > field_a.size)
        assert LazyDataset._LazyDataset__cache_size == cache_size
        # cache field_b
        dd.field_b[:]
        # previous arrays should be discarded now
        cache_size = dd.field_b.size
        assert LazyDataset._LazyDataset__cache_size == cache_size
        assert dd.field_b._LazyDataset__global_cache
        assert all(~f._LazyDataset__global_cache for f in dd.values() if f.size < field_b.size)
        assert all(f._data is None for f in dd.values() if f.size < field_b.size)
