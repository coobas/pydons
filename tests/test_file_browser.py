from pydons import MatStruct, FileBrowser
import numpy as np
import tempfile


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
