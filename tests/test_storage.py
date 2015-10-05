from pydons import MatStruct, loadmat, loadh5, load
import numpy as np
import tempfile
from pydons import _OrderedDict as OrderedDict
import h5py


def test_h5_storage():
    d = MatStruct()

    field_a = 'field a'
    field_b = (1, 'string', [3, 2.1])
    field_c = np.random.rand(3, 4)

    d.field_a = field_a
    d.field_b = field_b
    d.field_c = field_c

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)

        for load_func in (MatStruct.loadh5, loadh5, load):
            dd = load_func(tmpf.name)

            assert isinstance(dd, MatStruct)
            assert dd.field_a == field_a
            assert dd.field_b == field_b
            assert np.all(dd.field_c == field_c)


def test_loadh5():
    d = MatStruct()

    # field_a = 'field a'
    # field_b = (1, 'string', [3, 2.1])
    field_c = np.random.rand(3, 4)
    field_d = MatStruct()
    field_d.array = np.random.rand(3, 4, 6)

    # d.field_a = field_a
    # d.field_b = field_b
    d.field_c = field_c
    d.field_d = field_d

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        with h5py.File(tmpf.name, 'w') as fh:
            for k in ('field_c', ):
                fh.create_dataset(k, data=d[k])
            fh.create_group('field_d')
            grp = fh['field_d']
            grp.create_dataset('array', data=d.field_d.array)

        for load_func in (MatStruct.loadh5, loadh5, load):
            dd = load_func(tmpf.name)

        # TODO fails for upstream hdf5storage
        assert isinstance(dd, MatStruct)
        assert isinstance(dd.field_d, MatStruct)
        # assert dd.field_a == field_a
        # assert dd.field_b == field_b
        assert np.all(dd.field_c == field_c)
        assert np.all(dd.field_d.array == field_d.array)


def test_h5_storage_substruct():
    d = MatStruct()

    field_d = MatStruct()
    field_d.array = np.random.rand(3, 4, 6)

    d.field_d = field_d

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)
        dd = MatStruct.loadh5(tmpf.name)

    assert np.all(dd.field_d.array == field_d.array)
    assert isinstance(dd.field_d, MatStruct)


def test_mat_storage():
    d = MatStruct()

    field_a = 'field a'
    field_b = (1, 'string', [3, 2.1])
    field_c = np.random.rand(3, 4)

    d.field_a = field_a
    d.field_b = field_b
    d.field_c = field_c

    with tempfile.NamedTemporaryFile(suffix=".mat") as tmpf:
        d.savemat(tmpf.name)
        for load_func in (MatStruct.loadh5, loadh5, load):
            dd = load_func(tmpf.name)

            assert isinstance(dd, MatStruct)
            assert dd.field_a == field_a
            assert dd.field_b == field_b
            assert np.all(dd.field_c == field_c)


def test_dedict_h5():
    od = OrderedDict((('x', 1), ('y', 2.1)))
    od['z'] = OrderedDict((('x', 1), ('y', 2.1)))
    ms = MatStruct(od, dedict=True)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        ms.saveh5(tmpf.name)
        ms2 = MatStruct.loadh5(tmpf.name)
    assert isinstance(ms2['z'], OrderedDict)
    assert ms.diff(ms2).diff_max == 0
