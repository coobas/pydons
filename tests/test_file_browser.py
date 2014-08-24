from pydons import MatStruct, FileBrowser
import numpy as np
import tempfile


def test_hdf5():
    d = MatStruct()

    field_a = np.random.rand(3, 4)
    field_b = np.random.rand(2, 5, 3)

    d.field_a = field_a
    d.group_b = MatStruct()
    d.group_b.field_b = field_b

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        d.saveh5(tmpf.name)

        dd = FileBrowser(tmpf.name)

        assert np.all(dd.field_a[:] == field_a)
        assert np.all(dd.group_b.field_b[:] == field_b)
