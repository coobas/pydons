from pydons import MatStruct
import numpy as np
import tempfile


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
        dd = MatStruct.loadh5(tmpf.name)

    assert dd.field_a == field_a
    assert dd.field_b == field_b
    assert np.all(dd.field_c == field_c)


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
        dd = MatStruct.loadmat(tmpf.name)

    assert dd.field_a == field_a
    assert dd.field_b == field_b
    assert np.all(dd.field_c == field_c)
