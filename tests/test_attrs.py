from pydons import MatStruct, FileBrowser
import numpy as np
import tempfile
import h5py


def test_var_a():
    d = MatStruct()
    d.r = np.random.rand(3, 4)

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpf:
        file_name = tmpf.name
        with h5py.File(file_name, 'w') as fh:
            dset = fh.create_dataset('r', data=d.r)
            dset.attrs.create('a_text', 'text'.encode('utf8'))
            dset.attrs.create('a_num', 1)
            dset.attrs.create('a_arr', np.ones(3))
            dset.attrs.create('_hidden', True)

        fb = FileBrowser(file_name)
        assert fb.r.attrs.a_text.decode('utf8') == u'text'
        assert fb.r.attrs.a_num == 1
        assert np.all(fb.r.attrs.a_arr == 1)
        assert fb.r.attrs['_hidden']
