from pydons import MatStruct, FileBrowser, LazyDataset
import netCDF4
import numpy as np
import tempfile
import os


DATADIR = os.path.join(os.path.dirname(__file__), 'data')


def test_netcdf4():
    d = MatStruct()

    data1 = np.random.rand(np.random.randint(1, 1000))

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpf:
        fh = netCDF4.Dataset(tmpf.name, mode='w')
        grp = fh.createGroup('mygroup')
        dim1 = grp.createDimension('dim1')
        var1 = grp.createVariable('var1', data1.dtype.str, (dim1.name, ))
        var1[:] = data1
        fh.close()

        dd = FileBrowser(tmpf.name)

        assert 'mygroup' in dd
        assert 'var1' in dd.mygroup
        assert np.all(dd.mygroup.var1[:] == data1)
