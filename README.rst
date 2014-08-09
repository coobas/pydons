Pydons is a collection of numerical data manipulation add-ons.


MatStruct class
---------------

MatStruct is an OrderedDict-based Matlab-like container. Typical usage:

::

  import pydons
  import numpy as np
  struct = pydons.MatStruct()
  struct.string = 'A string'
  struct.list = ['list', 0, [1, 2]]
  struct.numpy = np.random.rand(3,3)

IPython representation is customized:

::

  In [12]: struct
  Out[12]: 
  string: A string
    list: ['list', 0, [1, 2]]
   numpy: [[ 0.71539338  0.69970494  0.19328026]
   [ 0.28645949  0.15262059  0.23362895]
   [ 0.14518748  0.79911631  0.22522526]]

MatStruct can be serialized to HDF5 files using ``hdf5storage``:

::

  In [15]: struct.saveh5('struct.h5')
  In [16]: pydons.MatStruct.loadh5('struct.h5')
  Out[16]: 
    list: ['list', 0, [1, 2]]
   numpy: [[ 0.71539338  0.69970494  0.19328026]
   [ 0.28645949  0.15262059  0.23362895]
   [ 0.14518748  0.79911631  0.22522526]]
  string: A string

(the field order is not maintained---to be fixed soon). 
Matlab HDF5 files can be used as well.
In this case, numpy arrays are transposed and additional
Matlab fields are written in the file.

::

  In [17]: struct.savemat('struct.mat')
  In [18]: pydons.MatStruct.loadmat('struct.mat')
  Out[18]: 
    list: ['list', 0, [1, 2]]
   numpy: [[ 0.71539338  0.69970494  0.19328026]
   [ 0.28645949  0.15262059  0.23362895]
   [ 0.14518748  0.79911631  0.22522526]]
  string: A string

This software is distributed under the MIT license (see the LICENSE file).

Documentation
-------------

`pydons.readthedocs.org <http://pydons.readthedocs.org/>`_
