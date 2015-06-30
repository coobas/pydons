Pydons is a collection of manipulation add-ons for hierarchichal numerical data.

MatStruct class
---------------

MatStruct is an ordered dict with string-only keys, which are accessible also
as properties. This makes the notation easier (``obj.group.subgroup.variable`` instead of 
``obj['group']['subgroup']['variable']``) and enables IPython's auto complete.

MatStruct can be serialized to HDF5 or Matlab files using the excellent
`hdf5storage <https://github.com/frejanordsiek/hdf5storage>`_ package.

LazyDataset class
-----------------

A lazy evaluate proxy class for data sets in HDF5 or netCDF4 files.

FileBrowser class
-----------------

FileBrowser employs MatStruct and LazyDataset to enable easy and fast browsing
of netCDF4 or HDF5 files.

Examples
--------

Items can be added using either ['keys'] or .properties:

::

  import pydons
  import numpy as np
  struct = pydons.MatStruct()
  struct['string'] = 'A string'
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

MatStruct can be serialized to HDF5 or Matlab files using 
``saveh5`` and ``savemat`` methods:

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

This software is distributed under the MIT license (see the LICENSE file).

Documentation
-------------

`pydons.readthedocs.org <http://pydons.readthedocs.org/>`_
