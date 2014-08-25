'''Pydons is a collection of manipulation add-ons for hierarchichal numerical data.
'''

__version__ = '0.2.1'

# import OrderedDict for Python < 2.7
# In Python 2.6 or Jython 2.5, ordereddict must be installed
for _collections in ('collections', 'ordereddict'):
    try:
        _collectionsModule = __import__(_collections)
        _OrderedDict = _collectionsModule.OrderedDict
    except:
        pass
try:
    _OrderedDict
    del _collectionsModule
except NameError:
    raise ImportError('No OrderedDict module found')
import numbers
import pydons.hdf5util
import hdf5storage
import numpy as np
import posixpath
import netCDF4
import h5py
import os
import sys


class MatStruct(_OrderedDict):
    """Matlab-like struct container

    Features:

    * Get and set fields as properties (obj.field = )
    * String-only fields
    * Save and load to/from Matlab-compatible HDF5 files
    * Ipython customized output
    """

    __FORBIDDEN_KEYS = tuple(dir(_OrderedDict) +
                             ['insert_after', 'insert_before',
                              'diff', 'merge', 'saveh5', 'loadh5',
                              'savemat', 'loadmat'])
    __MC = None

    @classmethod
    def __mc(cls):
        if cls.__MC is None:
            cls.__MC = hdf5storage.MarshallerCollection([pydons.hdf5util.MatStructMarshaller(cls)])
        return cls.__MC

    @classmethod
    def __is_valid_key(cls, key):
        '''Check if key is valid'''
        if not isinstance(key, (str, unicode)):
            raise KeyError('Keys can be only strings')
        if not key:
            raise KeyError('Key cannot be empty')
        if key in cls.__FORBIDDEN_KEYS:
            raise KeyError('"%s" conflicts with a method name' % (key))
        if not key[0].isalpha():
            raise KeyError('Key does not start with an alphabetic character')
        if any((not (ch.isalnum() or ch in '_') for ch in key[1:])):
            raise KeyError('Keys must be of alphanumeric characters or _')
        try:
            key.decode('ascii')
        except (UnicodeEncodeError, UnicodeDecodeError):
            raise KeyError('Keys must be ascii characters only')
        return True

    def __init__(self, *args, **kwargs):
        # hiding attributes via __dir__ does not seem to work in ipython
        # self._hide_methods = kwargs.pop('hide_methods', False)
        super(MatStruct, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError('no attribute "%s"' % item)

    def __setitem__(self, item, value):
        if not item in self:
            self.__is_valid_key(item)
        super(MatStruct, self).__setitem__(item, value)

    def __setattr__(self, item, value):
        if item.startswith('_'):
            # the default behavior must be implemented
            self.__dict__[item] = value
        else:
            self.__setitem__(item, value)

    def __delattr__(self, item):
        if item in self:
            del self[item]
        else:
            raise AttributeError("no attribute '%s'" % (item))

    # insertion inspired by https://gist.github.com/jaredks/6276032
    def __insertion(self, link_prev, key, value):
        '''Internal insertion method
        '''
        if link_prev[2] != key:
            if key in self:
                del self[key]
            link_next = link_prev[1]
            self._OrderedDict__map[key] = link_prev[1] = link_next[0] = [link_prev, link_next, key]
        dict.__setitem__(self, key, value)

    def insert_after(self, existing_key, key, value):
        '''Insert after an existing field

        :param existing_key: existing key
        :param key: new key
        :param value: inserted value
        '''
        self.__insertion(self._OrderedDict__map[existing_key], key, value)

    def insert_before(self, existing_key, key, value):
        '''Insert before an existing field

        :param existing_key: existing key
        :param key: new key
        :param value: inserted value
        '''
        self.__insertion(self._OrderedDict__map[existing_key][0], key, value)

    def __dir__(self):
        d = []
        d += self.keys()
        # if not self._hide_methods:
        d += dir(self.__class__)
        d += self.__dict__.keys()
        d.sort()
        return d

    def _repr_pretty_(self, p, cycle):
        '''Pretty representation for Ipython
        '''
        if cycle:
            p.text('%s(...)' % (self.__class__.__name__))
        else:
            if self:
                # adjust indentation, maximum 16 characters
                nspace = max((len(k) for k in self.keys()))
                nspace = nspace if nspace <= 16 else 0
                fmt = '%%%ds: %%s' % nspace
                # with p.group(0, 'MatStruct({', '}'):
                keys = list(self.keys())
                keys, lastkey = keys[:-1], keys[-1]
                for k in keys:
                    p.text(fmt % (k, self[k]))
                    p.break_()
                else:
                    p.text(fmt % (lastkey, self[lastkey]))
            else:
                p.text('%s()' % (self.__class__.__name__))

    def diff(self, other, **kwargs):
        '''Find numerical differences to another MatStruct, ignoring the keys order

        Returns a structure with diff_norm = average norm of all numerical differences,
        diff_max = maximum of norm differences,
        diff_uncomparable = number of uncomparable fields.

        :param other: MatStruct object to compare to

        Keyword arguments

        :param norm: norm function, default is numpy.linalg.norm
        :param rel_norm_thold: relative difference threshold above which relative difference is normalized by the norm of the field value
        '''

        norm = kwargs.get('norm', np.linalg.norm)
        rel_norm_thold = kwargs.get('rel_norm_thold', 1e-12)

        self_keys = set(self.keys())
        other_keys = set(other.keys())
        common_keys = self_keys & other_keys
        if set(('diff_norm', 'diff_max', 'diff_uncomparable')) & (self_keys | other_keys):
            raise KeyError("cannot calculate diff if "
                           "'diff_norm', 'diff_max' or 'diff_uncomparable'"
                           "is in the keys of the compared objects")
        res = MatStruct()
        nnorm = 0
        # TODO take care of nan's
        res['diff_norm'] = 0
        res['diff_max'] = 0
        res['diff_uncomparable'] = 0
        for key in self.keys():
            # common keys
            if key in common_keys:
                if isinstance(self[key], self.__class__):
                    if isinstance(other[key], dict):
                        # this allows for standard dict
                        res[key] = self.diff(other[key])
                        res['diff_norm'] += res[key]['diff_norm']
                        res['diff_max'] = max(res['diff_max'], res[key]['diff_norm'])
                        nnorm += 1
                    else:
                        res['diff_uncomparable'] += 1
                        res[key] = ('type(other["%s"]) is %s, not %s' %
                                    (key, type(other[key]), type(self[key])))
                elif isinstance(self[key], (numbers.Number, np.ndarray, np.generic)):
                    try:
                        diff = norm(self[key] - other[key])
                    except Exception as e:
                        res['diff_uncomparable'] += 1
                        res[key] = '%s' % e
                    else:
                        self_norm = norm(self[key])
                        res[key] = diff
                        if self_norm > rel_norm_thold:
                            res[key] /= self_norm
                        res['diff_max'] = max(res['diff_max'], res[key])
                        res['diff_norm'] += res[key]
                        nnorm += 1
                elif isinstance(self[key], (str, unicode)):
                    try:
                        if self[key] == str(other[key]):
                            res[key] = 0
                            nnorm += 1
                        else:
                            res[key] = 'string are not equal'
                            res['diff_uncomparable'] += 1
                    except Exception as e:
                        res['diff_uncomparable'] += 1
                        res[key] = '%s' % e
                else:
                    raise NotImplementedError('Not implemented for %s' % (type(self[key])))
            else:
                res['diff_uncomparable'] += 1
                res[key] = '%s not in other' % key
        # the result is an average of the sum of the diff norms
        res['diff_norm'] /= nnorm
        return res

    def merge(self, other):
        '''Merge fields form another MatStruct or any dict-like object

        :param other: object to merge from
        '''
        # TODO
        raise NotImplementedError('to be implemented')

    def saveh5(self, file_name, path='/', truncate_existing=False,
               matlab_compatible=False, **kwargs):
        """Save to an HDF5 file

        :param file_name: output file name
        :param path: group path to store fields to
        """
        hdf5storage.write(self, path, file_name, truncate_existing=truncate_existing,
                          marshaller_collection=self.__mc(),
                          matlab_compatible=matlab_compatible)

    @classmethod
    def loadh5(cls, file_name, path='/', matlab_compatible=False, **kwargs):
        """Load from an HDF5 file

        :param file_name: file name
        :param path: path toread data from
        """
        # TODO how to preserve order?
        return hdf5storage.read(path, file_name,
                                marshaller_collection=cls.__mc(),
                                matlab_compatible=matlab_compatible)

    def savemat(self, file_name, path='/', truncate_existing=False, **kwargs):
        """Save to a Matlab (HDF5 format) file

        :param file_name: output file name
        :param path: group path to store fields to
        """
        # TODO switch convert ints to doubles
        self.saveh5(file_name, path, truncate_existing=truncate_existing,
                    matlab_compatible=True, **kwargs)

    @classmethod
    def loadmat(cls, file_name, path='/', **kwargs):
        """Load from a Matlab (HDF5 format) file

        :param file_name: file name
        :param path: path toread data from
        """
        return cls.loadh5(file_name, path, matlab_compatible=True, **kwargs)


class NC4File(netCDF4.Dataset):
    """NetCDF 4 file with __getitem__"""
    def __init__(self, *args, **argv):
        super(NC4File, self).__init__(*args, **argv)

    def __getitem__(self, key):
        '''Get item from a key specified as a posix path'''
        grp = self
        # remove leading /
        while key.startswith('/'):
            key = key[1:]
        if not key:
            # get the root
            return self
        key = posixpath.normpath(key)
        keys = key.split('/')
        grps, var = keys[:-1], keys[-1]
        # get the final group
        for k in grps:
            grp = grp.groups[k]
        # get the variable or group
        if var in grp.variables:
            return grp.variables[var]
        elif var in grp.groups:
            return grp.groups[var]
        else:
            raise KeyError('%s not found' % key)


class LazyDataset(object):
    """NetCDF 4 / HDF5 data set object with lazy evaluation"""
    def __init__(self, grp, name, lazy_min_size=100, lazy_max_size=100000000):
        if isinstance(grp, (netCDF4.Group, netCDF4.Dataset)):
            self._fileclass = NC4File
            self._filepath = os.path.abspath(grp.filepath())
            self._path = posixpath.join(grp.path, name)
            dset = grp.variables[name]
        elif isinstance(grp, h5py.Group):
            self._fileclass = h5py.File
            self._filepath = os.path.abspath(grp.file.filename)
            self._path = posixpath.join(grp.name, name)
            self._data = None
            dset = grp[name]
        else:
            raise TypeError('%s not supported' % type(grp))
        self._data = None
        self._lazy_min_size = lazy_min_size
        self._lazy_max_size = lazy_max_size
        # preloaded attributes
        for prop in ('dtype', 'shape', 'size', 'ndim', 'dimensions', 'title', 'units'):
            if hasattr(dset, prop):
                setattr(self, prop, getattr(dset, prop))
        if self.size <= lazy_min_size:
            self._get_data()

    def _get_data(self, key=None):
        if self._data is None:
            with self._fileclass(self._filepath, 'r') as f:
                if self.size <= self._lazy_max_size:
                    self._data = f[self._path][:]
                    if key is None:
                        return self._data
                    else:
                        return self._data[key]
                else:
                    return f[self._path][key]
        else:
            if key is None:
                return self._data
            else:
                return self._data[key]

    def __getitem__(self, key):
        """Get slice (read rada)"""
        return self._get_data(key)

    def __getattr__(self, attr):
        if self._data:
            return getattr(self._data, attr)
        else:
            raise AttributeError('no attribute %s' % attr)
            # return getattr(self._fileclass(self._filepath, 'r')[self._path], attr)

    def __iter__(self):
        # iterate over data
        return iter(self._get_data())

    def __array__(self):
        # for numpy
        return self._get_data()

    if sys.version_info < (2, 0):
        # They won't be defined if version is at least 2.0 final
        def __getslice__(self, i, j):
            return self[max(0, i):max(0, j):]


class FileBrowser(MatStruct):
    """Load netCDF of HDF5 file into an offline MatStruct"""
    def __init__(self, file_name, lazy_min_size=100, lazy_max_size=100000000):
        """Read hierarchical data file into a MatStruct tree with data in LazyDataset

        :param file_name: file name
        :param lazy_min_size: data sets with a lower size will be always stored in the memory
        :param lazy_max_size: data sets with a larger size will never be stored in the memory
        """
        super(FileBrowser, self).__init__()
        # get the file type and corresponding classes
        _, ext = os.path.splitext(file_name)
        ext = ext[1:]
        if ext.lower() in ('nc', 'cdf'):
            fileclass, dataclass = NC4File, LazyDataset
        elif ext.lower() in ('h5', 'hdf5', 'he5'):
            fileclass, dataclass = h5py.File, LazyDataset
        else:
            raise TypeError('Unknow file extension: %s' % ext)
        # recursively read the file structure
        with fileclass(file_name, 'r') as fileobj:
            for key, val in _read_all(fileobj, dataclass,
                                      lazy_min_size=lazy_min_size,
                                      lazy_max_size=lazy_max_size).items():
                self[key] = val


def _read_all(basegrp, dataclass, lazy_min_size, lazy_max_size):
    """Recursively read all groups / variables

    :param basegrp: base group (the starting point)
    :param dataclass: data type to return
    """
    res = MatStruct()
    for grpname in _groups(basegrp):
        try:
            res[grpname] = _read_all(_groups(basegrp)[grpname], dataclass,
                                     lazy_min_size, lazy_max_size)
        except KeyError:
            res[grpname + '_'] = _read_all(_groups(basegrp)[grpname], dataclass,
                                           lazy_min_size, lazy_max_size)
    for varname in _variables(basegrp):
        try:
            res[varname] = dataclass(basegrp, varname, lazy_min_size, lazy_max_size)
        except KeyError:
            res[varname + '_'] = dataclass(basegrp, varname, lazy_min_size, lazy_max_size)
    return res


def _groups(basegrp):
    """Get group names from an HDF5/netCDF4 file group
    """
    if hasattr(basegrp, 'groups'):
        return basegrp.groups
    else:
        # for HDF5 return keys that are dict-like
        # return [key for key in basegrp.keys() if hasattr(basegrp[key], 'keys')]
        return _OrderedDict([(key, value) for key, value in basegrp.items() if hasattr(value, 'keys')])


def _variables(basegrp):
    """Get group names from an HDF5/netCDF4 file group
    """
    if hasattr(basegrp, 'variables'):
        return basegrp.variables
    else:
        # for HDF5 return keys that are not dict-like
        return _OrderedDict([(key, value) for key, value in basegrp.items() if not hasattr(value, 'keys')])
