'''pydons = Python numerical data manimulation add-ons
'''

__version__ = '0.1.1'

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
                              'diff', 'saveh5', 'loadh5',
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
        self.saveh5(file_name, path, truncate_existing=truncate_existing,
                    matlab_compatible=True, **kwargs)

    @classmethod
    def loadmat(cls, file_name, path='/', **kwargs):
        """Load from a Matlab (HDF5 format) file

        :param file_name: file name
        :param path: path toread data from
        """
        return cls.loadh5(file_name, path, matlab_compatible=True, **kwargs)
