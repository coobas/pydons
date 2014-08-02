'''pydons = Python data manimulation add-ons
'''

from collections import OrderedDict as _OrderedDict
import numbers
import pydons.hdf5util
import hdf5storage


class MatStruct(_OrderedDict):
    """Matlab-like struct container

    Features:
    * Get and set fields as properties (obj.field)
    * String-only fields
    * Save and load to/from Matlab-compatible HDF5 files
    * __repr__ is customized for ipython visual output
    """

    __FORBIDDEN_KEYS = tuple(dir(_OrderedDict) +
                             ['diff', 'saveh5', 'loadh5',
                              'savemat', 'loadmat'])

    @classmethod
    def __is_valid_key(cls, key):
        '''Check if key is valid'''
        if not isinstance(key, (str, unicode)):
            raise KeyError('Keys can be only strings')
        if not key:
            raise KeyError('Key cannot be empty')
        if key in cls.__FORBIDDEN_KEYS:
            raise KeyError('"{}" conflicts with a method name'.format(key))
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
        self.__HDF5MARSHALLERS = hdf5storage.MarshallerCollection(
            [pydons.hdf5util.MatStructMarshaller(self.__class__)])

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError('no attribute "{}"'.format(item))

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
            raise AttributeError("no attribute '{}'".format(item))

    # insertion inspired by https://gist.github.com/jaredks/6276032
    def __insertion(self, link_prev, key_value):
        key, value = key_value
        if link_prev[2] != key:
            if key in self:
                del self[key]
            link_next = link_prev[1]
            self._OrderedDict__map[key] = link_prev[1] = link_next[0] = [link_prev, link_next, key]
        dict.__setitem__(self, key, value)

    def insert_after(self, existing_key, key_value):
        self.__insertion(self._OrderedDict__map[existing_key], key_value)

    def insert_before(self, existing_key, key_value):
        self.__insertion(self._OrderedDict__map[existing_key][0], key_value)

    def __dir__(self):
        d = []
        d += self.keys()
        d += dir(self.__class__)
        d += self.__dict__.keys()
        d.sort()
        return d

    def __repr__(self):
        '''Custom MatStruct __repr__ for IPython
        '''
        if self:
            res = []
            # adjust indentation, maximum 16 characters
            nspace = max((len(k) for k in self.keys()))
            nspace = nspace if nspace <= 16 else 0
            fmt = '%%%ds: %%s' % nspace
            for k, v in self.items():
                res.append(fmt % (k, v))
            res = '\n'.join(res)
        else:
            res = '{}.{}()'.format(self.__module__, self.__class__.__name__)
        return res

    def __str__(self):
        return repr(self)

    def diff(self, other, mode='norm'):
        '''Find differences to another MatStruct, ignoring the keys order

        :param other: MatStruct object to compare to
        '''
        from numpy.linalg import norm
        from numpy import ndarray

        self_keys = set(self.keys())
        other_keys = set(other.keys())
        common_keys = self_keys & other_keys
        res = MatStruct()
        nnorm = 0
        # TODO take care of nan's
        if mode == 'norm':
            if 'diff_norm' in self or 'diff_max' in self:
                raise Exception('diff_norm and diff_max cannot be in the compared dict')
            res['diff_norm'] = 0 + 0j
            res['diff_max'] = 0
        for key in self.keys():
            # common keys
            if key in common_keys:
                if isinstance(self[key], dict):
                    if isinstance(other[key], dict):
                        # this allows for standard dict
                        res[key] = self.__class__.diff(self[key], other[key])
                        res['diff_norm'] += res[key]['diff_norm']
                        res['diff_max'] = max(res['diff_max'], res[key]['diff_norm'].real)
                        nnorm += 1
                    else:
                        if mode == 'norm':
                            res[key] = 1j
                            res['diff_norm'] += res[key]
                        else:
                            res[key] = 'other["{}"] is {}, not {}'.format(key,
                                                                          type(other[key]),
                                                                          type(self[key]))
                elif isinstance(self[key], (numbers.Number, ndarray)):
                    try:
                        diff = norm(self[key] - other[key])
                    except Exception as e:
                        if mode == 'norm':
                            res[key] = 1j
                            res['diff_norm'] += res[key]
                        else:
                            res[key] = '{}'.format(e)
                    else:
                        self_norm = norm(self[key])
                        if self_norm < 1e-12:
                            self_norm = 1
                        else:
                            res[key] = diff / self_norm
                        res['diff_max'] = max(res['diff_max'], res[key].real)
                        res['diff_norm'] += res[key]
                        nnorm += 1
                elif isinstance(self[key], (str, unicode)):
                    try:
                        if self[key] == other[key]:
                            res[key] = 0
                            nnorm += 1
                        else:
                            res[key] = 1j
                        res['diff_norm'] += res[key]
                    except Exception as e:
                        if mode == 'norm':
                            res[key] = 1j
                            res['diff_norm'] += res[key]
                        else:
                            res[key] = '{}'.format(e)
                else:
                    raise NotImplementedError('Not implemented for {}'.format(type(self[key])))
            else:
                if mode == 'norm':
                    res[key] = 1j
                    res['diff_norm'] += res[key]
                else:
                    res[key] = '{}'.format(e)

        if mode == 'norm':
            if isinstance(res['diff_norm'], numbers.Complex):
                res['diff_norm'] = res['diff_norm'].real / nnorm + 1j * res['diff_norm'].imag
            else:
                res['diff_norm'] /= nnorm
        return res

    def saveh5(self, file_name, path='/', truncate_existing=False,
               matlab_compatible=False, **kwargs):
        """Save to an HDF5 file

        :param file_name: output file name
        :param path: group path to store fields to
        """
        hdf5storage.write(self, path, file_name, truncate_existing=truncate_existing,
                          marshaller_collection=self.__HDF5MARSHALLERS,
                          matlab_compatible=matlab_compatible)

    @classmethod
    def loadh5(cls, file_name, path='/', matlab_compatible=False, **kwargs):
        """Load from an HDF5 file

        :param file_name: file name
        :param path: path toread data from
        """
        # TODO how to preserve order?
        mc = hdf5storage.MarshallerCollection([pydons.hdf5util.MatStructMarshaller(cls)])
        return hdf5storage.read(path, file_name,
                                marshaller_collection=mc,
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
