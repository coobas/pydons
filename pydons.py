'''pydons = Python data manimulation add-ons
'''

from collections import OrderedDict as _OrderedDict
import numbers


class StrONGDict(_OrderedDict):
    """String key ordered dict Next Generation"""

    __FORBIDDEN_KEYS = tuple(dir(_OrderedDict) +
                             ['diff', 'save_h5', 'load_h5'])

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
        if any((not (ch.isalnum() or ch in '_-') for ch in key[1:])):
            raise KeyError('Keys must be of alphanumeric characters or _ or -')
        return True

    def __init__(self, values=None):
        super(StrONGDict, self).__init__()

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError('no attribute "{}"'.format(item))

    def __setitem__(self, item, value):
        if not item in self:
            self.__is_valid_key(item)
        super(StrONGDict, self).__setitem__(item, value)

    def __setattr__(self, item, value):
        if item in self:
            self.__setitem__(item, value)
        else:
            # the default behavior must be implemented
            self.__dict__[item] = value

    def __delattr__(self, item):
        if item in self:
            del self[item]
        else:
            raise AttributeError("no attribute '{}'".format(item))

    def __dir__(self):
        d = []
        d += self.keys()
        d += dir(self.__class__)
        d += self.__dict__.keys()
        d.sort()
        return d

    def __repr__(self):
        '''Custom StrONGDict __str__ for IPython
        '''
        if self:
            res = []
            fmt = '%%%ds: %%s' % (max(map(len, self.keys())))
            for k, v in self.items():
                res.append(fmt % (k, v))
            res = '{' + '\n '.join(res) + '}'
        else:
            res = '{}'
        return res

    def diff(self, other, mode='norm'):
        '''Find differences to another StrONGDict, ignoring the keys order

        :param other: StrONGDict object to compare to
        '''
        from numpy.linalg import norm
        from numpy import ndarray

        self_keys = set(self.keys())
        other_keys = set(other.keys())
        common_keys = self_keys & other_keys
        res = StrONGDict()
        nnorm = 0
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

    @classmethod
    def __save_dict_to_h5(cls, data, group):
        if not isinstance(data, (cls, dict)):
            raise ValueError('data must be a dict type')
        for k, v in data.items():
            if isinstance(v, (cls, dict)):
                cls.__save_dict_to_h5(v, group.create_group(str(k)))
            elif isinstance(v, (list, tuple)):
                raise ValueError('list and tuples are not supported (yet)')
            else:
                group.create_dataset(str(k), data=v)

    @classmethod
    def __get_dict_from_h5(cls, d, group):
        if not isinstance(d, (cls, dict)):
            raise ValueError('data must be a dict type')
        for k, v in group.items():
            if hasattr(v, 'items'):
                # this is a group
                d[k] = cls()
                cls.__get_dict_from_h5(d[k], v)
            else:
                if v.shape:
                    d[k] = v[:]
                else:
                    # empty shape = scalar
                    d[k] = v[()]

    def save_h5(self, filename, mode='w'):
        """Serialize to an HDF5 file

        :param filename: output file name
        :param mode: file open mode, typically 'w' or 'a'
        """
        from h5py import File
        with File(filename, mode) as fh5:
            self.__save_dict_to_h5(self, fh5)

    @classmethod
    def load_h5(cls, filename):
        """Deserialize from an HDF5 file (created by save_h5)

        :param filename: input file name
        """
        from h5py import File
        self = cls()
        with File(filename, 'r') as fh5:
            self.__get_dict_from_h5(self, fh5)
        return self
