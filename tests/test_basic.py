from pydons import MatStruct
import numpy as np
import unittest
from pydons import _OrderedDict as OrderedDict


def test_fields():
    d = MatStruct()
    field_a = 'field a'
    field_b = (1, 'string', [3, 2.1])
    field_c = np.random.rand(3, 4)
    d.field_a = field_a
    d.field_b = field_b
    d.field_c = field_c
    assert d.field_a == field_a
    assert d.field_b == field_b
    assert np.all(d.field_c == field_c)


class TestKeys(unittest.TestCase):

    def test_keys(self):
        d = MatStruct()
        self.assertRaises(KeyError, d.__setitem__, 1, None)
        self.assertRaises(KeyError, d.__setitem__, '_a', None)

    def test_any_keys(self):
        d = MatStruct(any_keys=True)
        d[1] = 1
        d['_a'] = '_a'
        d.__setitem__(1, None)
        d.__setitem__('_a', None)
        del d[1]
        del d['_a']


def test_dedict():
    od = OrderedDict((('x', 1), ('y', 2.1)))
    od['z'] = OrderedDict((('x', 1), ('y', 2.1)))
    ms = MatStruct(od, dedict=True)
    assert isinstance(ms['z'], MatStruct)
    ms = MatStruct(od, dedict=False)
    assert isinstance(ms['z'], OrderedDict)
