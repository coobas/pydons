from pydons import MatStruct
import numpy as np
import os


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
