from pydons import MatStruct
import numpy as np
import random

MAX_INT = 1000000
MAX_FLOAT = 1e9
REL_THRESH = 1e-12


def sgn():
    return random.randint(0, 1) * 2 - 1


def randint(abs_min=1):
    return random.randint(abs_min, MAX_INT) * sgn()


def randfloat(abs_min=REL_THRESH, abs_max=MAX_FLOAT):
    return (random.random() + abs_min) * abs_max * sgn()


def test_diff_self():
    d = MatStruct()

    d.field_a = 'field a'
    d.field_b = 10
    d.field_c = np.random.rand(3, 4)

    ddiff = d.diff(d)

    assert ddiff.diff_norm == 0
    assert ddiff.diff_max == 0
    assert ddiff.diff_uncomparable == 0


def test_diff_numbers():
    d = MatStruct()
    d.int = randint(1)
    d.float = randfloat()

    dd = MatStruct()
    dd.int = randint(1)
    dd.float = randfloat()

    ddiff = d.diff(dd, rel_norm_thold=REL_THRESH)

    assert ddiff.int == abs(float(d.int - dd.int)) / abs(d.int)
    assert ddiff.float == abs(d.float - dd.float) / abs(d.float)
    assert ddiff.diff_norm == (ddiff.int + ddiff.float) / 2
    assert ddiff.diff_max == max(ddiff.int, ddiff.float)
    assert ddiff.diff_uncomparable == 0


def test_diff_small_numbers():
    d = MatStruct()
    d.int = 0
    d.float = randfloat(0, 0.9 * REL_THRESH)

    dd = MatStruct()
    dd.int = randint(1)
    dd.float = randfloat()

    ddiff = d.diff(dd)

    assert ddiff.int == abs(float(d.int - dd.int))
    assert ddiff.float == abs(d.float - dd.float)
    assert ddiff.diff_norm == (ddiff.int + ddiff.float) / 2
    assert ddiff.diff_max == max(ddiff.int, ddiff.float)
    assert ddiff.diff_uncomparable == 0
