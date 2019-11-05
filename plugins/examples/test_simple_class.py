#!/usr/bin/env python

import numpy
import bifrost
from bifrost.ndarray import asarray

from simple_class import *

a = bifrost.ndarray([ 1, 2, 3, 4], dtype=numpy.float32, space='system')
b = bifrost.ndarray([-1,-2,-3,-4], dtype=numpy.float32, space='system')


c = SimpleClass()
c.init(5.0)
c.execute(a, b)
for i in range(4):
    assert(b[i] == a[i] + 5.0)
