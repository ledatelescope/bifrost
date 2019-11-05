#!/usr/bin/env python

import numpy
import bifrost
from bifrost.ndarray import asarray

from simple_functions import *

a = bifrost.ndarray([ 1, 2, 3, 4], dtype=numpy.float32, space='system')
b = bifrost.ndarray([-1,-2,-3,-4], dtype=numpy.float32, space='system')


add_stuff(a, b, a)
for i in range(4):
    assert(a[i] == 0)
    
subtract_stuff(a, b, a)
for i in range(4):
    assert(a[i] == i+1)
