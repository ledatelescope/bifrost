"""
# bf_into_op.py

Tests for + - / * operations that return the output value in place
(i.e. a += b, a -=b, a *= b, and a /= b)

"""

import numpy as np
from bifrost.libbifrost import _bf, _check
import bifrost as bf

np.set_printoptions(precision=2)
vec_len = int(1e8)

def gpu_add_into(a_arr, b_arr):
    """ Add array b into array a """
    _check(_bf.AddInto(a_arr, b_arr))

def gpu_multiply_into(a_arr, b_arr):
    """ Add array b into array a """
    _check(_bf.MultiplyInto(a_arr, b_arr))   

def gpu_subtract_into(a_arr, b_arr):
    """ Add array b into array a """
    _check(_bf.SubtractInto(a_arr, b_arr)) 

def gpu_divide_into(a_arr, b_arr):
    """ Add array b into array a """
    _check(_bf.DivideInto(a_arr, b_arr)) 


def generate_test_vectors(vec_len):
    a_cpu = np.sin(np.arange(vec_len, dtype='float32')) + 1
    b_cpu = np.arange(vec_len, dtype='float32') * 4 + 1
    
    a_gpu = bf.ndarray(a_cpu, space='cuda')
    b_gpu = bf.ndarray(b_cpu, space='cuda')
    return a_cpu, b_cpu, a_gpu, b_gpu
 
#######
## CHECK ADD
#######
a_cpu, b_cpu, a_gpu, b_gpu = generate_test_vectors(vec_len)

# Do sum in GPU
gpu_add_into(a_gpu.as_BFarray(), b_gpu.as_BFarray())
add_gpu = a_gpu.copy('system')

# Do sum in CPU
add_cpu = a_cpu + b_cpu

print "GPU ADD", add_gpu
print "CPU ADD", add_cpu

assert np.allclose(add_cpu, add_gpu)

#######
## CHECK SUBTRACT
#######
a_cpu, b_cpu, a_gpu, b_gpu = generate_test_vectors(vec_len)

# Do sum in GPU
gpu_subtract_into(a_gpu.as_BFarray(), b_gpu.as_BFarray())
sub_gpu = a_gpu.copy('system')

# Do sum in CPU
sub_cpu = a_cpu - b_cpu

print "GPU SUB", sub_gpu
print "CPU SUB", sub_cpu

assert np.allclose(sub_cpu, sub_gpu)

#######
## CHECK MULTIPLY
#######
a_cpu, b_cpu, a_gpu, b_gpu = generate_test_vectors(vec_len)

# Do sum in GPU
gpu_multiply_into(a_gpu.as_BFarray(), b_gpu.as_BFarray())
mult_gpu = a_gpu.copy('system')

# Do sum in CPU
mult_cpu = a_cpu * b_cpu

print "GPU MULT", mult_gpu
print "CPU MULT", mult_cpu

assert np.allclose(mult_cpu, mult_gpu)

#######
## CHECK DIVIDE
#######
a_cpu, b_cpu, a_gpu, b_gpu = generate_test_vectors(vec_len)

# Do sum in GPU
gpu_divide_into(a_gpu.as_BFarray(), b_gpu.as_BFarray())
div_gpu = a_gpu.copy('system')

# Do sum in CPU
div_cpu = a_cpu / b_cpu

print "GPU DIV", div_gpu
print "CPU DIV", div_cpu

assert np.allclose(div_cpu, div_gpu)