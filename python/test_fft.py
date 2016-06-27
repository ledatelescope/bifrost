from bifrost.GPUArray import GPUArray
from bifrost.ring import Ring
from bifrost.fft import fft
from bifrost.libbifrost import _bf, _string2space
import numpy as np
import ctypes


BF_MAX_DIM = 3

class BFarray(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("space", _bf.BFspace),
        ("dtype", ctypes.c_uint),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_ulong*BF_MAX_DIM),
        ("strides", ctypes.c_ulong*BF_MAX_DIM)]

np.random.seed(4)
a = GPUArray(shape=(10), dtype=np.float32)
a.set(np.arange(10))
data = ctypes.cast(a.ctypes.data, ctypes.c_void_p)
space = _string2space('cuda')
c = (ctypes.c_ulong*BF_MAX_DIM)(*[10,0,0])
d = (ctypes.c_ulong*BF_MAX_DIM)(*[4*8,0,0])
myarray = BFarray(data,space,1,1,c,d)
print _bf.BFarray
#mybfarray = ctypes.cast(myarray,_bf.BFarray)
print fft(myarray, myarray)

