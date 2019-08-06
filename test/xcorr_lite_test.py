import numpy as np
import bifrost as bf
from bifrost.libbifrost import _bf

def compute_xcorr_cpu(d):
    dc = d.astype('float32').view('complex64')
    dc = dc.transpose((0,2,3,1)).copy()
    xcorr_cpu = np.einsum('...i,...j', dc,  np.conj(dc)).view('float32').astype('int32').sum(axis=-4)
    return xcorr_cpu

# Create complex data
N = 12
F = 2
T = 16

# Create complex data
d = np.random.randint(64, size=(F, N, T, 2), dtype='int8')
xcorr = np.zeros((F, N, N*2), dtype='int32')

d_gpu     = bf.ndarray(d, dtype='i8', space='cuda')
xcorr_gpu = bf.ndarray(xcorr, dtype='i32', space='cuda')

_bf.XcorrLite(d_gpu.as_BFarray(), xcorr_gpu.as_BFarray(), np.int32(N), np.int32(F), np.int32(T))

xcorr_gpu = np.array(xcorr_gpu.copy('system'))
xcorr_cpu = compute_xcorr_cpu(d)

assert np.allclose(xcorr_gpu.squeeze(), xcorr_cpu.squeeze())