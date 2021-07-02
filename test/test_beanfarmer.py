import numpy as np
import bifrost as bf
from bifrost.libbifrost import _bf

#WARP_SIZE = 32
#NTHREADS = 1024
#NWARPS_PER_BLOCK = NTHREADS/WARP_SIZE

NANTENNAS = 12
NPOL = 1
NBEAMS = 32
NACCUMULATE = 64
NSAMPLES = 8192
NCHANNELS = 512


print("\nCreating test data...")
#    std::size_t aptf_voltages_size = NPOL * NSAMPLES * NANTENNAS * NCHANNELS;
np.random.seed(123)
v_re = np.random.randint(64, size=(1, NCHANNELS, NSAMPLES, NPOL, NANTENNAS), dtype='int8')
v_im = np.random.randint(64, size=(1, NCHANNELS, NSAMPLES, NPOL, NANTENNAS), dtype='int8')
v = np.zeros_like(v_re, dtype=[('re', 'int8'), ('im', 'int8')])
v['re'] = v_re
v['im'] = v_im

#    std::size_t apbf_weights_size  = NANTENNAS * NPOL * NBEAMS * NCHANNELS;
#w = np.random.randint(64, size=(NCHANNELS, NBEAMS, NPOL, NANTENNAS, 2), dtype='int8')
w_re = np.ones((NCHANNELS, NBEAMS, NPOL, NANTENNAS), dtype='int8')
w_im = np.ones((NCHANNELS, NBEAMS, NPOL, NANTENNAS), dtype='int8')
w = np.zeros_like(w_re, dtype=[('re', 'int8'), ('im', 'int8')])

for ii in range(NBEAMS):
    w_re[:, ii] = ii + 1
    w_im[:, ii] = ii + 1

w['re'] = w_re
w['im'] = w_im


#    std::size_t tbf_powers_size    = NSAMPLES/NACCUMULATE * NBEAMS * NCHANNELS;
b = np.zeros((NCHANNELS, NBEAMS, NSAMPLES/NACCUMULATE), dtype='float32')

print("\tVoltage data shape: {s}".format(s=str(v.shape)))
print("\tWeights shape:      {s}".format(s=str(w.shape)))
print("\tOutput beams shape: {s}".format(s=str(b.shape)))

print("\nCopying to GPU...")

v_bf = bf.ndarray(v, dtype='ci8', space='cuda')
w_bf = bf.ndarray(w, dtype='ci8', space='cuda')
b_bf = bf.ndarray(b, dtype='f32', space='cuda')

print("\tBF Voltage data shape: {s}".format(s=str(v_bf.shape)))
print("\tBF Weights shape:      {s}".format(s=str(w_bf.shape)))
print("\tBF Output beams shape: {s}".format(s=str(b_bf.shape)))

v_bf_0 = np.array(v_bf.copy('cuda_host'))

print("\nCalling BeanFarmer")
res = _bf.BeanFarmer(v_bf.as_BFarray(), w_bf.as_BFarray(), b_bf.as_BFarray(), np.int32(NACCUMULATE))

b_gpu = np.array(b_bf.copy('cuda_host'))

print("Checking output")
for ii in range(NBEAMS):
    assert np.allclose(b_gpu[:, ii] / b_gpu[:, 0], np.ones_like(b[:, 0]) * (ii+1)**2)

 # CPU gold version
v_cpu = v.astype('float32')
v_cpu = v_cpu.sum(axis=-2)                   # Sum over antennas
b_cpu = v_cpu[..., 0]**2 + v_cpu[..., 1]**2  # Square output
b_cpu = b_cpu.reshape((NCHANNELS,-1, NACCUMULATE)).sum(axis=-1)

