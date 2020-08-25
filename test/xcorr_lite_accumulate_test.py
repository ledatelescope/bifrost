import numpy as np
import bifrost as bf
from bifrost.libbifrost import _bf

def test_array(testvec):
    # Create test vectors
    a = bf.ndarray(np.array(testvec), dtype='i32', space='cuda')
    b = bf.ndarray(np.zeros_like(testvec, dtype='float32'), dtype='f32', space='cuda')

    # Run kernel
    _bf.XcorrLiteAccumulate(a.as_BFarray(), b.as_BFarray(), 0)

    # Copy back from GPU
    b_out = b.copy('system')
    b_out = np.array(b_out)
    a_sys = a.copy('system')
    a_sys = np.array(a_sys)

    assert np.allclose(a_sys.astype('float32'), b_out)

    # Run kernel in a loop
    for ii in range(0, 100):
        _bf.XcorrLiteAccumulate(a.as_BFarray(), b.as_BFarray(), 0)
        b_out = np.array(b.copy('system'))
        assert np.allclose(a_sys.astype('float32') * (ii + 2), b_out)

    # Test reset 
    reset = 1
    _bf.XcorrLiteAccumulate(a.as_BFarray(), b.as_BFarray(), np.int32(reset))
    b_out = np.array(b.copy('system'))

    assert  np.allclose(a_sys.astype('float32'), b_out)

if __name__ == "__main__":
    test_array([1,2,3,4])

    tv = np.arange(0, 1024*320*12*12).reshape((1024, 320, 12, 12))
    test_array(tv)
