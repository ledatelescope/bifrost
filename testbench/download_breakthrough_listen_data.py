"""
# generate_test_data.py

Generate test data that can be used with a testbench
"""
import os
import numpy as np

bl_filelist = [
    'http://blpd0.ssl.berkeley.edu/blc2/dibas.20160304/AGBT16A_999_73/GUPPI/C/blc2_guppi_57451_62327_PSR_J0034-0721_0002.0013.raw',
    'http://blpd0.ssl.berkeley.edu/blc3/dibas.20160304/AGBT16A_999_73/GUPPI/D/blc3_guppi_57451_62327_PSR_J0034-0721_0002.0013.raw',
]

if __name__ == "__main__":
    
    if not os.path.exists('testdata'):
        os.mkdir('testdata')
    
    print "Downloading Breakthough Listen raw data"
    for filename in bl_filelist:
        bname = os.path.basename(filename)
        os.system("curl -O %s; mv %s testdata/" % (filename, bname))
   
        
