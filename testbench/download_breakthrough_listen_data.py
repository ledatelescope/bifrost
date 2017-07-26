"""
# generate_test_data.py

Generate test data that can be used with a testbench
"""
import os
import numpy as np

raw_filelist = [
    'http://blpd0.ssl.berkeley.edu/blc2/dibas.20160304/AGBT16A_999_73/GUPPI/C/blc2_guppi_57451_62327_PSR_J0034-0721_0002.0013.raw',
    'http://blpd0.ssl.berkeley.edu/blc3/dibas.20160304/AGBT16A_999_73/GUPPI/D/blc3_guppi_57451_62327_PSR_J0034-0721_0002.0013.raw']

pulsar_filelist =[
    'https://storage.googleapis.com/pulsar_fil/blc0_guppi_57407_61054_PSR_J1840%2B5640_0004.fil',
    'https://storage.googleapis.com/pulsar_fil/blc0_guppi_57430_09693_PSR_J0826%2B2637_0003.fil',
    'https://storage.googleapis.com/pulsar_fil/blc0_guppi_57432_24865_PSR_J1136%2B1551_0002.fil'
]

voyager_filelist = ['https://storage.googleapis.com/gbt_fil/voyager_f1032192_t300_v2.fil']

if __name__ == "__main__":

    cont = raw_input("This will download approximately 5GB of data. Type Y to continue: ")

    if not cont.lower() == 'y':
        exit()

    if not os.path.exists('testdata'):
        os.mkdir('testdata')

    if not os.path.exists('testdata/pulsars'):
        os.mkdir('testdata/pulsars')    

    if not os.path.exists('testdata/voyager'):
        os.mkdir('testdata/voyager')   

    if not os.path.exists('testdata/guppi_raw'):
        os.mkdir('testdata/guppi_raw')    

    print "Downloading Breakthough Listen raw data"
    for filename in raw_filelist:
        bname = os.path.basename(filename)
        os.system("curl -O %s; mv %s testdata/guppi_raw/" % (filename, bname))

    print "Downloading Breakthough Listen pulsar data"
    for filename in pulsar_filelist:
        bname = os.path.basename(filename)
        os.system("curl -O %s; mv %s testdata/pulsars/" % (filename, bname))

    print "Downloading Breakthough Listen Voyager data"
    for filename in voyager_filelist:
        bname = os.path.basename(filename)
        os.system("curl -O %s; mv %s testdata/" % (filename, bname))


