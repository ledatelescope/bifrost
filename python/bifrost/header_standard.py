"""@package header_standard
This file enforces a standard header for rings.

Required parameters:

(parameter type definition)
nchans int "Number of frequency channels. 1+"
nifs int "Number of separate IF channels. 1+"
nbits int "Number of bits per value. 1+"
fch1 float "Center frequency of first channel given in buffer (MHz). >0"
foff float "Bandwidth of each channel (MHz). Negative values used for when first channel specified has the largest frequency."
tstart float "Time stamp in MJD of first sample (seconds). >0"
tsamp float "Time interval between samples (seconds). >0"

Optional parameters (which some blocks require):

"""

## Define a header which we can check passed
## dictionaries with
STANDARD_HEADER = {
    'nchans':(int, 1),
    'nifs':(int, 1, ),
    'nbits':(int, 1),
    'fch1':(float, 0),
    'foff':(float, None),
    'tstart':(float, 0),
    'tsamp':(float, 0)}

def enforce_header_standard(header_dict):
    """Raise an error if the header dictionary passed
        does not fit the standard specified above."""
    if type(header_dict) != dict:
        return False
    for parameter, standard in STANDARD_HEADER.items():
        if parameter not in header_dict:
            return False
        if type(header_dict[parameter]) != standard[0]:
            return False
        if standard[1] != None and \
            header_dict[parameter] < standard[1]:
            return False

    return True

