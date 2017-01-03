"""@module correlator_pipeline
This program serves as an example of Bifrost's usage in a correlator pipeline, with kurtosis flagging.
"""
import numpy as np
import json, copy
from bifrost.block import Pipeline, MultiTransformBlock, SourceBlock, SinkBlock,  NumpyBlock
import bifrost.affinity as affinity


# When Python prints a type or turns it to string (for JSON), it gets wrapped in <type >
def str2type(s):
  if s == "<type 'numpy.float64'>": return np.float64
  if s == "<type 'numpy.complex64'>" or s == "complex64": return np.complex64

class Buffer(object):
    """ Allows the caller to add values to buffer in blocks of indeterminate size compared to the buffer size.
        If the buffer overflows the remainder is returned to the caller, who can add it later. 
        In this way the caller can eat up data.  """

    def __init__(self, size, typ):
       self.dtype = typ
       self.buffer = np.zeros(size, dtype=self.dtype)
       self.top = 0

    def full(self):
        return self.top == len(self.buffer)

    def clear(self):
        self.top = 0
        
    def append(self, values):
	# Cater for just one value
        if np.isscalar(values):
            if self.top == len(self.buffer): return values	# Can't do it
            else:
                self.buffer[self.top] = values
	        self.top += 1
		return None
        else:
          
            # Find out how much we can add in
            available_slots = len(self.buffer)-self.top
            if available_slots > len(values):
                available_slots = len(values)
            self.buffer[self.top:self.top+available_slots] = values[:available_slots]
            self.top += available_slots
            return values[available_slots:]		# remainder



class GenerateVoltages(SourceBlock):

    def __init__(self, out_length, how_many, core):
        super(GenerateVoltages, self).__init__()
	self.mu, self.sigma = 0.0, 1.0
        self.out_length = out_length
        self.how_many = how_many
        self.dtype = np.float64			# Type of the values return by random
        header = {
            'dtype': str(self.dtype),
            'shape': (out_length,)}
        self.output_header = json.dumps(header)
        #print "GenerateVoltages", self.output_header
        self.core = core

    def main(self, output_ring):
        affinity.set_core(self.core)
        i = 0 
        self.gulp_size = self.out_length*np.dtype(self.dtype).itemsize
        for ospan in self.iterate_ring_write(output_ring):
            ospan.data_view(self.dtype)[0][:] = np.random.normal(self.mu, self.sigma, self.out_length)
	    #print "V "+str(ospan.data_view(self.dtype)[0][:][0])
            if i == self.how_many: break
            i += 1


class PrintOp(SinkBlock):
 
    def __init__(self, core):
        super(PrintOp, self).__init__()
        self.core = core

    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.dtype = str2type(header["dtype"])
        self.gulp_size = header["shape"][0]*np.dtype(self.dtype).itemsize

    def main(self, input_ring):
        affinity.set_core(self.core)
        inspan_generator = span_generator = self.iterate_ring_read(input_ring)
        for inspan in inspan_generator:

            print "X "+str(inspan.data_view(self.dtype))



class Accumulator(MultiTransformBlock):
    """ Perform some operation on a chunk of data. The chunk size is independent of the incoming and outgoing
        gulp sizes. Because of this independence, data is accumulated in an internal buffer until the 
        required amount is obtained for the operation. Another internal output buffer is also maintained so that
        when enough data is available for output, it can be dumped into the output ring. The The input span size
        is fixed to the output span size of the previous block (as Bifrost does), the output span size and the
        buffer size for the operator are specified as parameters.
        """

    ring_names = {
        'in_data': "Ring containing arrays to be operated on",
        'out_data': "Outgoing ring containing the modified input data"}

    def __init__(self, func, op_required_length, out_length, out_dtype, core, args):            
        super(Accumulator, self).__init__()
        self.out_length = out_length
        self.core = core
        self.func = func
        self.op_required_length = op_required_length
        self.out_dtype = out_dtype
        self.args = args

    def load_settings(self):   
        """Calculate incoming/outgoing shapes and gulp sizes"""
        # Only update header if passed output header does not contain
        # parameters already

        self.header["out_data"] = copy.copy(self.header["in_data"])
        self.header["out_data"]["dtype"] = str(self.out_dtype)       # must make as str in header
        self.header["out_data"]["shape"] = (self.out_length,)
	self.in_dtype = str2type(self.header["in_data"]["dtype"])
        self.gulp_size["in_data"] = self.header["in_data"]["shape"][0]*np.dtype(self.in_dtype).itemsize
        self.gulp_size["out_data"] = self.out_length*np.dtype(self.out_dtype).itemsize
        self.out_buffer = Buffer(self.out_length, self.out_dtype)
        self.accumulator = Buffer(self.op_required_length, self.in_dtype)

    def main(self):
        """ Accumulate data, perform an operation 'func' on it, and write the data out """
	affinity.set_core(self.core)

        outspan_generator = self.write("out_data")

        for data in self.read("in_data"):	# Read spans from the ring
	    data = data[0]

	    remainder = self.accumulator.append(data)		# Accumulate. 

	    while self.accumulator.full():        # See if internal buffer full. In case span is longer than the buffer, we need to eat it up
 
		self.out_buffer.append(self.func(self.accumulator.buffer, self.args)); 	

		if self.out_buffer.full():     # Don't ever let overflow happen
                    outspan = outspan_generator.next()

                    outspan[0][:] = self.out_buffer.buffer[:] 

	            self.out_buffer.clear()
				    

	        self.accumulator.clear()
                remainder = self.accumulator.append(remainder)

# Define functions that also impose an accumulation of data. These get wrapped in an accumulator block.

def fft(data, args): 
  channel = args
  return (np.fft.fft(data)/len(data))[channel]

def correlate(data): return data*np.conj(data)

def integrate(data, args): return np.sum(data)
   

def kurtosis(data, args):
    fft_size, N = args
    M = len(data)
    data = 2*abs(data)/fft_size		# Data has been correlated, but these factors need to be added in
    S1 = np.sum(data)
    S2 = np.sum(data**2.0)	
    #Compute spectral kurtosis estimator
    Vk_square = (N*M+1)/(M-1)*(M*(S2/(S1**2.0))-1)	# eqn 8 http://mnrasl.oxfordjournals.org/content/406/1/L60.full.pdf
							# and eqn 50 https://web.njit.edu/~gary/assets/PASP_Published_652409.pdf
    return Vk_square

def kurtosis_variance(data, args):
  return np.var(data)

# Assemble the pipeline. Most blocks are accumulators, but some are not. There is a NumpyBlock (correlate).

blocks = []
voltage_chunk_size = 1024*1024; how_many_chunks = 1024*1024*1024; core = 0
blocks.append((GenerateVoltages(voltage_chunk_size, how_many_chunks, core), [], ["voltages"]))

fft_size = 1024; channel_values_buffer_size = 1024; which_channel = 10; core += 1
blocks.append((Accumulator(fft, fft_size, channel_values_buffer_size, np.complex64, core, (which_channel)), 
	{"in_data": "voltages", "out_data": "channel" }))

blocks.append((NumpyBlock(correlate), {'in_1': 'channel', 'out_1': 'auto_correlate'}))

integration_length = 1024; integrated_values_buffer_size = 1024; core += 1
blocks.append((Accumulator(integrate, integration_length, integrated_values_buffer_size, np.complex64, core, ()), 
	{"in_data": "auto_correlate", "out_data": "integrated" }))

M = integrated_values_buffer_size; N = integration_length; kurtosis_buffer_size = 1; core += 1
blocks.append((Accumulator(kurtosis, M, kurtosis_buffer_size, np.float64, core, (fft_size, N)), 
	{"in_data": "integrated", "out_data": "kurtosis" }))

variance_length = 1024; kurtosis_variance_buffer_size = 10; core += 1
blocks.append((Accumulator(kurtosis_variance, variance_length, kurtosis_variance_buffer_size, np.float64, core, ()), 
	{"in_data": "kurtosis", "out_data": "kurtosis_variance" }))

core += 1
blocks.append((PrintOp(core), ["kurtosis_variance"], []))



# Print what all the reductions give us. Find out how much data gets consumed to produce 1 variance value.
n = variance_length*M*integration_length*fft_size		# voltages
print "Num required voltage values", n
print "Reduction factor", ( "(%e)" % (float(1)/n) )
print "Threshold?", 4.0*M**2/((M-1)*(M+2)*(M+3))    # See eqn 53/54 in https://web.njit.edu/~gary/assets/PASP_Published_652409.pdf

if voltage_chunk_size*how_many_chunks < n:
  print "Error, there is not enough data to reduce to a variance value"
  exit(1)

# Run
Pipeline(blocks).main()



