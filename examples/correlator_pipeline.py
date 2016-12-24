"""@module correlator_pipeline
This program serves as an example of Bifrost's usage in a correlator pipeline
"""
import numpy as np
import json
from bifrost.block import Pipeline


SIZEOF_FLOAT32 = 4
SIZEOF_FLOAT64 = 8
SIZEOF_COMPLEX = 8

class Buffer(object):
    """ Allows the caller to add values in blocks of indeterminate size compared to the buffer size.
        If the buffer overflows the remainder is returned to the caller, who can add it later. 
        In this way the caller can eat up a large buffer  """

    def __init__(self, size, typ):
        self.buffer = np.zeros(size, dtype=typ)
        self.typ = typ
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
        else:
          
            # Find out how much we can add in
            available_slots = len(self.buffer)-self.top
            if available_slots > len(values):
                available_slots = len(values)
            self.buffer[self.top:self.top+available_slots] = values[:available_slots]
            self.top += available_slots
            return values[available_slots:]		# remainder



class GenerateVoltages(object):
    """Generate Gaussian random values being the voltage in 1 stand, 1 polarization"""

    def __init__(self, out_length):
        self.out_length = out_length	# Make sure multiple of FFT size
        self.mu, self.sigma = 0.0, 1.0
        self.oring_span_size = self.out_length*SIZEOF_FLOAT64;		# because the value is float64

    def main(self, input_rings, output_rings):
        ohdr = json.dumps({ "LENGTH" : self.out_length })
        output_rings[0].resize(self.oring_span_size)
        with output_rings[0].begin_writing() as oring:
            with oring.begin_sequence("Voltages", header=ohdr) as osequence:
                for i in range(4):
                    with osequence.reserve(self.oring_span_size) as wspan:
			voltages = np.random.normal(self.mu, self.sigma, self.out_length) 
			#print "V Out", voltages[0]
			wspan.data[0][:] = voltages.view(dtype=np.uint8)
                        wspan.commit(self.oring_span_size)

class FFTVoltages(object):
    """Perform FFTs on voltages and send out one channel i.e. one bin value.
       Chunk the voltages so the FFT is of size FFT_SIZE  """

    def __init__(self, out_length, fft_size, channel):      
        self.out_length = out_length
        self.out_buffer = Buffer(out_length, np.complex64)
        self.fft_size = fft_size
        self.fft_buffer = Buffer(fft_size, np.float64)
        self.channel = channel

    def main(self, input_rings, output_rings):
        ohdr = json.dumps({ "LENGTH" : self.out_length })
        oring_span_size = self.out_length*SIZEOF_COMPLEX
        output_rings[0].resize(oring_span_size)

        with output_rings[0].begin_writing() as oring:
            with oring.begin_sequence("Channel", header=ohdr) as osequence:
                
                for iseq in input_rings[0].read():
                    ihdr = json.loads(iseq.header.tostring())
	            iring_span_size = ihdr["LENGTH"]*SIZEOF_FLOAT64
                    input_rings[0].resize(iring_span_size)
                    for ispan in iseq.read(iring_span_size):
	                data = ispan.data.view(np.float64)[0]

			remainder = self.fft_buffer.append(data)
			while self.fft_buffer.full():
			    print "Do fft", len(remainder)
	    	            FFT = np.fft.fft(self.fft_buffer.buffer)
	                    FFT /= self.fft_size

			    self.out_buffer.append(FFT[self.channel])	

		            if self.out_buffer.full():     # Don't ever let overflow happen
			        with osequence.reserve(oring_span_size) as wspan:
				    #print "Ch Out", channel_value_by_time[0] 
			            wspan.data[0][:] = self.out_buffer.buffer.view(dtype=np.uint8)
                        	    wspan.commit(oring_span_size)
			        self.out_buffer.clear()
				    

			    self.fft_buffer.clear()

			    remainder = self.fft_buffer.append(remainder)


class Correlate(object):
    """Perform FFTs on voltages and send out one channel i.e. one bin value.
       Chunk the voltages so the FFT is of size FFT_SIZE  """
    def __init__(self, out_length):      
        self.out_length = out_length

    def main(self, input_rings, output_rings):

        ohdr = json.dumps({ "LENGTH" : self.out_length })
        oring_span_size = self.out_length*SIZEOF_COMPLEX		# Complex numbers
        output_rings[0].resize(oring_span_size)
        correlated_value_by_time = np.zeros(0, dtype=np.complex64)	 # output accumulation buffer

        with output_rings[0].begin_writing() as oring:
            with oring.begin_sequence("Visibility", header=ohdr) as osequence:
                
                for iseq in input_rings[0].read():
                    ihdr = json.loads(iseq.header.tostring())
	            iring_span_size = ihdr["LENGTH"]*SIZEOF_COMPLEX
                    input_rings[0].resize(iring_span_size)
                    for ispan in iseq.read(iring_span_size):
	                data = ispan.data.view(np.complex64)[0]
		                          
		        #print "Num FFTs", num_FFTs, "Ch In", data[0] 
		
        	        data = data**2	    # need to buffer

			correlated_value_by_time = np.append(correlated_value_by_time, data)
			if len(correlated_value_by_time) > self.out_length:
	    	            
			    with osequence.reserve(oring_span_size) as wspan:
			        #print "Corr Out", data[0] 
			        wspan.data[0][:] = correlated_value_by_time[:self.out_length].view(dtype=np.uint8)
                                wspan.commit(oring_span_size)
				correlated_value_by_time = correlated_value_by_time[self.out_length:]

				
class Integrate(object):
    """Perform FFTs on voltages and send out one channel i.e. one bin value.
       Chunk the voltages so the FFT is of size FFT_SIZE  """

    def __init__(self, out_length, integration_length):      
        self.out_length = out_length
        self.integration_length = integration_length

    def main(self, input_rings, output_rings):
        ohdr = json.dumps({ "LENGTH" : self.out_length })
        oring_span_size = self.out_length*SIZEOF_COMPLEX
        output_rings[0].resize(oring_span_size)
        channel_value_by_integration = np.zeros(self.out_length, dtype=np.complex64)	 # output accumulation buffer
        ch_index = 0

        with output_rings[0].begin_writing() as oring:
            with oring.begin_sequence("Integrated Channel", header=ohdr) as osequence:
                
                for iseq in input_rings[0].read():
                    ihdr = json.loads(iseq.header.tostring())
	            iring_span_size = ihdr["LENGTH"]*SIZEOF_COMPLEX
                    input_rings[0].resize(iring_span_size)
                    for ispan in iseq.read(iring_span_size):
	                data = ispan.data.view(np.complex64)[0]
		        
                        num_integrations = len(data)/self.integration_length			# Assuming data is a multiple of integration length. 
		        #print "Num FFTs", num_FFTs, "V In", data[0] 
		
        	        for i, chunk in enumerate(np.split(data, num_integrations)):

                            channel_value_by_integration[ch_index] = np.sum(chunk)
			    ch_index += 1

		            if ch_index == self.out_length:
			        with osequence.reserve(oring_span_size) as wspan:
				    #print "Ch Out", channel_value_by_integration[0] 
			            wspan.data[0][:] = channel_value_by_integration.view(dtype=np.uint8)
                        	    wspan.commit(oring_span_size)

				ch_index = 0


class Kurtosis(object):
    """Perform FFTs on voltages and send out one channel i.e. one bin value.
       Chunk the voltages so the FFT is of size FFT_SIZE  """

    def __init__(self, out_length, fft_size, N, M):   
        self.out_length = out_length
        self.fft_size = fft_size
        self.M = M
        self.N = N

    def main(self, input_rings, output_rings):
        ohdr = json.dumps({ "LENGTH" : self.out_length })
        oring_span_size = self.out_length*SIZEOF_FLOAT32
        output_rings[0].resize(oring_span_size)
        kurtosis = np.zeros(self.out_length, dtype=np.float32)	 	# output accumulation buffer
        ch_index = 0

        with output_rings[0].begin_writing() as oring:
            with oring.begin_sequence("Kurtosis", header=ohdr) as osequence:
                
                for iseq in input_rings[0].read():
                    ihdr = json.loads(iseq.header.tostring())
	            iring_span_size = ihdr["LENGTH"]*SIZEOF_COMPLEX
                    input_rings[0].resize(iring_span_size)
                    for ispan in iseq.read(iring_span_size):
	                data = ispan.data.view(np.complex64)[0]
			data = 2*abs(data)/self.fft_size	# Data has been correlated, but these factors need to be added in, and turned to PSD
		    
                        num_Ms = len(data)/self.M			# Assuming data is a multiple of M length. 
		        #print "Num Ms", num_Ms, "I In", data[0] 
		
        	        for i, chunk in enumerate(np.split(data, num_Ms)):
			    S1 = np.sum(chunk)
			    S2 = np.sum(chunk**2.0)

		
				#Compute spectral kurtosis estimator
			    Vk_square = (self.N*self.M+1)/(self.M-1)*(self.M*(S2/(S1**2.0))-1)
			    kurtosis[ch_index] = Vk_square
			    ch_index += 1

		            if ch_index == self.out_length:
			        with osequence.reserve(oring_span_size) as wspan:
				    #print "K Out", kurtosis[0] 
			            wspan.data[0][:] = kurtosis.view(dtype=np.uint8)
                        	    wspan.commit(oring_span_size)

				ch_index = 0
  
class PrintOp(object):
    def main(self, input_rings, output_rings):
        num_received = 0
 
	for iseq in input_rings[0].read():
            ihdr = json.loads(iseq.header.tostring())
	    ring_span_size = ihdr["LENGTH"]*SIZEOF_FLOAT32
            input_rings[0].resize(ring_span_size)
	    for ispan in iseq.read(ring_span_size):
		data = ispan.data.view(np.float32)[0]
		#print "K In", data[0] 	
		num_received += len(data)

        print "Total received", num_received

# Lots of reductions happen
voltage_out_buf_length = 1024*1024
# FFT performs a reduction. For each FFT, only 1 value comes out - the FFT component for one channel
fft_size = 1024
fft_out_buf_length = 1024
# Correlation does not do a reduction
correlate_out_buf_length = 1024
# Integration does a reduction
integrate_out_buf_len = 10
integ_length = 32

blocks = []
blocks.append((GenerateVoltages(voltage_out_buf_length), [], [0]))
blocks.append((FFTVoltages(fft_size, fft_out_buf_length, 20), [0], [1]))
blocks.append((Correlate(correlate_out_buf_length), [1], [2]))
blocks.append((Integrate(integrate_out_buf_len, integ_length), [2], [3]))
blocks.append((Kurtosis(2, 1024, 10, 10), [3], [4]))
blocks.append((PrintOp(), [4], []))
Pipeline(blocks).main()
exit()



