
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import bifrost as bf
from bifrost.ring2 import Ring, ring_view
from temp_storage import TempStorage

from collections import defaultdict
from contextlib2 import ExitStack
import threading
import time
from copy import copy

def izip(*iterables):
	while True:
		yield [it.next() for it in iterables]

thread_local = threading.local()
thread_local.pipeline_stack = []
def get_default_pipeline():
	return thread_local.pipeline_stack[-1]

thread_local.blockscope_stack = []
def get_current_block_scope():
	if len(thread_local.blockscope_stack):
		return thread_local.blockscope_stack[-1]
	else:
		return None

def block_scope(*args, **kwargs):
	return BlockScope(*args, **kwargs)

class BlockScope(object):
	instance_count = 0
	def __init__(self,
	             name=None,
	             gulp_nframe=None,
	             buffer_nframe=None,
	             buffer_factor=None,
	             core=None,
	             gpu=None,
	             share_temp_storage=False,
	             fuse=False):
		if name is None:
			name = 'BlockScope_%i' % BlockScope.instance_count
			BlockScope.instance_count += 1
		self._name = name
		self._gulp_nframe   = gulp_nframe
		self._buffer_nframe = buffer_nframe
		self._buffer_factor = buffer_factor
		self._core          = core
		self._gpu           = gpu
		self._share_temp_storage = share_temp_storage
		self._temp_storage_ = {}
		self._fused = fuse
		if fuse:
			#if self._buffer_factor is None:
			#	self._buffer_factor = 1.0
			if self._share_temp_storage is None:
				self._share_temp_storage = True
		self._parent_scope = get_current_block_scope()
		if self._parent_scope is not None:
			self._parent_scope.children.append(self)
		self._children = []
	def __enter__(self):
		thread_local.blockscope_stack.append(self)
	def __exit__(self, type, value, tb):
		assert(thread_local.blockscope_stack.pop() is self)
	def __getattr__(self, name):
		# Use child's value if set, othersize defer to parent
		if not hasattr(self, '_'+name):
			raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__,name))
		self_value = getattr(self, '_'+name)
		if self_value is not None:
			return self_value
		else:
			if self._parent_scope is not None:
				return getattr(self._parent_scope, name)
			else:
				return None
	def _get_temp_storage(self, space):
		if space not in self._temp_storage_:
			self._temp_storage_[space] = TempStorage(space)
		return self._temp_storage_[space]
	def _get_scope_hierarchy(self):
		"""Returns list of BlockScopes from root ancestor to self"""
		scope_hierarchy = []
		parent = self._parent_scope
		while parent is not None:
			scope_hierarchy.append(parent)
			parent = parent._parent_scope
		return reversed(scope_hierarchy)
	def cache_scope_hierarchy(self):
		self.scope_hierarchy = self._get_scope_hierarchy()
		self.fused_ancestor = None
		for ancestor in self.scope_hierarchy:
			if ancestor._fused:
				self.fused_ancestor = ancestor
				break
	def is_fused_with(self, other):
		return (self.fused_ancestor is not None and
		        self.fused_ancestor is other.fused_ancestor)
	def get_temp_storage(self, space):
		# TODO: Cache the first share_temp_storage scope to avoid walking each time
		for scope in self.scope_hierarchy:#self.get_scope_hierarchy():
			if scope.share_temp_storage:
				return scope._get_temp_storage(space)
		return self._get_temp_storage(space)
	def dot_graph(self, parent_graph=None):
		from graphviz import Digraph
		
		g = Digraph('cluster_'+self._name) if parent_graph is None else \
		    parent_graph.subgraph('cluster_'+self._name,
		                          label=self._name)
		for child in self._children:
			if isinstance(child, Block):
				block = child
				g.node(block.name,
				       #label='%s: %s' % (block.type,block.name),
				       label=block.name,
				       shape='box')
				for oring in block.orings:
					g.node(oring.name,
					       shape='ellipse')
					g.edge(block.name, oring.name)
				for iring in block.irings:
					g.edge(iring.name, block.name)
			else:
				#child.dot_graph(g)
				g.subgraph(child.dot_graph())
		return g

class Pipeline(BlockScope):
	def __init__(self, **kwargs):
		super(Pipeline, self).__init__(**kwargs)
		self.blocks = []
	def as_default(self):
		return PipelineContext(self)
	def run(self):
		#print "Launching %i blocks" % len(self.blocks)
		threads = [threading.Thread(target=block.run, name=block.name)
		           for block in self.blocks]
		for thread in threads:
			thread.start()
		#print "Waiting for blocks to finish"
		for thread in threads:
			thread.join()
	def __enter__(self):
		thread_local.pipeline_stack.append(self)
		return self
	def __exit__(self, type, value, tb):
		assert(thread_local.pipeline_stack.pop() is self)

# Create the default pipeline object
thread_local.pipeline_stack.append(Pipeline())
thread_local.blockscope_stack.append(get_default_pipeline())

def get_ring(block_or_ring):
	try:
		return block_or_ring.orings[0]
	except AttributeError:
		return block_or_ring

def block_view(block, header_transform):
	new_block = copy(block)
	new_block.orings = [ring_view(oring, header_transform)
	                    for oring in new_block.orings]
	return new_block

class Block(BlockScope):
	instance_counts = defaultdict(lambda: 0)
	def __init__(self, irings,
	             name=None, # TODO: Move this into BlockScope and join to parent scope name with '/'
	             type_=None,
	             **kwargs):
		super(Block, self).__init__(**kwargs)
		self.type = type_ or self.__class__.__name__
		self.name = name or '%s_%i' % (self.type, Block.instance_counts[self.type])
		Block.instance_counts[self.type] += 1
		
		self.pipeline = get_default_pipeline()
		self.pipeline.blocks.append(self)
		
		# Allow Block instances to be passed in place of rings
		irings = [get_ring(iring) for iring in irings]
		self.irings = irings
		valid_inp_spaces = self._define_valid_input_spaces()
		for i, (iring, valid_spaces) in enumerate(zip(irings, valid_inp_spaces)):
			if not bf.memory.space_accessible(iring.space, valid_spaces):
				raise ValueError("Block %s input %i's space must be accessible from one of: %s" %
				                 (self.name, i, str(valid_spaces)))
		self.orings = [] # Update this in subclass constructors
	def create_ring(self, *args, **kwargs):
		return Ring(*args, owner=self, **kwargs)
	def run(self):
		#bf.affinity.set_openmp_cores(cpus) # TODO
		core = self.core
		if core is not None:
			bf.affinity.set_core(core if isinstance(core, int) else core[0])
		if self.gpu is not None:
			bf.device.set_device(self.gpu)
		self.cache_scope_hierarchy()
		with ExitStack() as oring_stack:
			active_orings = self.begin_writing(oring_stack, self.orings)
			self.main(active_orings)
	def num_outputs(self):
		# TODO: This is a little hacky
		return len(self.orings)
	def begin_writing(self, exit_stack, orings):
		return [exit_stack.enter_context(oring.begin_writing())
		        for oring in orings]
	def begin_sequences(self, exit_stack, orings, oheaders, igulp_nframes):
		ogulp_nframes = self._define_output_nframes(igulp_nframes)
		for ohdr, ogulp_nframe in zip(oheaders, ogulp_nframes):
			ohdr['gulp_nframe'] = ogulp_nframe
		# Note: This always specifies buffer_factor=1 on the assumption that
		#         additional buffering is defined by the reader(s) rather
		#         than the writer.
		obuf_nframes = [1*ogulp_nframe for ogulp_nframe in ogulp_nframes]
		return [exit_stack.enter_context(oring.begin_sequence(ohdr,obuf_nframe))
		        for (oring,ohdr,obuf_nframe) in zip(orings,oheaders,obuf_nframes)]
	def reserve_spans(self, exit_stack, oseqs, ispans):
		igulp_nframes = [span.nframe for span in ispans]
		ogulp_nframes = self._define_output_nframes(igulp_nframes)
		return [exit_stack.enter_context(oseq.reserve(ogulp_nframe))
		        for (oseq,ogulp_nframe) in zip(oseqs,ogulp_nframes)]
	def _define_output_nframes(self, input_nframes):
		return self.define_output_nframes(input_nframes)
	def define_output_nframes(self, input_nframes):
		"""Return output nframe for each output, given input_nframes.
		"""
		raise NotImplementedError
	def _define_valid_input_spaces(self):
		return self.define_valid_input_spaces()
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for each input"""
		return ['any']*len(self.irings)

class SourceBlock(Block):
	def __init__(self, sourcenames, gulp_nframe, *args, **kwargs):
		super(SourceBlock, self).__init__([], *args, gulp_nframe=gulp_nframe, **kwargs)
		self.sourcenames = sourcenames
		default_space = 'cuda_host' if bf.core.cuda_enabled() else 'system'
		self.orings = [self.create_ring(space=default_space)]
		self._seq_count = 0
	def main(self, orings):
		for sourcename in self.sourcenames:
			with self.create_reader(sourcename) as ireader:
				oheaders = self.on_sequence(ireader, sourcename)
				for ohdr in oheaders:
					if 'time_tag' not in ohdr:
						ohdr['time_tag'] = self._seq_count
				self._seq_count += 1
				with ExitStack() as oseq_stack:
					oseqs = self.begin_sequences(oseq_stack, orings, oheaders, igulp_nframes=[])
					while True:
						with ExitStack() as ospan_stack:
							ospans = self.reserve_spans(ospan_stack, oseqs, ispans=[])
							ostrides = self.on_data(ireader, ospans)
							bf.device.stream_synchronize()
							for ospan, ostride in zip(ospans, ostrides):
								ospan.commit(ostride)
							# TODO: Is this an OK way to detect end-of-data?
							if any([ostride==0 for ostride in ostrides]):
								break
	def define_output_nframes(self, _):
		"""Return output nframe for each output, given input_nframes.
		"""
		return [self.gulp_nframe] * self.num_outputs()
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for each input"""
		return []
	def create_reader(self, sourcename):
		"""Return an object to use for reading source data"""
		# TODO: Should return a dummy reader object here?
		raise NotImplementedError
	def on_sequence(self, reader, sourcename):
		"""Return header for each output"""
		raise NotImplementedError
	def on_data(self, reader, ospans):
		"""Process data from from ispans to ospans and return the number of
		frames to commit for each output."""
		raise NotImplementedError


def _span_slice(soft_slice):
	start = soft_slice.start or 0
	return slice(start,
	             soft_slice.stop,
	             soft_slice.step or (soft_slice.stop - start))

class MultiTransformBlock(Block):
	def __init__(self, irings_, guarantee=True, *args, **kwargs):
		super(MultiTransformBlock, self).__init__(irings_, *args, **kwargs)
		# Note: Must use self.irings rather than irings_ because they may
		#         actually be Block instances.
		self.guarantee = guarantee
		self.orings = [self.create_ring(space=iring.space)
		               for iring in self.irings]
		self._seq_count = 0
	def main(self, orings):
		for iseqs in izip(*[iring.read(guarantee=self.guarantee)
		                    for iring in self.irings]):
			oheaders, islices = self._on_sequence(iseqs)
			for ohdr in oheaders:
				if 'time_tag' not in ohdr:
					ohdr['time_tag'] = self._seq_count
			self._seq_count += 1
			
			# Allow passing None to mean slice(gulp_nframe)
			if islices is None:
				islices = [None]*len(self.irings)
			default_igulp_nframes = [self.gulp_nframe or iseq.header['gulp_nframe']
			                        for iseq in iseqs]
			islices = [islice or slice(igulp_nframe)
			           for (islice,igulp_nframe) in
			           zip(islices,default_igulp_nframes)]
			
			islices = [_span_slice(slice_) for slice_ in islices]
			for iseq, islice in zip(iseqs, islices):
				if self.buffer_factor is None:
					src_block = iseq.ring.owner
					if src_block is not None and self.is_fused_with(src_block):
						buffer_factor = 1
					else:
						buffer_factor = None
				else:
					buffer_factor = self.buffer_factor
				iseq.resize(gulp_nframe=(islice.stop - islice.start),
				            buf_nframe=self.buffer_nframe,
				            buffer_factor=buffer_factor)
			
			igulp_nframes = [islice.stop - islice.start for islice in islices]
			
			with ExitStack() as oseq_stack:
				oseqs = self.begin_sequences(oseq_stack, orings, oheaders, igulp_nframes)
				prev_time = time.time()
				for ispans in izip(*[iseq.read(islice.stop - islice.start,
				                              islice.step,
				                              islice.start)
				                    for (iseq,islice)
				                    in zip(iseqs,islices)]):
					cur_time = time.time()
					acquire_time = cur_time - prev_time
					prev_time = cur_time
					with ExitStack() as ospan_stack:
						ospans = self.reserve_spans(ospan_stack, oseqs, ispans)
						cur_time = time.time()
						reserve_time = cur_time - prev_time
						prev_time = cur_time
						# *TODO: See if can fuse together multiple on_data calls here before
						#          calling stream_synchronize().
						#        Consider passing .data instead of rings here
						ostrides = self._on_data(ispans, ospans)
						# TODO: // Default to not spinning the CPU: cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
						bf.device.stream_synchronize()
						# Allow returning None to indicate complete consumption
						if ostrides is None:
							ostrides = [ospan.nframe for ospan in ospans]
						ostrides = [ostride if ostride is not None else ospan.nframe
						            for (ostride,ospan) in zip(ostrides,ospans)]
						for ospan, ostride in zip(ospans, ostrides):
							ospan.commit(ostride)
					cur_time = time.time()
					process_time = cur_time - prev_time
					prev_time = cur_time
					# TODO: Do something with *_time variables (e.g., WAMP PUB)
					#total_time = acquire_time + reserve_time + process_time
					#print acquire_time / total_time, reserve_time / total_time, process_time / total_time
	def _on_sequence(self, iseqs):
		return self.on_sequence(iseqs)
	def _on_data(self, ispans, ospans):
		return self.on_data(ispans, ospans)
	def define_output_nframes(self, input_nframes):
		"""Return output nframe for each output, given input_nframes.
		"""
		return input_nframes
	def on_sequence(self, iseqs):
		"""Return: oheaders (one per output) and islices (one per input)
		"""
		raise NotImplementedError
	def on_data(self, ispans, ospans):
		"""Process data from from ispans to ospans and return the number of
		frames to commit for each output (or None to commit complete spans)."""
		raise NotImplementedError

class TransformBlock(MultiTransformBlock):
	def __init__(self, iring, *args, **kwargs):
		super(TransformBlock, self).__init__([iring], *args, **kwargs)
		self.iring = self.irings[0]
	def _define_valid_input_spaces(self):
		spaces = self.define_valid_input_spaces()
		return [spaces]
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for the input"""
		return 'any'
	def _define_output_nframes(self, input_nframes):
		output_nframe = self.define_output_nframes(input_nframes[0])
		return [output_nframe]
	def define_output_nframes(self, input_nframe):
		"""Return number of frames that will be produced given input_nframe
		"""
		return input_nframe
	def _on_sequence(self, iseqs):
		ret = self.on_sequence(iseqs[0])
		if isinstance(ret, tuple):
			ohdr, islice = ret
		else:
			ohdr = ret
			islice = None
		return [ohdr], [islice]
	def on_sequence(self, iseq):
		"""Return oheader or (oheader, islice)"""
		raise NotImplementedError
	def _on_data(self, ispans, ospans):
		nframe_commit = self.on_data(ispans[0], ospans[0])
		return [nframe_commit]
	def on_data(self, ispan, ospan):
		"""Return the number of output frames to commit, or None to commit all
		"""
		raise NotImplementedError

# TODO: Need something like on_sequence_end to allow closing open files etc.
class SinkBlock(MultiTransformBlock):
	def __init__(self, iring, *args, **kwargs):
		super(SinkBlock, self).__init__([iring], *args, **kwargs)
		self.orings = []
		self.iring  = self.irings[0]
	def _define_valid_input_spaces(self):
		spaces = self.define_valid_input_spaces()
		return [spaces]
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for the input"""
		return 'any'
	def _define_output_nframes(self, input_nframes):
		return []
	def _on_sequence(self, iseqs):
		islice = self.on_sequence(iseqs[0])
		return [], [islice]
	def on_sequence(self, iseq):
		"""Return islice or None to use simple striding"""
		raise NotImplementedError
	def _on_data(self, ispans, ospans):
		self.on_data(ispans[0])
		return []
	def on_data(self, ispan):
		"""Return nothing"""
		raise NotImplementedError
