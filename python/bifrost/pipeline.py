# -*- coding: utf-8 -*-

# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

import sys
import threading
import Queue
import time
import signal
from copy import copy
from collections import defaultdict
from contextlib2 import ExitStack
import traceback

import bifrost as bf
from bifrost.ring2 import Ring, ring_view
from temp_storage import TempStorage
from bifrost.proclog import ProcLog
from bifrost.ndarray import memset_array # TODO: This feels a bit hacky

# Note: This must be called before any devices are initialized. It's also
#          almost always desirable when running pipelines, so we do it here at
#          module import time to make things easy.
bf.device.set_devices_no_spin_cpu()

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
            #    self._buffer_factor = 1.0
            if self._share_temp_storage is None:
                self._share_temp_storage = True
        self._parent_scope = get_current_block_scope()
        if self._parent_scope is not None:
            self._parent_scope.children.append(self)
            self.name = self._parent_scope.name + '/' + self.name
        self._children = []
    def __enter__(self):
        thread_local.blockscope_stack.append(self)
    def __exit__(self, type, value, tb):
        if __debug__: assert(thread_local.blockscope_stack.pop() is self)
        else: thread_local.blockscope_stack.pop()
    def __getattr__(self, name):
        # Use child's value if set, othersize defer to parent
        if not hasattr(self, '_' + name):
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__, name))
        self_value = getattr(self, '_' + name)
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
        for scope in self.scope_hierarchy:
            if scope.share_temp_storage:
                return scope._get_temp_storage(space)
        return self._get_temp_storage(space)
    def dot_graph(self, parent_graph=None):
        from graphviz import Digraph

        #graph_attr = {'label': self._name}
        graph_attr = {}
        if parent_graph is None:
            g = Digraph('cluster_' + self._name, graph_attr=graph_attr)
        else:
            g = parent_graph.subgraph('cluster_' + self._name,
                                      label=self._name)
        for child in self._children:
            if isinstance(child, Block):
                block = child
                label = block.name.split('/', 1)[1]
                block_colors = defaultdict(lambda: 'white')
                block_colors['CopyBlock'] = 'lightsteelblue'
                block_type = block.__class__.__name__
                fillcolor = block_colors[block_type]
                g.node(block.name,
                       #label='%s: %s' % (block.type,block.name),
                       label=label,
                       shape='box',
                       style='filled',
                       fillcolor=fillcolor)
                for oring in block.orings:
                    space_colors = {
                        'system':    'orange',
                        'cuda':      'limegreen',
                        'cuda_host': 'deepskyblue'
                    }
                    g.node(oring.name,
                           shape='ellipse',
                           style='filled',
                           fillcolor=space_colors[oring.space])
                    g.edge(block.name, oring.name)
                for iring in block.irings:
                    g.edge(iring.name, block.name)
            else:
                #child.dot_graph(g)
                g.subgraph(child.dot_graph())
        return g

def try_join(thread, timeout=0.):
    thread.join(timeout)
    return not thread.is_alive()
# Utility function for joining a collection of threads with a timeout
def join_all(threads, timeout):
    deadline = time.time() + timeout
    alive_threads = list(threads)
    while True:
        alive_threads = [t for t in alive_threads if not try_join(t)]
        available_time = max(deadline - time.time(), 0)
        if (len(alive_threads) == 0 or
            available_time == 0):
            return alive_threads
        alive_threads[0].join(available_time)

class PipelineInitError(Exception):
    pass

class Pipeline(BlockScope):
    instance_count = 0
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'Pipeline_%i' % Pipeline.instance_count
            Pipeline.instance_count += 1
        super(Pipeline, self).__init__(name=name, **kwargs)
        self.blocks = []
        self.shutdown_timeout = 5.
        self.all_blocks_finished_initializing_event = threading.Event()
        self.block_init_queue = Queue.Queue()
    def as_default(self):
        return PipelineContext(self)
    def synchronize_block_initializations(self):
        # Wait for all blocks to finish initializing
        uninitialized_blocks = set(self.blocks)
        while len(uninitialized_blocks):
            # Note: This will get stuck if a transform block has no input ring
            block, init_succeeded = self.block_init_queue.get()
            uninitialized_blocks.remove(block)
            if not init_succeeded:
                self.shutdown()
                raise PipelineInitError(
                    "The following block failed to initialize: " + block.name)
        # Tell blocks that they can begin data processing
        self.all_blocks_finished_initializing_event.set()
    def run(self):
        # Launch blocks as threads
        self.threads = [threading.Thread(target=block.run, name=block.name)
                        for block in self.blocks]
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        self.synchronize_block_initializations()
        # Wait for blocks to finish processing
        for thread in self.threads:
            # Note: Doing it this way allows signals to be caught here
            while thread.is_alive():
                thread.join(timeout=2**30)
    def shutdown(self):
        for block in self.blocks:
            block.shutdown()
        # Ensure all blocks can make progress
        self.all_blocks_finished_initializing_event.set()
        join_all(self.threads, timeout=self.shutdown_timeout)
        for thread in self.threads:
            if thread.is_alive():
                print "WARNING: Thread %s did not shut down on time and will be killed" % thread.name
    def shutdown_on_signals(self, signals=None):
        if signals is None:
            signals = [signal.SIGHUP,
                       signal.SIGINT,
                       signal.SIGQUIT,
                       signal.SIGTERM,
                       signal.SIGTSTP]
        for sig in signals:
            signal.signal(sig, self._handle_signal_shutdown)
    def _handle_signal_shutdown(self, signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and
                            not v.startswith('SIG_'))
        print "WARNING: Received signal %i %s, shutting down pipeline" % (signum, SIGNAL_NAMES[signum])
        self.shutdown()
    def __enter__(self):
        thread_local.pipeline_stack.append(self)
        return self
    def __exit__(self, type, value, tb):
        if __debug__: assert(thread_local.pipeline_stack.pop() is self)
        else: thread_local.pipeline_stack.pop()

# Create the default pipeline object
thread_local.pipeline_stack.append(Pipeline())
thread_local.blockscope_stack.append(get_default_pipeline())

def get_ring(block_or_ring):
    try:
        return block_or_ring.orings[0]
    except AttributeError:
        return block_or_ring

def block_view(block, header_transform):
    """View a block with modified output headers

    Use this function to adjust the output headers of a ring
    on-the-fly, effectively producing a new 'view' of the block.

    Args:
        block (Block): Input block.
        header_transform (function): A function f(hdr) -> new_hdr.

    Returns:
        A new block that acts as the old block but modifies its sequence
        headers on-the-fly.
    """
    new_block = copy(block)
    new_block.orings = [ring_view(oring, header_transform)
                        for oring in new_block.orings]
    return new_block

class Block(BlockScope):
    instance_counts = defaultdict(lambda: 0)
    def __init__(self, irings,
                 name=None,
                 type_=None,
                 **kwargs):
        self.type = type_ or self.__class__.__name__
        self.name = name or '%s_%i' % (self.type, Block.instance_counts[self.type])
        Block.instance_counts[self.type] += 1
        super(Block, self).__init__(**kwargs)
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
        self.shutdown_event = threading.Event()
        self.bind_proclog = ProcLog(self.name + "/bind")
        self.in_proclog = ProcLog(self.name + "/in")

        rnames = {'nring': len(self.irings)}
        for i, r in enumerate(self.irings):
            rnames['ring%i' % i] = r.name
        self.in_proclog.update(rnames)
        self.init_trace = ''.join(traceback.format_stack()[:-1])
    def shutdown(self):
        self.shutdown_event.set()
    def create_ring(self, *args, **kwargs):
        return Ring(*args, owner=self, **kwargs)
    def run(self):
        #bf.affinity.set_openmp_cores(cpus) # TODO
        core = self.core
        if core is not None:
            bf.affinity.set_core(core if isinstance(core, int) else core[0])
        self.bind_proclog.update({'ncore': 1,
                                  'core0': bf.affinity.get_core()})
        if self.gpu is not None:
            bf.device.set_device(self.gpu)
        self.cache_scope_hierarchy()
        with ExitStack() as oring_stack:
            active_orings = self.begin_writing(oring_stack, self.orings)
            try:
                self.main(active_orings)
            except Exception:
                self.pipeline.block_init_queue.put((self, False))
                sys.stderr.write("From block instantiated here:\n")
                sys.stderr.write(self.init_trace)
                raise
    def num_outputs(self):
        # TODO: This is a little hacky
        return len(self.orings)
    def begin_writing(self, exit_stack, orings):
        return [exit_stack.enter_context(oring.begin_writing())
                for oring in orings]
    def begin_sequences(self, exit_stack, orings, oheaders,
                        igulp_nframes, istride_nframes):
        # Note: The gulp_nframe that is set in the output header does not
        #         include the overlap (i.e., it's based on stride not gulp).
        ostride_nframes = self._define_output_nframes(istride_nframes)
        for ohdr, ostride_nframe in zip(oheaders, ostride_nframes):
            ohdr['gulp_nframe'] = ostride_nframe
        ogulp_nframes = self._define_output_nframes(igulp_nframes)
        # Note: This always specifies buffer_factor=1 on the assumption that
        #         additional buffering is defined by the reader(s) rather
        #         than the writer.
        obuf_nframes = [1 * ogulp_nframe for ogulp_nframe in ogulp_nframes]
        oseqs = [exit_stack.enter_context(oring.begin_sequence(ohdr,
                                                               ogulp_nframe,
                                                               obuf_nframe))
                 for (oring, ohdr, ogulp_nframe, obuf_nframe)
                 in zip(orings, oheaders, ogulp_nframes, obuf_nframes)]

        # Synchronize all blocks here to ensure no sequence race conditions
        self.pipeline.block_init_queue.put((self, True))
        self.pipeline.all_blocks_finished_initializing_event.wait()

        ogulp_overlaps = [ogulp_nframe - ostride_nframe
                          for ogulp_nframe, ostride_nframe
                          in zip(ogulp_nframes, ostride_nframes)]
        return oseqs, ogulp_overlaps
    def reserve_spans(self, exit_stack, oseqs, igulp_nframes=[]):
        ogulp_nframes = self._define_output_nframes(igulp_nframes)
        return [exit_stack.enter_context(oseq.reserve(ogulp_nframe))
                for (oseq, ogulp_nframe) in zip(oseqs, ogulp_nframes)]
    def commit_spans(self, ospans, ostrides_actual, ogulp_overlaps):
        # Allow returning None to indicate complete consumption
        if ostrides_actual is None:
            ostrides = [None] * len(ospans)
        # Note: If ospan.nframe < ogulp_overlap, no frames will be committed
        ostrides = [ostride if ostride is not None
                    else max(ospan.nframe - ogulp_overlap, 0)
                    for (ostride, ospan, ogulp_overlap)
                    in zip(ostrides_actual, ospans, ogulp_overlaps)]
        for ospan, ostride in zip(ospans, ostrides):
            ospan.commit(ostride)
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
        return ['any'] * len(self.irings)

class SourceBlock(Block):
    def __init__(self, sourcenames, gulp_nframe, space=None, *args, **kwargs):
        super(SourceBlock, self).__init__([], *args, gulp_nframe=gulp_nframe, **kwargs)
        self.sourcenames = sourcenames
        default_space = 'cuda_host' if bf.core.cuda_enabled() else 'system'
        if space is None:
            space = default_space
        self.orings = [self.create_ring(space=space)]
        self._seq_count = 0
        self.perf_proclog = ProcLog(self.name + "/perf")
        self.out_proclog = ProcLog(self.name + "/out")

        rnames = {'nring': len(self.orings)}
        for i, r in enumerate(self.orings):
            rnames['ring%i' % i] = r.name
        self.out_proclog.update(rnames)

    def main(self, orings):
        for sourcename in self.sourcenames:
            if self.shutdown_event.is_set():
                break
            with self.create_reader(sourcename) as ireader:
                oheaders = self.on_sequence(ireader, sourcename)
                for ohdr in oheaders:
                    if 'time_tag' not in ohdr:
                        ohdr['time_tag'] = self._seq_count
                    if 'name' not in ohdr:
                        ohdr['name'] = 'unnamed-sequence-%i' % self._seq_count
                self._seq_count += 1
                with ExitStack() as oseq_stack:
                    oseqs, ogulp_overlaps = self.begin_sequences(
                        oseq_stack, orings, oheaders,
                        igulp_nframes=[],
                        istride_nframes=[])
                    while not self.shutdown_event.is_set():
                        prev_time = time.time()
                        with ExitStack() as ospan_stack:
                            ospans = self.reserve_spans(ospan_stack, oseqs)
                            cur_time = time.time()
                            reserve_time = cur_time - prev_time
                            prev_time = cur_time
                            ostrides_actual = self.on_data(ireader, ospans)
                            bf.device.stream_synchronize()
                            self.commit_spans(ospans, ostrides_actual, ogulp_overlaps)
                            # TODO: Is this an OK way to detect end-of-data?
                            if any([ostride == 0 for ostride in ostrides_actual]):
                                break
                        cur_time = time.time()
                        process_time = cur_time - prev_time
                        prev_time = cur_time
                        self.perf_proclog.update({
                            'acquire_time': -1,
                            'reserve_time': reserve_time,
                            'process_time': process_time})
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
    # Infers optional values in soft_slice (i.e., those that are None)
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
        self.perf_proclog = ProcLog(self.name + "/perf")
        self.sequence_proclogs = [ProcLog(self.name + "/sequence%i" % i)
                                  for i in xrange(len(self.irings))]
        self.out_proclog = ProcLog(self.name + "/out")

        rnames = {'nring': len(self.orings)}
        for i, r in enumerate(self.orings):
            rnames['ring%i' % i] = r.name
        self.out_proclog.update(rnames)

    def main(self, orings):
        for iseqs in izip(*[iring.read(guarantee=self.guarantee)
                            for iring in self.irings]):
            if self.shutdown_event.is_set():
                break
            for i, iseq in enumerate(iseqs):
                self.sequence_proclogs[i].update(iseq.header)
            oheaders = self._on_sequence(iseqs)
            for ohdr in oheaders:
                if 'time_tag' not in ohdr:
                    ohdr['time_tag'] = self._seq_count
            self._seq_count += 1

            igulp_nframes = [self.gulp_nframe or iseq.header['gulp_nframe']
                             for iseq in iseqs]
            igulp_overlaps = self._define_input_overlap_nframe(iseqs)
            istride_nframes = igulp_nframes[:]
            igulp_nframes = [igulp_nframe + nframe_overlap
                             for igulp_nframe, nframe_overlap
                             in zip(igulp_nframes, igulp_overlaps)]

            for iseq, igulp_nframe in zip(iseqs, igulp_nframes):
                if self.buffer_factor is None:
                    src_block = iseq.ring.owner
                    if src_block is not None and self.is_fused_with(src_block):
                        buffer_factor = 1
                    else:
                        buffer_factor = None
                else:
                    buffer_factor = self.buffer_factor
                iseq.resize(gulp_nframe=igulp_nframe,
                            buf_nframe=self.buffer_nframe,
                            buffer_factor=buffer_factor)

            # TODO: Ever need to specify starting offset?
            iframe0s = [0 for _ in igulp_nframes]

            force_skip = False

            with ExitStack() as oseq_stack:
                oseqs, ogulp_overlaps = self.begin_sequences(
                    oseq_stack, orings, oheaders,
                    igulp_nframes, istride_nframes)
                if self.shutdown_event.is_set():
                    break
                prev_time = time.time()
                for ispans in izip(*[iseq.read(igulp_nframe,
                                               istride_nframe,
                                               iframe0)
                                    for (iseq, igulp_nframe, istride_nframe, iframe0)
                                    in zip(iseqs, igulp_nframes, istride_nframes, iframe0s)]):
                    if self.shutdown_event.is_set():
                        return

                    if any([ispan.nframe_skipped for ispan in ispans]):
                        # There were skipped (overwritten) frames
                        with ExitStack() as ospan_stack:
                            iskip_slices = [slice(iframe0,
                                                  iframe0 + ispan.nframe_skipped,
                                                  istride_nframe)
                                            for iframe0, istride_nframe, ispan in
                                            zip(iframe0s, istride_nframes, ispans)]
                            iskip_nframes = [ispan.nframe_skipped
                                             for ispan in ispans]
                            # ***TODO: Need to loop over multiple ospans here,
                            #            because iskip_nframes can be
                            #            arbitrarily large!
                            ospans = self.reserve_spans(ospan_stack, oseqs, iskip_nframes)
                            ostrides_actual = self._on_skip(iskip_slices, ospans)
                            bf.device.stream_synchronize()
                            self.commit_spans(ospans, ostrides_actual, ogulp_overlaps)

                    if all([ispan.nframe == 0 for ispan in ispans]):
                        # No data to see here, move right along
                        continue

                    cur_time = time.time()
                    acquire_time = cur_time - prev_time
                    prev_time = cur_time

                    with ExitStack() as ospan_stack:
                        igulp_nframes = [ispan.nframe for ispan in ispans]
                        ospans = self.reserve_spans(ospan_stack, oseqs, igulp_nframes)
                        cur_time = time.time()
                        reserve_time = cur_time - prev_time
                        prev_time = cur_time

                        if not force_skip:
                            # *TODO: See if can fuse together multiple on_data calls here before
                            #          calling stream_synchronize().
                            #        Consider passing .data instead of rings here
                            ostrides_actual = self._on_data(ispans, ospans)
                            bf.device.stream_synchronize()

                        any_frames_overwritten = any([ispan.nframe_overwritten
                                                      for ispan in ispans])
                        if force_skip or any_frames_overwritten:
                            # Note: To allow interrupted pipelines to catch up,
                            #         we force-skip an additional gulp whenever
                            #         a span is overwritten during on_data.
                            force_skip = any_frames_overwritten
                            iskip_slices = [slice(ispan.frame_offset,
                                                  ispan.frame_offset + ispan.nframe_overwritten,
                                                  istride_nframe)
                                            for ispan, istride_nframe
                                            in zip(ispans, istride_nframes)]
                            ostrides_actual = self._on_skip(iskip_slices, ospans)
                            bf.device.stream_synchronize()

                        self.commit_spans(ospans, ostrides_actual, ogulp_overlaps)
                    cur_time = time.time()
                    process_time = cur_time - prev_time
                    prev_time = cur_time
                    self.perf_proclog.update({
                        'acquire_time': acquire_time,
                        'reserve_time': reserve_time,
                        'process_time': process_time})
            # **TODO: This will not be called if an exception is raised
            #           Need to call it from a context manager somehow
            self._on_sequence_end(iseqs)
    def _on_sequence(self, iseqs):
        return self.on_sequence(iseqs)
    def _on_sequence_end(self, iseqs):
        return self.on_sequence_end(iseqs)
    def _on_data(self, ispans, ospans):
        return self.on_data(ispans, ospans)
    def _on_skip(self, islices, ospans):
        return self.on_skip(islices, ospans)
    def _define_input_overlap_nframe(self, iseqs):
        return self.define_input_overlap_nframe(iseqs)
    def define_input_overlap_nframe(self, iseqs):
        """Return no. input frames that should overlap between successive spans
        for each input sequence.
        """
        return [0] * len(self.irings)
    def define_output_nframes(self, input_nframes):
        """Return output nframe for each output, given input_nframes.
        """
        return input_nframes
    def on_sequence(self, iseqs):
        """Return: oheaders (one per output)
        """
        raise NotImplementedError
    def on_sequence_end(self, iseqs):
        """Do any necessary cleanup"""
        pass
    def on_data(self, ispans, ospans):
        """Process data from from ispans to ospans and return the number of
        frames to commit for each output (or None to commit complete spans)."""
        raise NotImplementedError
    def on_skip(self, islices, ospans):
        """Handle skipped frames"""
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
    def _define_input_overlap_nframe(self, iseqs):
        return [self.define_input_overlap_nframe(iseqs[0])]
    def define_input_overlap_nframe(self, iseq):
        """Return no. input frames that should overlap between successive spans.
        """
        return 0
    def _define_output_nframes(self, input_nframes):
        output_nframe = self.define_output_nframes(input_nframes[0])
        return [output_nframe]
    def define_output_nframes(self, input_nframe):
        """Return number of frames that will be produced given input_nframe
        """
        return input_nframe
    def _on_sequence(self, iseqs):
        return [self.on_sequence(iseqs[0])]
    def on_sequence(self, iseq):
        """Return oheader"""
        raise NotImplementedError
    def _on_sequence_end(self, iseqs):
        return [self.on_sequence_end(iseqs[0])]
    def on_sequence_end(self, iseq):
        """Do any necessary cleanup"""
        pass
    def _on_data(self, ispans, ospans):
        nframe_commit = self.on_data(ispans[0], ospans[0])
        return [nframe_commit]
    def on_data(self, ispan, ospan):
        """Return the number of output frames to commit, or None to commit all
        """
        raise NotImplementedError
    def _on_skip(self, islices, ospans):
        return [self.on_skip(islices[0], ospans[0])]
    def on_skip(self, islice, ospan):
        """Handle skipped frames"""
        # Note: This zeros the whole gulp, even though only part of the gulp
        #         may have been overwritten.
        memset_array(ospan.data, 0)
        #for i in xrange(0, ispan.nframe_skipped, igulp_nframe):
        #    inframe = min(igulp_nframe, inskipped - i)
        #    onframe = self._define_output_nframes(inframe)
        #    with oseq.reserve(onframe) as ospan:
        #        bf.ndarray.memset_array(ospan.data, 0)

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
    def _define_input_overlap_nframe(self, iseqs):
        return [self.define_input_overlap_nframe(iseqs[0])]
    def define_input_overlap_nframe(self, iseq):
        """Return no. input frames that should overlap between successive spans.
        """
        return 0
    def _define_output_nframes(self, input_nframes):
        return []
    def _on_sequence(self, iseqs):
        self.on_sequence(iseqs[0])
        return []
    def on_sequence(self, iseq):
        """Return islice or None to use simple striding"""
        raise NotImplementedError
    def _on_sequence_end(self, iseqs):
        return [self.on_sequence_end(iseqs[0])]
    def on_sequence_end(self, iseq):
        """Do any necessary cleanup"""
        pass
    def _on_data(self, ispans, ospans):
        self.on_data(ispans[0])
        return []
    def on_data(self, ispan):
        """Return nothing"""
        raise NotImplementedError
