#!/usr/bin/env python
"""

Can the VBO `dphotons.data.pbuffer` be made to act as a replacement 
for `chroma.gpu.photon.GPUPhotons` whilst visualizing ?

Compare with the old propagate method, photons are oscillating
between host and device::

   gpu_photons = GPUPhotons(photons)
   gpu_photons.propagate(self.ctx.gpu_geometry, 
                         self.ctx.rng_states,
                         nthreads_per_block=self.ctx.nthreads_per_block,
                         max_blocks=self.ctx.max_blocks,
                         max_steps=max_steps)
   photons_end = gpu_photons.get()
   return photons_end 


Task is: 

#. adapt /usr/local/env/chroma_env/src/chroma/chroma/gpu/photon.py
   to VBO data structure in conjunction with new kernel propagate_vbo.cu

#. use CUDA techniques from ~/e/chroma/chroma_camera/pbo_renderer.py

"""

import logging
log = logging.getLogger(__name__)

import numpy as np
from operator import mul
mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # int division roundup without iffing 

import pycuda.driver as cuda
import pycuda.gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3


class DAEPhotonsPropagator(object):
    def __init__(self, dphotons, ctx ):
        """
        :param dphotons: DAEPhotons instance
        :param ctx: `DAEChromaContext` instance, for GPU config and geometry
        """
        self.max_time = 4.
        self.dphotons = dphotons
        self.max_slots = dphotons.data.max_slots
        self.ctx = ctx
        self.compile_kernel()
        self.uploaded_queues = False

    def compile_kernel(self, template_vars = None ):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        module = get_cu_module('propagate_vbo.cu', options=cuda_options, template_vars=template_vars )
        kernel = module.get_function( 'propagate_vbo' )
        kernel.prepare("iiPPPPiiiPi")

        self.g_mask = module.get_global("g_mask")[0]  
        self._mask = None
        self.kernel = kernel

    nphotons = property(lambda self:self.dphotons.data.nphotons)

    def initialize_constants(self):
        self.mask = [-1,-1,-1,-1]

    def update_constants(self):
        self.mask = self.dphotons.param.kernel_mask

    def _get_mask(self):
        return self._mask 
    def _set_mask(self, mask):
        if mask == self._mask:return
        self._mask = mask
        log.info("_set_mask : memcpy_htod %s " % repr(mask))
        cuda.memcpy_htod(self.g_mask, ga.vec.make_int4(*mask))
    mask = property(_get_mask, _set_mask, doc="setter copies to device __constant__ memory, getter returns cached value") 

    def reset(self):
        self.uploaded_queues = False

    def upload_queues(self, nwork):
        """
        Only needed once ? When single stepping with repeated calls ?

        ::

            In [136]: input_queue
            Out[136]: array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint32)

        """
        if self.uploaded_queues:return
        self.uploaded_queues = True

        log.info("upload_queues %s " % nwork )

        input_queue = np.empty(shape=nwork+1, dtype=np.uint32)
        input_queue[0] = 0
        input_queue[1::1] = np.arange(nwork, dtype=np.uint32) 

        output_queue = np.zeros(shape=nwork+1, dtype=np.uint32)
        output_queue[0] = 1

        self.input_queue_gpu = ga.to_gpu(input_queue)
        self.output_queue_gpu = ga.to_gpu(output_queue)

    def swap_queues(self):
        """
        Swaps queues and returns photons remaining to propagate

        The setting of length 1 array quells a pycuda warning
        """
        temp = self.input_queue_gpu
        self.input_queue_gpu = self.output_queue_gpu
        self.output_queue_gpu = temp
        self.output_queue_gpu[:1].set(np.ones(shape=1, dtype=np.uint32))  
        slot0minus1 = self.input_queue_gpu[:1].get()[0] - 1  # which was just now the output_queue before swap
        log.info("swap_queues slot0minus1 %s " % slot0minus1 )
        return slot0minus1

    def propagate(self, 
                  vbo_dev_ptr, 
                  max_steps=10, 
                  use_weights=False,
                  scatter_first=0):
        """
        Adaption of chroma.gpu.photon.GPUPhotons:propagate

        Propagate photons on GPU to termination or max_steps, whichever
        comes first. May be called repeatedly without reloading photon information if
        single-stepping through photon history.

        ..warning::
            `rng_states` must have at least `nthreads_per_block`*`max_blocks` number of curandStates.

        Hmm how to grab from VBO equivalently to::

            if ga.max(self.flags).get() & (1 << 31):
                print >>sys.stderr, "WARNING: ABORTED PHOTONS"

        """
        nwork = self.nphotons
        self.upload_queues(nwork)

        nthreads_per_block = self.ctx.nthreads_per_block
        max_blocks = self.ctx.max_blocks

        small_remainder = nthreads_per_block * 16 * 8
        block=(nthreads_per_block,1,1)

        step = 0
        while step < max_steps:
            pass
            if nwork < small_remainder or use_weights:
                nsteps = max_steps - step       
                log.info("increase nsteps for stragglers: small_remainder %s nwork %s nsteps %s max_steps %s " % (small_remainder, nwork, nsteps, max_steps))
            else:
                nsteps = 1
            pass

            log.info("nwork %s step %s max_steps %s nsteps %s " % (nwork, step,max_steps, nsteps) )

            times = []
            abort = False
            for first_photon, photons_this_round, blocks in chunk_iterator(nwork, nthreads_per_block, max_blocks):
                if abort:
                    t = -1 
                else:
                    log.info("prepared_call first_photon %s photons_this_round %s nsteps %s " % (first_photon, photons_this_round, nsteps))

                    grid=(blocks, 1)
                    args = ( np.int32(first_photon), 
                             np.int32(photons_this_round), 
                             self.input_queue_gpu[1:].gpudata, 
                             self.output_queue_gpu.gpudata, 
                             self.ctx.rng_states, 
                             vbo_dev_ptr,
                             np.int32(nsteps), 
                             np.int32(use_weights), 
                             np.int32(scatter_first), 
                             self.ctx.gpu_geometry.gpudata,
                             np.int32(self.max_slots )) 

                    get_time = self.kernel.prepared_timed_call( grid, block, *args )
                    t = get_time()
                    times.append(t)
                    if t > self.max_time:
                        abort=True
                        log.warn("kernel launch time %s > max_time %s : ABORTING " % (t, self.max_time) )
                    pass
                pass  
            pass
            log.info("launch sequence times %s " % repr(times))

            step += nsteps
            scatter_first = 0 # Only allow non-zero in first pass
            if step < max_steps:
                nwork = self.swap_queues()
                log.info("result of swap_queues nwork %s " % nwork )  
            else:
                log.info("DONE step %s max_steps %s " % (step, max_steps))
            pass
        pass 
        cuda.Context.get_current().synchronize()

 


if __name__ == '__main__':
    pass



