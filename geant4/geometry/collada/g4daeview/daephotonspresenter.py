#!/usr/bin/env python
"""
::


    g4daeview.sh --with-chroma --load 1 --debugshader --prescale 1000 --max-slots 100
    
        ## terrible performance when push up --max-slots to hold the entire history of the 
        ## propagation, so much so that forced to push up --prescale 
        ## more than proportionately    

    g4daeview.sh --with-chroma --load 1 --debugshader --prescale 1000 --max-slots 10

    udp.py --fphopoint 100

        ## so can see where the few photons are 


Performance killed by doing non-trivial "presenter" kernel 
before every draw, looking thru the slots involves lots of memory 
reads when max_slots 100. Need to split the handling:

#. one kernel post-propagation, that caches info needed for interpolation ? 
   or store info needed whilst doing the propagation 

#. minimal action interpolator, called before every draw that does the minimum
   needed 

Problem is the large number of steps, max_steps of propagation is 
set to 100. And it looks to need that.  The slots are potentially more
the steps.

"""
import logging
log = logging.getLogger(__name__)

import numpy as np
from operator import mul
mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # int division roundup without iffing 

import pycuda.driver as cuda_driver
import pycuda.gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3


from daephotonskernelfunc import DAEPhotonsKernelFunc


class DAEPhotonsPresenter(DAEPhotonsKernelFunc):
    kernel_name = "propagate_vbo.cu"
    kernel_func = "present_vbo"
    kernel_args = "iP"

    def __init__(self, dphotons, ctx, debug=1):
        """
        :param dphotons: DAEPhotons instance
        :param ctx: `DAEChromaContext` instance, for GPU config and geometry
        """
        DAEPhotonsKernelFunc.__init__(self, dphotons, ctx, debug=debug)

    def present(self, vbo_dev_ptr):
        """
        """
        nthreads_per_block = self.ctx.nthreads_per_block
        max_blocks = self.ctx.max_blocks
        photons_this_round = self.nphotons

        block=(nthreads_per_block,1,1)
        nblocks = div_(photons_this_round,nthreads_per_block)
        grid=(nblocks, 1)

        log.info("present photons_this_round %s  grid %s block %s " % (photons_this_round, repr(grid), repr(block)))

        abort = False
        args = ( 
                  np.int32(photons_this_round), 
                  vbo_dev_ptr, 
               )
        get_time = self.kernel.prepared_timed_call( grid, block, *args )
        t = get_time()

        if t > self.max_time:
            abort = True
            log.warn("kernel launch time %s > max_time %s : ABORTING " % (t, self.max_time) )

        cuda_driver.Context.get_current().synchronize()

        log.info("%s time %s " % (self.kernel_func, t )) 

        if abort:assert 0


    def interop_present(self, buf):
        """
        :param buf: OpenGL VBO eg renderer.pbuffer

        Invoke CUDA kernel with VBO argument, 
        allowing VBO changes.
        """ 
        buf_mapping = buf.cuda_buffer_object.map()

        vbo_dev_ptr = buf_mapping.device_ptr()
        self.present( vbo_dev_ptr )   

        cuda_driver.Context.synchronize()
        buf_mapping.unmap()

 


if __name__ == '__main__':
    pass



