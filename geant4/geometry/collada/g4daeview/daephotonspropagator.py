#!/usr/bin/env python
"""
For individual photon debug tracing::

   g4daeview.sh --with-chroma --load 1 --debugshader --max-slots 10 --debugkernel --debugphoton 4


Low stat observations:

#. many photons only need a few steps, but some are taking more than 50    
#. normally repeatably get same photons on each launch

   * BUT sometimes get different ones, seed problem ? rng setup ordering 

NEXT:

#. pull back the VBO into numpy arrays, so can find photons with interesting histories 


::

2014-06-18 20:47:36,658 env.geant4.geometry.collada.g4daeview.daephotonspropagator:144 prepared_call first_photon 0 photons_this_round 4165 nsteps 100 
FILL_STATE       START    [  1000] slot  0 steps  1 lht   1262 tpos    9.693  -17818.29 -800100.38   -7063.31    w  412.76   dir    -0.49    -0.01    -0.87 pol    0.161   -0.983   -0.084 
TO_BOUNDARY      CONTINUE [  1000] slot -1 steps  1 lht   1262 tpos   15.668  -18406.47 -800108.31   -8104.00    w     inf   dir     0.21     0.54    -0.82 pol    0.690   -0.673   -0.267 BULK_REEMIT 
FILL_STATE       CONTINUE [  1000] slot  1 steps  2 lht   1260 tpos   15.668  -18406.47 -800108.31   -8104.00    w     inf   dir     0.21     0.54    -0.82 pol    0.690   -0.673   -0.267 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  2 lht   1260 tpos   18.875  -18270.04 -799757.94   -8635.00    w     inf   dir     0.21     0.54    -0.82 pol    0.690   -0.673   -0.267 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  2 lht   1260 tpos   18.875  -18270.04 -799757.94   -8635.00    w     inf   dir     0.21     0.53    -0.82 pol    0.297    0.763    0.574 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  2 steps  3 lht    996 tpos   18.875  -18270.04 -799757.94   -8635.00    w     inf   dir     0.21     0.53    -0.82 pol    0.297    0.763    0.574 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  3 lht    996 tpos   18.966  -18266.22 -799748.12   -8650.00    w     inf   dir     0.21     0.53    -0.82 pol    0.297    0.763    0.574 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  3 lht    996 tpos   18.966  -18266.22 -799748.12   -8650.00    w     inf   dir     0.21     0.54    -0.82 pol    0.296    0.761    0.578 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  3 steps  4 lht   4133 tpos   18.966  -18266.22 -799748.12   -8650.00    w     inf   dir     0.21     0.54    -0.82 pol    0.296    0.761    0.578 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  4 lht   4133 tpos   19.500  -18243.53 -799689.88   -8738.33    w     inf   dir     0.21     0.54    -0.82 pol    0.296    0.761    0.578 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  4 lht   4133 tpos   19.500  -18243.53 -799689.88   -8738.33    w     inf   dir     0.21     0.55    -0.81 pol   -0.108   -0.811   -0.574 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  4 steps  5 lht   4136 tpos   19.500  -18243.53 -799689.88   -8738.33    w     inf   dir     0.21     0.55    -0.81 pol   -0.108   -0.811   -0.574 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  5 lht   4136 tpos   19.929  -18225.30 -799642.62   -8808.48    w     inf   dir     0.21     0.55    -0.81 pol   -0.108   -0.811   -0.574 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  5 lht   4136 tpos   19.929  -18225.30 -799642.62   -8808.48    w     inf   dir     0.21     0.54    -0.82 pol   -0.110   -0.816   -0.567 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  5 steps  6 lht   3929 tpos   19.929  -18225.30 -799642.62   -8808.48    w     inf   dir     0.21     0.54    -0.82 pol   -0.110   -0.816   -0.567 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  6 lht   3929 tpos   20.630  -18195.50 -799566.06   -8924.48    w     inf   dir     0.21     0.54    -0.82 pol   -0.110   -0.816   -0.567 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  6 lht   3929 tpos   20.630  -18195.50 -799566.06   -8924.48    w     inf   dir     0.22     0.54    -0.81 pol   -0.837   -0.323   -0.443 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  6 steps  7 lht   3932 tpos   20.630  -18195.50 -799566.06   -8924.48    w     inf   dir     0.22     0.54    -0.81 pol   -0.837   -0.323   -0.443 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  7 lht   3932 tpos   21.148  -18172.56 -799509.50   -9009.07    w     inf   dir     0.22     0.54    -0.81 pol   -0.837   -0.323   -0.443 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  7 lht   3932 tpos   21.148  -18172.56 -799509.50   -9009.07    w     inf   dir     0.21     0.54    -0.82 pol   -0.839   -0.329   -0.433 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  7 steps  8 lht    634 tpos   21.148  -18172.56 -799509.50   -9009.07    w     inf   dir     0.21     0.54    -0.82 pol   -0.839   -0.329   -0.433 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  8 lht    634 tpos   21.649  -18151.25 -799454.81   -9092.00    w     inf   dir     0.21     0.54    -0.82 pol   -0.839   -0.329   -0.433 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  8 lht    634 tpos   21.649  -18151.25 -799454.81   -9092.00    w     inf   dir     0.21     0.53    -0.82 pol    0.932   -0.363    0.000 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  8 steps  9 lht 631702 tpos   21.649  -18151.25 -799454.81   -9092.00    w     inf   dir     0.21     0.53    -0.82 pol    0.932   -0.363    0.000 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps  9 lht 631702 tpos   21.756  -18146.67 -799443.06   -9110.00    w     inf   dir     0.21     0.53    -0.82 pol    0.932   -0.363    0.000 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps  9 lht 631702 tpos   21.756  -18146.67 -799443.06   -9110.00    w     inf   dir     0.20     0.52    -0.83 pol    0.932   -0.363    0.000 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot  9 steps 10 lht 632006 tpos   21.756  -18146.67 -799443.06   -9110.00    w     inf   dir     0.20     0.52    -0.83 pol    0.932   -0.363    0.000 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps 10 lht 632006 tpos   21.815  -18144.23 -799436.81   -9119.90    w     inf   dir     0.20     0.52    -0.83 pol    0.932   -0.363    0.000 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps 10 lht 632006 tpos   21.815  -18144.23 -799436.81   -9119.90    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot 10 steps 11 lht 632304 tpos   21.815  -18144.23 -799436.81   -9119.90    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT 
TO_BOUNDARY      PASS     [  1000] slot -1 steps 11 lht 632304 tpos   21.815  -18144.20 -799436.75   -9119.95    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT 
AT_BOUNDARY      CONTINUE [  1000] slot -1 steps 11 lht 632304 tpos   21.815  -18144.20 -799436.75   -9119.95    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT 
FILL_STATE       PASS     [  1000] slot 11 steps 12 lht 632331 tpos   21.815  -18144.20 -799436.75   -9119.95    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT 
TO_BOUNDARY      BREAK    [  1000] slot -1 steps 12 lht     -1 tpos   21.815  -18144.20 -799436.75   -9119.95    w     inf   dir     0.30     0.78    -0.55 pol    0.932   -0.363    0.000 BULK_REEMIT BULK_ABSORB 
2014-06-18 20:47:37,729 env.geant4.geometry.collada.g4daeview.daephotonspropagator:167 launch sequence times [0.63269482421875] 
2014-06-18 20:47:37,729 env.geant4.geometry.collada.g4daeview.daephotonspropagator:175 DONE step 100 max_steps 100 




"""

import logging
log = logging.getLogger(__name__)

import numpy as np
from operator import mul
mul_ = lambda _:reduce(mul, _)          # product of elements 
div_ = lambda num,den:(num+den-1)//den  # int division roundup without iffing 

import pycuda.driver as cuda_driver
import pycuda.gpuarray as ga

from chroma.gpu.tools import chunk_iterator


from daephotonskernelfunc import DAEPhotonsKernelFunc


class DAEPhotonsPropagator(DAEPhotonsKernelFunc):
    kernel_name = "propagate_vbo.cu"
    kernel_func = "propagate_vbo"
    kernel_args = "iiPPPPiiiP"

    def __init__(self, dphotons, ctx, debug=0):
        """
        :param dphotons: DAEPhotons instance
        :param ctx: `DAEChromaContext` instance, for GPU config and geometry

        #. max_slots, numquad are interpolated into kernel source coming from DAEPhotonsData
        """
        DAEPhotonsKernelFunc.__init__(self, dphotons, ctx, debug=debug)  
        self.uploaded_queues = False

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

        log.debug("upload_queues %s " % nwork )

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
        log.debug("swap_queues slot0minus1 %s " % slot0minus1 )
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
                log.debug("increase nsteps for stragglers: small_remainder %s nwork %s nsteps %s max_steps %s " % (small_remainder, nwork, nsteps, max_steps))
            else:
                nsteps = 1
            pass

            log.debug("nwork %s step %s max_steps %s nsteps %s " % (nwork, step,max_steps, nsteps) )

            times = []
            abort = False
            for first_photon, photons_this_round, blocks in chunk_iterator(nwork, nthreads_per_block, max_blocks):
                if abort:
                    t = -1 
                else:
                    log.debug("prepared_call first_photon %s photons_this_round %s nsteps %s " % (first_photon, photons_this_round, nsteps))

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
                             self.ctx.gpu_geometry.gpudata) 

                    get_time = self.kernel.prepared_timed_call( grid, block, *args )
                    t = get_time()
                    times.append(t)
                    if t > self.max_time:
                        abort=True
                        log.warn("kernel launch time %s > max_time %s : ABORTING " % (t, self.max_time) )
                    pass
                pass  
            pass
            log.debug("launch sequence times %s " % repr(times))

            step += nsteps
            scatter_first = 0 # Only allow non-zero in first pass
            if step < max_steps:
                nwork = self.swap_queues()
                log.info("result of swap_queues nwork %s " % nwork )  
            else:
                log.debug("DONE step %s max_steps %s " % (step, max_steps))
            pass
        pass 
        cuda_driver.Context.get_current().synchronize()


    def interop_propagate(self, buf, max_steps=10 ):
        """
        :param buf: OpenGL VBO eg renderer.pbuffer

        Invoke CUDA kernel with VBO argument, 
        allowing VBO changes.
        """ 
        buf_mapping = buf.cuda_buffer_object.map()

        vbo_dev_ptr = buf_mapping.device_ptr()
        self.propagate( vbo_dev_ptr, max_steps=max_steps )   

        cuda_driver.Context.synchronize()
        buf_mapping.unmap()

 


if __name__ == '__main__':
    pass



