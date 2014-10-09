#!/usr/bin/env python
"""
DAEChromaContext
==================

To keep this usable from different environments, keep top level 
imports to a minimum. Especially ones that require contexts to be active.

For example DAERaycaster pulls in PixelBuffer which requires 
an active OpenGL context so defer the import until needed.

"""
import os, time, logging
log = logging.getLogger(__name__)

import numpy as np
import pycuda.gl.autoinit  # after this can use pycuda.gl.BufferObject(unsigned int)

def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16)

class DAEChromaContext(object):
    """
    DCC is intended as a rack on which to hang objects, 
    avoid "doing" anything substantial here 
    (eg do stepping in the propagator not here)
    """
    dummy = False
    def __init__(self, config, chroma_geometry ):
        log.debug("DAEChromaContext init, CUDA_PROFILE %s " % os.environ.get('CUDA_PROFILE',"not-defined") )
        self.config = config
        self.chroma_geometry = chroma_geometry
        pass
        self.deviceid = config.args.deviceid 
        self.nthreads_per_block = config.args.threads_per_block
        self.max_blocks = config.args.max_blocks
        self.seed = config.args.seed
        pass
        self.setup_random_seed()
        pass
        self._gpu_geometry = None
        self._gpu_detector = None
        self._rng_states = None
        self._raycaster = None
        self._propagator = None

    def setup_random_seed(self):
        if self.seed is None:
            self.seed = pick_seed() 
            log.warn("RANDOMLY SETTING SEED TO %s " % self.seed )
        else:
            log.info("using seed %s " % self.seed )
        pass 
        np.random.seed(self.seed)

    def setup_rng_states(self):
        from chroma.gpu.tools import get_rng_states
        log.info("setup_rng_states using seed %s "  % self.seed )
        rng_states = get_rng_states(self.nthreads_per_block*self.max_blocks, seed=self.seed)
        return rng_states

    def setup_raycaster(self):
        from daeraycaster import DAERaycaster
        return DAERaycaster( self )

    def setup_propagator(self):
        from env.chroma.chroma_propagator.propagator import Propagator
        return Propagator( self )

    def setup_gpu_geometry(self):
        from chroma.gpu.geometry import GPUGeometry
        return GPUGeometry( self.chroma_geometry )

    def setup_gpu_detector(self):
        """
        For add_pmt rather than add_solid which have a channel_id
        to copy onto the GPU 

        Use either gpu_geometry OR gpu_detector, NOT BOTH
        """
        from chroma.gpu.detector import GPUDetector
        return GPUDetector( self.chroma_geometry )

    def make_cuda_buffer_object(self, buffer_id ):
        import pycuda.gl as cuda_gl
        return cuda_gl.BufferObject(long(buffer_id))  

    def _get_gpu_geometry(self):
        if self._gpu_geometry is None:
            self._gpu_geometry = self.setup_gpu_geometry()
        return self._gpu_geometry
    gpu_geometry = property(_get_gpu_geometry)

    def _get_gpu_detector(self):
        if self._gpu_detector is None:
            self._gpu_detector = self.setup_gpu_detector()
        return self._gpu_detector
    gpu_detector = property(_get_gpu_detector)

    def _get_rng_states(self):
        log.info("_get_rng_states")
        if self._rng_states is None:
            self._rng_states = self.setup_rng_states()
        return self._rng_states
    rng_states = property(_get_rng_states)

    def _get_raycaster(self):
        if self._raycaster is None:
            self._raycaster = self.setup_raycaster()
        return self._raycaster
    raycaster = property(_get_raycaster)

    def _get_propagator(self):
        if self._propagator is None:
           self._propagator = self.setup_propagator()
        return self._propagator
    propagator = property(_get_propagator)  




if __name__ == '__main__':
    pass

