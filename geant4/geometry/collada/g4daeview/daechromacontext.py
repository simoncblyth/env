#!/usr/bin/env python

import os, time, logging
log = logging.getLogger(__name__)

import numpy as np

import pycuda.gl.autoinit

from chroma import gpu
from chroma.gpu.geometry import GPUGeometry

from env.chroma.chroma_propagator.propagator import Propagator
from daeraycaster import DAERaycaster     

def pick_seed():
    """Returns a seed for a random number generator selected using
    a mixture of the current time and the current process ID."""
    return int(time.time()) ^ (os.getpid() << 16)


class DAEChromaContext(object):
    def __init__(self, config, chroma_geometry, seed=None ):
        log.info("DAEChroma init, CUDA_PROFILE %s " % os.environ.get('CUDA_PROFILE',"not-defined") )
        self.config = config
        self.raycaster = None
        self.propagator = None

        self.deviceid = config.args.deviceid 
        self.nthreads_per_block = config.args.threads_per_block
        self.max_blocks = config.args.max_blocks
        #
        # doing gpu_create_cuda_context as well 
        # as "import pycuda.gl.autoinit" leads to PyCUDA error at exit
        # self.context = gpu.create_cuda_context(self.deviceid)  
        #
        self.context = None

        self.seed = pick_seed() if seed is None else seed
        np.random.seed(self.seed)
        self.rng_states = gpu.get_rng_states(self.nthreads_per_block*self.max_blocks, seed=self.seed)

        self.gpu_geometry = GPUGeometry( chroma_geometry )
        self.raycaster = DAERaycaster( self )
        self.propagator = Propagator( self )

    def step(self, photons, max_steps=1):
        """
        :return: chroma.event.Photons instance
        """
        return self.propagator.propagate( photons, max_steps=max_steps )

    def __del__(self):
        log.info("DAEChromaContext.__del__ not popping context")
        #self.context.pop()




if __name__ == '__main__':
    pass

