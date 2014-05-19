#!/usr/bin/env python
"""
Simplification of chroma.sim.Simulation and bin/chroma-server
that just does propagation.

TODO:

#. kernel launch timings
#. convert to prepared launch

"""
import logging
log = logging.getLogger(__name__)

from chroma.gpu.photon import GPUPhotons


class Propagator(object):
    def __init__(self, ctx ):
        self.ctx = ctx

    def propagate(self, photons, max_steps=100):
        """
        :param photons: 

        Reuse gpu_photons allocation ? with eye to ease of OpenGL/CUDA interop mapping/unmapping

        """
        gpu_photons = GPUPhotons(photons)
        gpu_photons.propagate(self.ctx.gpu_geometry, 
                              self.ctx.rng_states,
                              nthreads_per_block=self.ctx.nthreads_per_block,
                              max_blocks=self.ctx.max_blocks,
                              max_steps=max_steps)

        photons_end = gpu_photons.get()
        return photons_end 

if __name__ == '__main__':
    pass


