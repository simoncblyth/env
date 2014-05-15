#!/usr/bin/env python
"""
Simplification of chroma.sim.Simulation and bin/chroma-server
that just does propagation.

TODO:

#. adopt chroma.event.Photons as DAEChromaPhotonListBase ? to avoid contortions 
#. kernel launch timings
#. convert to prepared launch

#. hmm each call to gpu.GPUGeometry makes its own copy, so with raycaster and propagater get two ??

   * need to pass around a GPU context object ? on which to slot the thangs to avoid duplication  


"""
import time
import os
import numpy as np
import logging
log = logging.getLogger(__name__)

from chroma import gpu

import pycuda.driver as cuda

class Propagator(object):
    def __init__(self, ctx ):
        self.ctx = ctx

    def propagate(self, photons, max_steps=100):
        """
        Reuse gpu_photons allocation ? with eye to ease of OpenGL/CUDA interop mapping/unmapping
        """
        gpu_photons = gpu.GPUPhotons(photons)
        gpu_photons.propagate(self.ctx.gpu_geometry, 
                              self.ctx.rng_states,
                              nthreads_per_block=ctx.self.nthreads_per_block,
                              max_blocks=self.ctx.max_blocks,
                              max_steps=max_steps)

        photons_end = gpu_photons.get()
        return photons_end 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")

    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl
    from env.geant4.geometry.collada.g4daeview.daechromaphotonlistbase import DAEChromaPhotonListBase
    from env.geant4.geometry.collada.g4daeview.daeconfig import DAEConfig
    from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry

    config = DAEConfig(__doc__)
    config.init_parse()
    config.report()

    config.args.with_chroma = True 

    geometry = DAEGeometry(config.args.geometry, config)
    geometry.flatten()
    chroma_geometry = geometry.make_chroma_geometry() 

    path = config.resolve_event_path("1")
    cpl = load_cpl(path,config.args.key)

    photons = DAEChromaPhotonListBase(cpl, chroma=config.args.with_chroma)
    DAEChromaPhotonListBase.dump_(photons)  # contortion 
 
    propagator = Propagator(chroma_geometry, config )
    photons2 = propagator.propagate( photons, max_steps=1 )    # chroma.event.Photons instance

    DAEChromaPhotonListBase.dump_(photons2)  # contortion





