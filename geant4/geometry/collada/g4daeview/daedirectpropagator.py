#!/usr/bin/env python
"""
Test Usage::

    daedirectpropagator.sh


"""
import logging
log = logging.getLogger(__name__)


import numpy as np
import IPython as IP

from env.g4dae.types import ChromaPhoton, Photon, G4Step, NPY

import pycuda.driver as cuda_driver
import pycuda.gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3
from chroma.gpu.photon_hit import GPUPhotonsHit
from chroma.gpu.gensteps import GPUGenSteps
from chroma.gpu.geometry import GPUGeometry




class DAEDirectPropagator(object):
    def __init__(self, config, chroma):
        """
        :param config:
        :param chroma: DAEChromaContext instance 
        """
        self.config = config
        self.chroma = chroma

    def incoming(self, request):
        """
        Branch handling based on itemshape (excluding first dimension) 
        of the request array 
        """
        self.chroma.incoming(request)  # do any config contained in request
        itemshape = request.shape[1:]
        log.info("incoming itemshape %s " % repr(itemshape))
        extra = False 
        results = {}
        response = NPY.empty()

        if self.chroma.ctrl.get('onlycopy',0) == 1:

            self.onlycopy(request)

        elif itemshape == ():

            log.warn("empty itemshape received %s " % str(itemshape))
            extra = True

        elif itemshape == (6,4):

            #response, results = self.generate(request)
            response, results = self.generate_and_propagate(request)

        elif itemshape == (4,4):

            response, results = self.propagate(request)

        else:

            log.warn("itemshape %s not recognized " % str(itemshape))


        if self.chroma.ctrl.get('noreturn',0) == 1:
            response = NPY.empty()

        return self.chroma.outgoing(response, results, extra=extra)


    def onlycopy(self, request):
        itype = self.chroma.ctrl.get('type',None)
        evt = self.chroma.ctrl.get('evt',None)
        if not itype is None:
            self.config.save_npy(request, evt, itype)   
        else:
            log.warn("failed to onlycopy")
        pass


    def save(self, npy, prefix=None):
        """
        Better to keep type info and metadata with the npy 
        rather than using common place in chroma 
        """
        itype = self.chroma.ctrl.get('type',None)
        evt = self.chroma.ctrl.get('evt',None)
        if itype is None or evt is None:    
            log.warn("failed to save, missing type/evt metadata")
   
        utype = itype if prefix is None else "%s%s" % (prefix, itype)
        self.config.save_npy(npy, evt, utype)   

    def generate_and_propagate(self, request):
        log.info("generate_and_propagate %s " % repr(request.shape))
        gpu_photons = GPUPhotonsHit(gensteps=request)        

        results = gpu_photons.propagate_hit(self.chroma.gpu_detector, 
                                            self.chroma.rng_states,
                                            self.chroma.parameters)

        hit = self.chroma.parameters['hit']
        pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights = gpu_photons.get()
        photons = ChromaPhoton.from_arrays(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights, hit=hit)

        return photons, results

    def generate(self, request):
        """
        """
        if self.chroma.ctrl.get('sidesave',0) == 1:
            self.save(request)

        gpu_gensteps = GPUGenSteps(request)

        results = gpu_gensteps.generate(self.chroma.gpu_detector, 
                                        self.chroma.rng_states,
                                        self.chroma.parameters)

        hit = 0 
        pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights = gpu_gensteps.get()
        photons = ChromaPhoton.from_arrays(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights, hit=hit)

        if self.chroma.ctrl.get('sidesave',0) == 1:
            self.save(photons, prefix='op')

        return photons, results


    def propagate(self, request):
        """
        :param request: np.ndarray (or NPY subclass) [formerly ChromaPhotonList]
        :return response: same type as request, but propagated 

        This method is invoked by the DAEDirectResponder handler, 
        which specializes NPYResponder by implementing reply method 
        which does::

           response = self.handler( request )

        """
        photons = ChromaPhoton.from_array(request)

        gpu_photons = GPUPhotonsHit(photons=photons)        

        results = gpu_photons.propagate_hit(self.chroma.gpu_detector, 
                                            self.chroma.rng_states,
                                            self.chroma.parameters)

        hit = self.chroma.parameters['hit']
        pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights = gpu_photons.get()
        photons = ChromaPhoton.from_arrays(pos, dir, pol, wavelengths, t, last_hit_triangles, flags, weights, hit=hit)

        return photons, results


def main():
    """
    Debugging CPL and Photons handling/conversions and propagation
    of canned photons from event "1" 

    #. loads persisted CPL
    #. converts into `photons` chroma.event.Photons (fallback photons.Photons)
    #. runs chroma propagate_hit kernel
    #. creates new CPL from the propagated `photons`

    DONE:

    #. move to prepared timed kernel call
    #. reproducibility check on propagation

       * OK from quick check of getting same digest on multiple runs

    #. getting hit pmtids reported

       * photons has last_hit_triangles that CPL misses

    TODO: 

    #. propagating channel_id gleaned into output photons structure   

    #. check roundtripping with hit formation in StackAction
 
       * how to handle ProcessHits detector element transforms ? 
         presumably need to cache the transforms somehow

    HOLD:

    #. look into material/surface/process map, why the index variability ? 

       * workaround for this is writing the json maps at every geometry creation
         but would be better to avoid the variability

    """
    from daedirectconfig import DAEDirectConfig
    config = DAEDirectConfig(__doc__)
    config.parse()
    assert config.args.with_chroma

    load = config.args.clargs[0]
    log.info("load:%s" % load )   # eg mock001

    from daegeometry import DAEGeometry 
    geometry = DAEGeometry.get(config) 
    chroma_geometry = geometry.make_chroma_geometry() 

    from daechromacontext import DAEChromaContext     
    chroma = DAEChromaContext( config, chroma_geometry )

    propagator = DAEDirectPropagator(config, chroma)

    request, = NPY.mget(1,"opcerenkov") 

    response, results = propagator.propagate(request) 

    lht = response.last_hit_triangles
    flg = response.flags

    assert len(lht) == len(flg)
    SURFACE_DETECT = 0x1 << 2
    detected = np.where( flg & SURFACE_DETECT  )

    # when mis-using lht to output surface index, this worked   
    #assert np.all( lht[detected] == geometry.chroma_surface_map.shortname2code['PmtHemiCathode'] )

    #for solid_index in lht[detected]:
    #    chroma_solid = chroma_geometry.solids[solid_index]
    #    node = chroma_solid.node
    #    print "0x%7x  %s " % (node.channel_id, node )

    for channel_id in lht[detected]:
        print "0x%7x " % (channel_id )




    IP.embed()

if __name__ == '__main__':
    main()


    


