#!/usr/bin/env python
"""
Test Usage::

    daedirectpropagator.sh


"""
import logging
log = logging.getLogger(__name__)


import numpy as np
import IPython as IP
from env.chroma.ChromaPhotonList.cpl import examine_cpl, random_cpl, save_cpl, load_cpl, create_cpl_from_photons_very_slowly
from photons import Photons

import pycuda.driver as cuda_driver
import pycuda.gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3
#from chroma.gpu.photon import GPUPhotons
from chroma.gpu.photon_hit import GPUPhotonsHit
from chroma.gpu.geometry import GPUGeometry

class NPY(np.ndarray):pass


class DAEDirectPropagator(object):
    def __init__(self, config, chroma):
        """
        :param config:
        :param chroma: DAEChromaContext instance 
        """
        self.config = config
        self.chroma = chroma

    def propagate(self, request):
        """
        :param request: np.ndarray (or NPY subclass) [formerly ChromaPhotonList]
        :return response: same type as request, but propagated 

        This method is invoked by the DAEDirectResponder handler, 
        which specializes NPYResponder by implementing reply method 
        which does::

           response = self.handler( request )

        TODO: simplify marshalling to avoid going via chroma.event.Photons 
        """

        ctrl = request.meta[0].get('ctrl',{})
        args = request.meta[0].get('args',{})
        parameters = self.chroma.configure_parameters(ctrl, args, dump=True)

        if parameters['reset_rng_states']:
            log.warn("reset_rng_states")
            #self.chroma.rng_states = None  gpu_seed setter does this
            self.chroma.gpu_seed = parameters['seed']
        pass

        photons = Photons.from_obj( request, extend=True) # TODO: short circuit this, moving to NPL
        self.photons_beg = photons

        gpu_photons = GPUPhotonsHit(photons)        
        gpu_detector = self.chroma.gpu_detector

        results = gpu_photons.propagate_hit(gpu_detector, 
                                            self.chroma.rng_states,
                                            parameters)


        # pycuda get()s from GPU back into ndarrays and creates NPL, formerly event.Photon instance
        photons_end = gpu_photons.get(npl=1,hit=parameters['hit'])
        self.photons_end = photons_end

        metadata = {}
        metadata['parameters'] = parameters
        metadata['results'] = results
        metadata['geometry'] = gpu_detector.metadata
        photons_end.meta = [metadata]

        return photons_end


    def check_unpropagated_roundtrip(self, cpl, extend=False):
        """
        """
        photons = Photons.from_cpl(cpl, extend=extend)  # CPL into chroma.event.Photons OR photons.Photons   
        cpl2 = create_cpl_from_photons_very_slowly(photons) 
        digests = (cpl.GetDigest(),cpl2.GetDigest()) 
        log.info( "digests %s " % repr(digests))

        if not extend:
            assert digests[0] == digests[1], ("Digest mismatch between cpl and cpl2 ", digests)
        else:
            assert digests[0] != digests[1], ("Digest mismatch expected in extend mode", digests)
        pass

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

    cpl_begin = config.load_cpl(load)
    propagator = DAEDirectPropagator(config, chroma)
    #propagator.check_unpropagated_roundtrip(cpl_begin)

    cpl_end = propagator.propagate(cpl_begin) 
    log.info("cpl_begin digest %s " % cpl_begin.GetDigest())
    log.info("cpl_end   digest %s " % cpl_end.GetDigest())

    photons_end = propagator.photons_end 

    lht = photons_end.last_hit_triangles
    flg = photons_end.flags
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


    


