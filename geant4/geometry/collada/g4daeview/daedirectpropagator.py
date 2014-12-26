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
from chroma.gpu.gensteps import GPUGenSteps
from chroma.gpu.geometry import GPUGeometry

class NPY(np.ndarray):
    pass
    @classmethod
    def empty(cls):
        a = np.array((), dtype=np.float32)
        return a.view(cls)



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
        if self.chroma.ctrl.get('onlycopy',0) == 1:
            self.onlycopy(request)
            response = NPY.empty()
            results = {}
        elif itemshape == ():
            log.warn("empty itemshape received %s " % str(itemshape))
            response = NPY.empty()
            results = {}
            extra = True
        elif itemshape == (6,4):
            response, results = self.generate(request)
        elif itemshape == (4,4):
            response, results = self.propagate(request)
        else:
            log.warn("itemshape %s not recognized " % str(itemshape))
            response = NPY.empty()
            results = {}
        pass
        return self.chroma.outgoing(response, results, extra=extra)


    def onlycopy(self, request):
        itype = self.chroma.ctrl.get('type',None)
        evt = self.chroma.ctrl.get('evt',None)
        if not itype is None:
            self.config.save_npy(request, evt, itype)   
        else:
            log.warn("failed to onlycopy")
        pass

    def sidesave(self, request, photons):
        itype = self.chroma.ctrl.get('type',None)
        evt = self.chroma.ctrl.get('evt',None)
        if not itype is None:
            self.config.save_npy(request, evt, itype)   
            otype = "op%s" % itype
            self.config.save_npy(photons, evt, otype)   
        else:
            log.warn("failed to sidesave")
        pass


    def generate(self, request):
        """
        ::

            In [98]: gensteps[0:100,0,3].view(np.int32).sum()
            Out[98]: 4711

        """


        results = {}
        gpu_gensteps = GPUGenSteps(request)

        results = gpu_gensteps.generate(self.chroma.gpu_detector, 
                                        self.chroma.rng_states,
                                        self.chroma.parameters)
        photons = gpu_gensteps.get()


        if self.chroma.ctrl.get('sidesave',0) == 1:
            self.sidesave(request, photons)

        if self.chroma.ctrl.get('noreturn',0) == 1:
            response = NPY.empty()
        else:
            response = photons
        pass
        return response, results


    def propagate(self, request):
        """
        :param request: np.ndarray (or NPY subclass) [formerly ChromaPhotonList]
        :return response: same type as request, but propagated 

        This method is invoked by the DAEDirectResponder handler, 
        which specializes NPYResponder by implementing reply method 
        which does::

           response = self.handler( request )

        TODO: simplify marshalling to avoid going via chroma.event.Photons 
        TODO: move most of this into DAEChromaContext, because thats common between the
              two flavors of propagation
        """

        photons = Photons.from_obj( request, extend=False) # TODO: short circuit this, moving to NPL

        gpu_photons = GPUPhotonsHit(photons)        

        results = gpu_photons.propagate_hit(self.chroma.gpu_detector, 
                                            self.chroma.rng_states,
                                            self.chroma.parameters)

        # pycuda get()s from GPU back into ndarrays and creates NPL, formerly event.Photon instance
        photons_end = gpu_photons.get(npl=1,hit=self.chroma.parameters['hit'])

        return photons_end, results


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


    


