#!/usr/bin/env python
"""
Plain ZMQRoot responder and Chroma propagator with no OpenGL/glumpy   
"""
import logging, os, time
import IPython

log = logging.getLogger(__name__)

import numpy as np

from daedirectconfig import DAEDirectConfig
from daedirectresponder import DAEDirectResponder
from daedirectpropagator import DAEDirectPropagator
from daechromacontext import DAEChromaContext     

from chroma.detector import Detector

class G4DAEChroma(object):
    """
    Not a graphical scene, just following the structure of g4daeview.py for sanity 
    """
    def __init__(self, chroma_geometry, config ):
        """
        :param chroma_geometry: chroma.Detector or chroma.Geometry instance
        :param config: DAEConfig instance
        """
        self.chroma_geometry = chroma_geometry  
        self.config = config

        chroma = DAEChromaContext( config, chroma_geometry, gl=0)

        propagator = DAEDirectPropagator(config, chroma)
        def handler(obj):
            log.info("handler got obj (cpl or npl)") 
            return propagator.propagate( obj )

        self.responder = DAEDirectResponder(config, handler )
        self.propagator = propagator

    def poll_forever(self):
        log.info("start polling responder: %s " % repr(self.responder))
        count = 0 
        while True:
            if count % 10 == 0:
                log.info("polling %s " % count ) 
            self.responder.poll()
            count += 1 
            time.sleep(1) 
        pass
        log.info("terminating")

 

def main():
    """
    Note that there is no need for the DAEGeometry when running from 
    cache, when not visualizing it is just an intermediate object 
    needed to parse collada and create the chroma_geometry  
    """
    config = DAEDirectConfig(__doc__)
    config.parse()
    assert config.args.with_chroma
    np.set_printoptions(precision=3, suppress=True)
    log.info("***** g4daechroma start")

    if config.args.geocacheupdate:
        config.wipe_geocache()

    chroma_geometry = None 
    if config.args.geocache:
        chroma_geometry = Detector.get(config.chromacachepath)  
    
    if chroma_geometry is None:
        from daegeometry import DAEGeometry
        geometry = DAEGeometry.get(config) 
        log.info("***** post DAEGeometry.get")
        chroma_geometry = geometry.make_chroma_geometry() 
        log.info("***** post make_chroma_geometry ")
        if config.args.geocache or config.args.geocacheupdate:
            log.info("as --geocache/--geocacheupdate : saving chroma_geometry to %s " % config.chromacachepath )
            chroma_geometry.save(config.chromacachepath)
        pass
    pass

    if config.args.ipython:
        IPython.embed()

    gdc = G4DAEChroma(chroma_geometry, config )
    log.info("***** post G4DAEChroma ctor")
    gdc.poll_forever()


if __name__ == '__main__':
    main()




