#!/usr/bin/env python
"""
Plain ZMQRoot responder and Chroma propagator with no OpenGL/glumpy   
"""
import logging, os, time
log = logging.getLogger(__name__)

import numpy as np

from daedirectconfig import DAEDirectConfig
from daegeometry import DAEGeometry
from daedirectresponder import DAEDirectResponder
from daedirectpropagator import DAEDirectPropagator

class DAEChromaContextDummy(object):
    raycaster = None
    propagator = None
    dummy = True


class G4DAEChroma(object):
    """
    Not a graphical scene, just following the structure of g4daeview.py for sanity 
    """
    def __init__(self, geometry, config ):
        """
        :param geometry: DAEGeometry instance
        :param config: DAEConfig instance
        """
        self.geometry = geometry  
        self.config = config

        if self.config.args.with_chroma:
            from daechromacontext import DAEChromaContext     
            chroma_geometry = geometry.make_chroma_geometry() 
            chroma = DAEChromaContext( config, chroma_geometry, gl=0)
        else:
            chroma = DAEChromaContextDummy()
        pass

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
    config = DAEDirectConfig(__doc__)
    config.parse()
    np.set_printoptions(precision=3, suppress=True)
    log.info("***** post DAEDirectConfig.parse")

    geometry = DAEGeometry.get(config) 
    log.info("***** post DAEGeometry.get")

    gdc = G4DAEChroma(geometry, config )
    log.info("***** post G4DAEChroma ctor")
    gdc.poll_forever()


if __name__ == '__main__':
    main()




