#!/usr/bin/env python
# keep this minimalistic, no OpenGL..
import logging, datetime, time
log = logging.getLogger(__name__)
from photons import Photons
import numpy as np

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

class DAEEventBase(object):
    def __init__(self, config, scene ):
        self.config = config
        self.scene = scene

    def external_npl_base(self, npl ):
        if self.config.args.saveall:
            log.info("external_npl timestamp_save due to --saveall option")
            self.timestamped_save(npl)
            key = None
            self.config.save_npl( timestamp(), key, npl )   
        else:
            log.info("external_npl not saving ")
        pass

        log.info("external_npl_base sleeping for 3 seconds ")
        time.sleep(3) 
        log.info("external_npl_base after sleeping for 3 seconds ")

        self.setup_npl(npl) 
 
    def external_cpl_base(self, cpl ):
        """
        :param cpl: ChromaPhotonList instance

        External ZMQ messages containing CPL arrive at DAEResponder and are routed here
        via glumpy event system.

        TODO: check that response is sent with the propagated photons
        """
        if self.config.args.saveall:
            log.info("external_cpl timestamp_save due to --saveall option")
            key = None
            self.config.save_cpl( timestamp(), key, cpl.cpl)   
        else:
            log.info("external_cpl not saving ")
        pass
        self.setup_cpl(cpl) 

    def setup_cpl(self, cpl):
        """
        :param cpl: ChromaPhotonList instance

        Convert serialization level ChromaPhotonList into operation level Photons
        """
        photons = Photons.from_cpl(cpl, extend=True)   
        self.setup_photons( photons ) 

    def setup_npl(self, npl):
        """
        :param npl: NPY deserialized array of shape (nphoton,4,4)

        """
        photons = Photons.from_npl(npl, extend=True)   
        self.setup_photons( photons ) 


    def setup_photons_base( self, photons ):
        log.info("setup_photons_base")
        self.dphotons.photons = photons   ## this setter triggers propagation  

 



