#!/usr/bin/env python
# keep this minimalistic, no OpenGL..
import logging, datetime
log = logging.getLogger(__name__)
from photons import Photons
import numpy as np

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

class DAEEventBase(object):
    def __init__(self, config, scene ):
        self.config = config
        self.scene = scene

    def external_cpl_base(self, cpl ):
        """
        :param cpl: ChromaPhotonList instance

        External ZMQ messages containing CPL arrive at DAEResponder and are routed here
        via glumpy event system.

        TODO: check that response is sent with the propagated photons
        """
        if self.config.args.saveall:
            log.info("external_cpl timestamp_save due to --saveall option")
            self.timestamped_save(cpl)
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

    def setup_photons_base( self, photons ):
        log.info("setup_photons_base")
        self.dphotons.photons = photons   ## this setter triggers propagation  

    def timestamped_save(self, cpl, key=None):
        path_ = timestamp()
        self.config.save_cpl( path_, key, cpl.cpl)   
 



