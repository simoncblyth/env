#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from env.chroma.ChromaPhotonList.cpl_responder import CPLResponder

class DAEDirectResponder(CPLResponder):
    """
    ZMQRoot/ChromaPhotonList responder
    Allows TObject subclass instances to be received from a remote 
    Geant4 application that invokes::

        fZMQRoot->SendObject(fPhotonList)

    """
    def __init__(self, config, handler ):
        class Cfg(object):
            mode = 'connect' # worker, as opposed to 'bind' for server
            endpoint = config.args.zmqendpoint
            timeout = 100  # millisecond
            sleep = 0.5
        pass
        cfg = Cfg()
        CPLResponder.__init__(self, cfg )
        self.cfg = cfg 
        self.handler = handler
        log.debug("init %s " % repr(self))

    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__, self.cfg.mode, self.cfg.endpoint )

    def reply(self, cpl ):
        """
        Overrides CPLResponder.reply method that dumps and creates random CPL
        """
        log.info("responder reply %s " % repr(cpl) )
        return self.handler( cpl )

if __name__ == '__main__':
    pass


