#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

#from env.chroma.ChromaPhotonList.cpl_responder import CPLResponder
from env.zeromq.pyzmq.npyresponder import NPYResponder

class DAEDirectResponder(NPYResponder):
    """
    Receives and replies to messages from C++/Geant4 
    process containing NPY serializations sent from C++ with::

       G4DAESocketBase* socket = new G4DAESocketBase(frontend);
       G4DAEPhotons* photons = G4DAEPhotons::Load("1") ;
       G4DAEPhotons* hits = socket->SendReceiveObject(photons);

    Was formerly based on CPLResponder:

    ZMQRoot/ChromaPhotonList responder
    Allows TObject subclass instances to be received from a remote 
    Geant4 application that invokes::

        fZMQRoot->SendObject(fPhotonList)

    """
    def __init__(self, config, handler ):
        class Cfg(object):
            mode = 'connect' # 'connect' for worker, 'bind' for server
            endpoint = config.args.zmqendpoint
            timeout = 100  # millisecond
            sleep = 0.5
        pass
        cfg = Cfg()
        NPYResponder.__init__(self, cfg )
        self.cfg = cfg 
        self.handler = handler
        log.debug("init %s " % repr(self))

    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__, self.cfg.mode, self.cfg.endpoint )

    def reply(self, request ):
        """
        Overrides base responder .reply method 
        """
        log.info("DAEDirectResponder request %s " % repr(request.shape) )
        response = self.handler( request )
        log.info("DAEDirectResponder response %s " % repr(response.shape) )
        return response 

        

if __name__ == '__main__':
    pass


