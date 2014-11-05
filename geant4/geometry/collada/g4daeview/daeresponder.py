#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import numpy as np

from glumpy.window import event

#from env.chroma.ChromaPhotonList.cpl_responder import CPLResponder
from env.zeromq.pyzmq.npyresponder import NPYResponder


class DAEResponder(event.EventDispatcher, NPYResponder):
    """
    Glue between ZMQRoot/ChromaPhotonList and glumpy event dispatch. 
    Allows TObject subclass instances to be received from a remote 
    Geant4 application that invokes::

        fZMQRoot->SendObject(fPhotonList)

    See the LXe example `lxe-`

    http://www.pyglet.org/doc/programming_guide/creating_your_own_event_dispatcher.html
    """
    def __init__(self, config):
        class Cfg(object):
            mode = 'connect' # worker, as opposed to 'bind' for server
            endpoint = config.args.zmqendpoint
            timeout = 100  # millisecond
            sleep = 0.5
            handler = 'on_external_npl'
            #handler = 'on_external_cpl'
        pass
        cfg = Cfg()
        NPYResponder.__init__(self, cfg )
        self.cfg = cfg 
        self.live = config.args.live
        log.info("init %s " % repr(self))

    def __repr__(self):
        return "%s %s %s live:%s" % ( self.__class__.__name__, self.cfg.mode, self.cfg.endpoint, self.live )

    def update(self):
        if not self.live:
            log.info("DAEResponder.update : not live, no polling")
            return
        log.info("DAEResponder.update calling poll")
        self.poll()

    def reply(self, request ):
        """
        TODO: this needs to return the propagated 

        #. how to access the response pulled of the GPU indirectly ? this is kinda decoupled  
        #. some kinda singleton approach for last propagation ? 

        """
        log.info("DAEResponder.reply : request %s  " % repr(request) )
        self.dispatch_event(self.config.handler, request)   # results in setting the photons and propagating 
        log.info("DAEResponder.reply : after dispatch_event with handler %s completes " % self.config.handler )
        response = np.arange(10*4*4, dtype=np.float32).reshape(10,4,4) # dummy standin for NPY 
        log.info("DAEResponder.reply : response %s  " % repr(response) )
        return response

    def on_external_cpl(self, cpl):
        """
        To prevent this being called ensure that the other handler returns True
        """
        log.info("default on_external_cpl %s " % cpl )    

    def on_external_npl(self, npl):
        """
        To prevent this being called ensure that the other handler returns True
        """
        log.info("default on_external_npl %s " % npl )    



DAEResponder.register_event_type('on_external_cpl')
DAEResponder.register_event_type('on_external_npl')



