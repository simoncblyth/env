#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

import numpy as np
from glumpy.window import event

#from env.chroma.ChromaPhotonList.cpl_responder import CPLResponder
from env.zeromq.pyzmq.npyresponder import NPYResponder


class DAEResponder(event.EventDispatcher, NPYResponder):
    """
    Glue between transport mechanism and glumpy event dispatch

    #. polls for messages on the transport socket 
    #. when messages arrive that contain serialized 
       photon lists pass these along using the pyglet 
       inspired glumpy event dispatch 

    Connection to glumpy event mechanism:

    * `DAEInteractivityHandler.hookup_zmq_responder` 
      instantiates this responder and configures a timer 
      which fires several times a second:

    * timer invokes `responder.update()` 
    * update invokes base class `.poll()` checking for new messages

    Transport is initiated at the Geant4/C++ end 

    #. ZMQRoot/ChromaPhotonList  
 
       * TObject serialization of ChromPhotonList instances, sent with::

            fZMQRoot->SendObject(fPhotonList)

    #. G4DAESocket<G4DAEPhotonList> NPY (numpy native serialization)

            G4DAEPhotonList* photon_req = ..
            socket = new G4DAESocket<G4DAEPhotonList>(frontend)
            socket->SendObject( photon_req )
            G4DAEPhotonList* photon_rep = socket->ReceiveObject()

    See also:

    #. LXe example `lxe-`
    #. http://www.pyglet.org/doc/programming_guide/creating_your_own_event_dispatcher.html

    """
    def __init__(self, config, scene):
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
        self.scene = scene   # here or on base class ?
        log.info("init %s " % repr(self))

    def __repr__(self):
        return "%s %s %s live:%s" % ( self.__class__.__name__, self.cfg.mode, self.cfg.endpoint, self.live )

    def update(self):
        if not self.live:
            log.info("DAEResponder.update : not live, no polling")
            return
        #log.info("DAEResponder.update calling poll")
        self.poll()

    def reply(self, request ):
        """
        :param request: photon data instance
        :return response: propagated photon data with hit selection applied

        Photon list instances are both of the same type, either:

        #. np.ndarray shaped (nphotons, 4, 4) : deserialized from NPY transport using numpy itself  
        #. ChromaPhotonList : deserialized from ROOT/TObject with ZMQRoot 

        TODO: this needs to return the propagated 

        #. how to access the response pulled of the GPU indirectly ? this is kinda decoupled  
        #. some kinda singleton approach for last propagation ? 

        """
        log.info("DAEResponder.reply : request %s  " % repr(request) )
        self.dispatch_event(self.config.handler, request)   # results in setting the photons and propagating 
        log.info("DAEResponder.reply : after dispatch_event with handler %s completes " % self.config.handler )

        r = self.scene.event.dphotons.lastpropagated
        log.info("DAEResponder.reply : response \n%s\n%s\n%s  " % (r,r.shape,r.dtype.descr) )
        return r

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



