#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from glumpy.window import event
from env.chroma.ChromaPhotonList.cpl_responder import CPLResponder


class DAEResponder(event.EventDispatcher, CPLResponder):
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
        pass
        cfg = Cfg()
        CPLResponder.__init__(self, cfg )
        self.cfg = cfg 
        self.live = config.args.live
        log.debug("init %s " % repr(self))

    def __repr__(self):
        return "%s %s %s live:%s" % ( self.__class__.__name__, self.cfg.mode, self.cfg.endpoint, self.live )

    def update(self):
        #log.info("DAEResponder update calling poll")
        if not self.live:return
        self.poll()

    def reply(self, cpl ):
        """
        Overrides CPLResponder.reply method that dumps and creates random CPL
        Instead of doing that just pass the cpl to the scene.
        """
        log.info("responder reply %s " % repr(cpl) )
        self.dispatch_event('on_external_cpl', cpl)
        return cpl

    def on_external_cpl(self, cpl):
        """
        To prevent this being called ensure that the other handler returns True
        """
        log.info("default on_external_cpl %s " % cpl )    


DAEResponder.register_event_type('on_external_cpl')



