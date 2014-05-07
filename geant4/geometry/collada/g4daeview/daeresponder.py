#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

from glumpy.window import event
from env.chroma.ChromaPhotonList.responder import CPLResponder


class DAEResponder(event.EventDispatcher, CPLResponder):
    """
    http://www.pyglet.org/doc/programming_guide/creating_your_own_event_dispatcher.html
    """
    def __init__(self, config):
        class Cfg(object):
            bind = config.args.zmqbind
            timeout = 100  # millisecond
            sleep = 0.5
            random = False
            dump = True
        pass
        cfg = Cfg()
        CPLResponder.__init__(self, cfg )
        self.cfg = cfg 

    def __repr__(self):
        return "%s %s " % ( self.__class__.__name__, self.cfg.bind )

    def update(self):
        self.poll()

    def reply(self, cpl ):
        log.info("responder reply %s " % repr(cpl) )
        self.dispatch_event('on_external_cpl', cpl)
        return cpl

    def on_external_cpl(self, cpl):
        pass
        log.info("default on_external_cpl %s " % cpl )   # huh both this 


DAEResponder.register_event_type('on_external_cpl')



