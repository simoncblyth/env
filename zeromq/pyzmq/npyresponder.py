#!/usr/bin/env python
"""
NPYResponder
==============

Amalgamation of CPLResponder and ZMQRootResponder
shifting to NPY serialized transport rather than 
ChromaPhotonList (ROOT TObject serialization)
This means can eliminate ROOT dependency.

"""
import logging, os, io, time, errno, codecs
import json, pprint
import numpy as np
import zmq 
log = logging.getLogger(__name__)

from npycontext import NPYContext


class NPYResponder(object):
    """
    Subclasses need to impleent a reply method
    that accepts and returns an obj of the transported class 
    """
    def __init__(self, config): 
        context = NPYContext()
        socket = context.socket(zmq.REP)

        if config.mode == 'bind':
            log.info("bind to endpoint [%s] (server-like)" % config.endpoint )
            socket.bind( config.endpoint )
        elif config.mode == 'connect':
            log.debug("connect to endpoint [%s] (worker-like) " % config.endpoint )
            socket.connect( config.endpoint )
        else:
            log.fatal("mode must be bind or connect not [%s][%s] " % (config.mode,config.endpoint) )
            assert 0, config.mode
            pass
        pass

        config.flags = zmq.POLLIN

        self.socket = socket
        self.config = config
        log.debug("socket mode: %s to %s hwm %s timeout %s sleep %s " % ( config.mode, config.endpoint, socket.hwm, config.timeout, config.sleep ))

    def __repr__(self):
        return "%s %s %s " % (self.__class__.__name__, self.config.mode, self.config.endpoint )

    def poll(self):
        """
        https://github.com/zeromq/pyzmq/issues/348
        https://gist.github.com/minrk/5258909
        """
        events = None
        try:
            events = self.socket.poll(timeout=self.config.timeout, flags=self.config.flags )
        except zmq.ZMQError as e:
            if e.errno == errno.EINTR:
                log.debug("got zmq.ZMQError : Interrupted System Call, return to poll again sometime, resizing terminal windowtriggers this")
                return 
            else:
                raise
            pass
        if events:
            request = self.socket.recv_npy(copy=False)

            if hasattr(request, 'meta'):
                try:
                    meta = map(lambda _:json.loads(_), request.meta )
                except ValueError:
                    log.warn("JSON load error for %s " % repr(request.meta))
                    meta = []
                pass 
                request.meta = meta
                #log.info("NPYResponder converting request.meta to dict %s " % pprint.pformat(request.meta, width=20) )
            pass

            response = self.reply(request) 

            if hasattr(response, 'meta'):
                meta = map(lambda _:json.dumps(_), response.meta )
                response.meta = meta 
                #log.info("NPYResponder converting response.meta to dict %s " % pprint.pformat(response.meta, width=20) )
            pass

            self.socket.send_npy(response)    

    def reply(self, obj):
        log.info("reply : override this in object specific subclass") 
        assert 0
        return obj



class ExamplePhotonListResponder(NPYResponder):
    def __init__(self, config):
        NPYResponder.__init__(self, config)
        
    def reply(self, a ):
        log.info("ExamplePhotonListResponder mirroring  ")
        if self.config.dump:
            print a
        return a 


def check_npyresponder():
    class Config(object):
        mode = 'connect'
        endpoint = os.environ['ZMQ_BROKER_URL_BACKEND']
        timeout = None # milliseconds  (None means dont timeout, just block)
        sleep = 0.5  # seconds
        dump = True
        random = False
    pass 
    cfg = Config()
    responder = ExamplePhotonListResponder(cfg)
    log.info("polling: %s " % repr(responder))
    responder.poll() 

def main():
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(suppress=True, precision=3)
    check_npyresponder()

if __name__ == '__main__':
    main()


