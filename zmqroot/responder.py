#!/usr/bin/env python

import logging, time
log = logging.getLogger(__name__)

import zmq     
from serialize import serialize, deserialize


class ZMQRootResponder(object):
    """
    Subclasses need to impleent a reply method
    that accepts and returns an obj of the transported class 
    """
    def __init__(self, config): 
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind( config.bind )
        config.flags = zmq.POLLIN

        self.socket = socket
        self.config = config

        log.info("bound to %s hwm %s timeout %s sleep %s " % ( config.bind, socket.hwm, config.timeout, config.sleep ))

    def poll(self):
        events = self.socket.poll(timeout=self.config.timeout, flags=self.config.flags )
        if events:
            req = self.recv_object()
            rep = self.reply( req ) 
            time.sleep(self.config.sleep)
            self.send_object(rep)    

    def recv_object(self):
        req = self.socket.recv(copy=False)
        log.info("recv_object req of length %s %s " % (len(req), repr(req)))
        obj = deserialize( req.bytes )
        return obj 

    def send_object(self, obj):
        rep = serialize(obj)  
        log.info("send_object rep of length %s %s " % (len(rep),repr(rep)))
        self.socket.send(rep)

    def reply(self, obj):
        log.info("reply : override this in object specific subclass") 
        return obj




if __name__ == '__main__':
    pass


