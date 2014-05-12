#!/usr/bin/env python
"""


"""
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

        if config.mode == 'bind':
            log.info("bind to endpoint [%s] (server-like)" % config.endpoint )
            socket.bind( config.endpoint )
        elif config.mode == 'connect':
            log.info("connect to endpoint [%s] (worker-like) " % config.endpoint )
            socket.connect( config.endpoint )
        else:
            log.fatal("mode must be bind or connect not [%s][%s] " % (config.mode,config.endpoint) )
            assert 0, config.mode
            pass
        pass

        config.flags = zmq.POLLIN

        self.socket = socket
        self.config = config

        log.info("socket mode: %s to %s hwm %s timeout %s sleep %s " % ( config.mode, config.endpoint, socket.hwm, config.timeout, config.sleep ))

    def __repr__(self):
        return "%s %s %s " % (self.__class__.__name__, self.config.mode, self.config.endpoint )

    def poll(self):
        events = self.socket.poll(timeout=self.config.timeout, flags=self.config.flags )
        if events:
            req = self.recv_object()
            rep = self.reply( req ) 
            time.sleep(self.config.sleep)
            self.send_object(rep)    

    def recv_object(self):
        """
        """
        req = self.socket.recv(copy=False)
        log.info("recv_object req of length %s %s " % (len(req), repr(req)))

        # suspect this leads to files that are unreadable by zmsg_load
        #
        #if not self.config.filename is None:
        #    filename = self.config.filename
        #    log.info("writing message to file %s " % filename )
        #    with open(filename, 'wb') as f:
        #        f.write(req.bytes)
        #    pass
        pass
        obj = deserialize( req.bytes )
        return obj 

    def send_object(self, obj):
        rep = serialize(obj)  
        log.info("send_object rep of length %s %s " % (len(rep),repr(rep)))
        self.socket.send(rep)

    def reply(self, obj):
        log.info("reply : override this in object specific subclass") 
        return obj



def check_responder():
    """
    Deserialization will fail if dictionary not found::

        INFO:__main__:polling: ZMQRootResponder connect tcp://203.64.184.126:5002 /tmp/lastmsg.zmq 
        INFO:__main__:recv_object req of length 457 <zmq.backend.cython.message.Frame object at 0x10350b680> 
        INFO:__main__:writing message to file /tmp/lastmsg.zmq 
        Error in <TClass::Load>: dictionary of class ChromaPhotonList not found
        Error in <TClass::Load>: dictionary of class ChromaPhotonList not found
        INFO:__main__:reply : override this in object specific subclass
        INFO:__main__:send_object rep of length 12 <serialize.c_char_Array_12 object at 0x10350b7a0> 

    The dud object returned causes segmentation violation for the client::

        ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://203.64.184.126:5001] 
        ChromaPhotonList::Print  [6]
        ZMQRoot::SendObject sent bytes: 457 
        ZMQRoot::ReceiveObject received bytes: 12 
        ZMQRoot::ReceiveObject reading TObject from the TMessage 
        ZMQRoot::ReceiveObject returning TObject 
        ReceiveObject

         *** Break *** segmentation violation

    Conclusion, do the check at CPL level ? So a valid CPL can be returned.
    """
    class Config(object):
        mode = 'connect'
        endpoint = os.environ['ZMQ_BROKER_URL_BACKEND']
        #filename = '/tmp/lastmsg.zmq'
        timeout = None # milliseconds  (None means dont timeout, just block)
        sleep = 0.5  # seconds
    pass
    cfg = Config()
    responder = ZMQRootResponder(cfg)
    log.info("polling: %s " % repr(responder))
    responder.poll()

def main():
    import os
    logging.basicConfig(level=logging.INFO)
    check_responder()

if __name__ == '__main__':
    pass

