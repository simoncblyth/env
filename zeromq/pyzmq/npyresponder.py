#!/usr/bin/env python
"""
NPYResponder
==============

Amalgamation of CPLResponder and ZMQRootResponder
shifting to NPY serialized transport rather than 
ChromaPhotonList (ROOT TObject serialization)
This means can eliminate ROOT dependency.

"""
import logging, os, io, time
import numpy as np
import zmq 
log = logging.getLogger(__name__)

class NPYSocket(zmq.Socket):
    """
    Based on http://stackoverflow.com/questions/24483597/pass-numpy-array-without-copy-using-pyzmq
    which uses multipart zmq messages and a json metadata header
    The problem with that approach is that difficult to use with C/C++ at the other end
    of the socket.
    Also means cannot use np.getbuffer np.frombuffer ?

    http://stackoverflow.com/questions/12462547/numpy-getbuffer-and-numpy-frombuffer
    """
    def send_npy(self, a, flags=0, copy=True, track=False):
        """
        NPY serialize to a buffer and then send

        Probably this is doing three copies:

        #. array to stream (inevitable as want the NPY serialization)
        #. stream to buf
        #. buf into zmq internals
      
        Perhaps streams might provide access to their internal 
        buffers ? Which would allow to get rid of one copy.

        http://python.6.x6.nabble.com/Buffer-protocol-for-io-BytesIO-td1902991.html        
        """ 
        stream = io.BytesIO()
        np.save( stream, a )
        stream.seek(0)
        buf = buffer(stream.read())
        return self.send( buf, flags=flags, copy=copy, track=track)

    def recv_npy(self, flags=0, copy=True, track=False):
        if copy:
            msg = self.recv(flags=flags, copy=copy, track=track)  # bytes
            buf = buffer(msg)
        else:
            frame = self.recv(flags=flags,copy=False, track=track)    
            buf = frame.buffer            # memoryview object 
        pass
        stream = io.BytesIO(buf)
        a = np.load(stream)
        return a 

class NPYContext(zmq.Context):
    _socket_class = NPYSocket



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
                log.debug("got zmq.ZMQError : Interrupted System Call, return to poll again sometime")
                return 
            else:
                raise
            pass
        if events:
            request = self.socket.recv_npy(copy=False)
            response = self.reply(request) 
            time.sleep(self.config.sleep)
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


