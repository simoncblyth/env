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


class NP(np.ndarray):
    """
    Subclass to enable hanging metadata on the array 

    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """ 
    pass

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
        np.save( stream, a )    # serialize ndarray into stream
        stream.seek(0)
        buf = buffer(stream.read())

        bufs = [buf]
        if hasattr(a, 'meta'):
            for s in a.meta:
                bufs.append(s) 
            pass

        return self.send_multipart( bufs, flags=flags, copy=copy, track=track)


    def recv_npy(self, flags=0, copy=True, track=False, meta_encoding="ascii"):
        """
        When copy=True receive bytes otherwise receive frames


        http://zeromq.github.io/pyzmq/api/zmq.html#frame

        frame.buffer
             A read-only buffer view of the message contents.

        frame.bytes
             The message content as a Python bytes object.
             The first time this property is accessed, a copy of the message contents is made. 
             From then on that same copy of the message is returned.


        TODO:

        * work out how to peek at first byte and check for 0x93 in position 0 
          (signalling NPY serialization) : currently assuming the first frame to 
          to contain the NPY

        """ 
        assert copy == False, "implementation needs rejig when copying "
        if copy:
            msgs = self.recv_multipart(flags=flags, copy=copy, track=track)  # bytes
            bufs = map(lambda msg:buffer(msg),msgs)
        else:
            frames = self.recv_multipart(flags=flags,copy=False, track=track)    
            bufs = map(lambda frame:frame.buffer, frames)            # memoryview object 
        pass
        
        # peek into the memoryview to find NPY serialization, from magic bytes 
        jbuf = -1
        meta = []
        for ibuf,buf in enumerate(bufs):   # the buf are memoryview 
            #print ibuf, buf, len(buf), dir(buf)
            #if buf[0] == b'\x93':  ## IndexError: invalid indexing of 0-dim memory
            #if buf[1:7].tobytes() == b'NUMPY':   ## ditto
            if ibuf == 0:
                jbuf = ibuf
            else:
                meta.append(codecs.decode(buf.tobytes()))
        pass
        assert jbuf > -1, "failed to find NPY serialization in any of the multipart frames" 

        if jbuf > -1:
            stream = io.BytesIO(bufs[jbuf])  # file like access to memory buffer
            a = np.load(stream)
            aa = a.view(NP)       # view as subclass, to enable attaching metadata
        else:
            aa = NP(0)            # nonce 
        pass
        aa.meta = meta
        return aa 

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


