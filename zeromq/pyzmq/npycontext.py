#!/usr/bin/env python
"""
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



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    evt = "1"
    path = os.environ['DAECERENKOV_PATH_TEMPLATE'] % evt
    a = np.load(path)
    log.info("loaded %s %s\n%s " % (path, str(a.shape), repr(a) ))    

    endpoint = os.environ['ZMQ_BROKER_URL_FRONTEND']

    context = NPYContext()
    socket = context.socket(zmq.REQ)

    log.info("connect to endpoint %s " % endpoint ) 
    socket.connect(endpoint)

    socket.send_npy(a)




