#!/usr/bin/env python
"""
"""
import logging, os, io, time, errno, codecs
import json, pprint
import numpy as np
import zmq 
import IPython
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
    def send_npy(self, a, flags=0, copy=False, track=False, ipython=False):
        """
        NPY serialize to a buffer and then send

        Probably this is doing two copies:

        #. array to stream (inevitable as need the NPY serialization with its header etc..)
        #. stream to content

        Unfortunately it seems io.BytesIO streams does not provide 
        a buffer interface to give access without copying.
        This means are forced to make a copy of the bytes into "content"

        http://python.6.x6.nabble.com/Buffer-protocol-for-io-BytesIO-td1902991.html        
        """
        if copy:
            log.warn("using slower copy=True option ")
            assert 0
        pass
        bufs = []
        if not a is None:
            stream = io.BytesIO()
            np.save( stream, a )     # write ndarray into stream
            stream.seek(0)
            content = stream.read()  # NPY format serialized bytes 
            buf = memoryview(content)
            bufs.append(buf)
            if hasattr(a, 'meta'):
                for jsd in a.meta:
                    bufs.append(json.dumps(jsd))   # convert dicts to json strings
                pass
            pass
        else:
            log.warn("send_npy sending empty json") 
            bufs.append(json.dumps({}))
        pass
        if ipython:
            log.info("stopped in send_npy just after creating the bufs to send")
            IPython.embed()
        pass
        log.info("send_npy sending %s bufs copy %s " % (len(bufs),copy))
        return self.send_multipart( bufs, flags=flags, copy=copy, track=track)


    def recv_npy(self, flags=0, copy=False, track=False, meta_encoding="ascii", ipython=False):
        """
        When copy=True receive bytes otherwise receive frames

        http://zeromq.github.io/pyzmq/api/zmq.html#frame

        frame.buffer
             A read-only buffer view of the message contents.

        frame.bytes
             The message content as a Python bytes object.
             The first time this property is accessed, a copy of the message contents is made. 
             From then on that same copy of the message is returned.


        When `copy=False` the list of frames provided by recv_multipart
        provide with `.buffer` memoryview objects.  But these have 

        #. ndim 0
        #. itemsize 1
        #. shape = strides = None

        Presumably this means no buffer interface is implemented, which 
        makes attempts to index into the memoryview fail with::

            IndexError: invalid indexing of 0-dim memory

        Workaround adopted to peek at the bytes is to use io.BytesIO 
        which provides peek/seek/read file like access into memory.
        """ 
        if copy:
            log.warn("using slower copy=True option ")
            assert 0
            msgs = self.recv_multipart(flags=flags, copy=True, track=track)  # bytes
            bufs = map(lambda msg:buffer(msg),msgs)
        else:
            frames = self.recv_multipart(flags=flags,copy=False, track=track)    
            bufs = map(lambda frame:frame.buffer, frames)            # memoryview object 
        pass
        
        if ipython:
            log.info("stopped in recv_npy just after receiving the bufs (list of memoryview)")
            IPython.embed()
        pass

        arys = []
        meta = []
        other = []

        for buf in bufs:   
            stream = io.BytesIO(buf)     # file like access to memory buffer
            peek = stream.read(1)
            stream.seek(0)
            if peek == '\x93':
                a = np.load(stream)
                aa = a.view(NP)          # view as subclass, to enable attaching metadata
                arys.append(aa)
            else:
                txt = codecs.decode(stream.read(-1))
                if peek == '{':
                    try:
                        jsdict = json.loads(txt)   
                    except ValueError:
                        log.warn("JSON load error for %s " % repr(txt))
                    pass
                    meta.append(jsdict)
                else:
                    other.append(txt)
                pass
            pass
        pass

        log.info("recy_npy got %s frames: %s NPY, %s json metadata, %s other " % (len(bufs),len(arys),len(meta),len(other))) 
        if len(arys) == 0:
            aa = NP(0)   # zombie ndarray 
            log.warn("no NPY serialization found in any of the multipart frames ")
        elif len(arys) == 1:
            aa = arys[0]
        else:
            aa = arys[0]
            log.warn("found %s NPY in the multipart frames, returning first only " % len(arys) ) 
        pass
        aa.meta  = meta
        aa.other = other
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




