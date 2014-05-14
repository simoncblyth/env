#!/usr/bin/env python
"""
ZMQRoot Responder
==================

ISSUE: Terminal Window resize causes Interrupted system call
---------------------------------------------------------------

Changing terminal window size whilst the responder is polling
results in `zmq.error.ZMQError: Interrupted system call`
and death of the responder : so arrange terminal window size
before starting

Same problem happening from inside g4daeview.py DAEResponder::


    2014-05-12 19:06:17,857 env.geant4.geometry.collada.g4daeview.daeinteractivityhandler:130 DAEResponder connect tcp://203.64.184.126:5002
    Traceback (most recent call last):
      File "_ctypes/callbacks.c", line 314, in 'calling callback function'
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/OpenGL/GLUT/special.py", line 155, in deregister
        function( value )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy/window/backend_glut.py", line 571, in func
        handler(dt)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeinteractivityhandler.py", line 133, in _check_responder
        responder.update()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeresponder.py", line 37, in update
        self.poll()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/zmqroot/responder.py", line 45, in poll
        events = self.socket.poll(timeout=self.config.timeout, flags=self.config.flags )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/socket.py", line 411, in poll
        evts = dict(p.poll(timeout))
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/sugar/poll.py", line 110, in poll
        return zmq_poll(self.sockets, timeout=timeout)
      File "_poll.pyx", line 115, in zmq.backend.cython._poll.zmq_poll (zmq/backend/cython/_poll.c:1640)
      File "checkrc.pxd", line 21, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/_poll.c:2003)
    zmq.error.ZMQError: Interrupted system call


How to reproduce
~~~~~~~~~~~~~~~~~~~

Simply launch `cpl_responder.sh` and resize the terminal window (happens with both Terminal.app and iTerm.app).

TODO 
~~~~~


* check if this also happens with bare C ZMQ, seemingly not with zmq_broker.c (which uses zmq_proxy call) 

* investigate `pyzmq/zmq/eventloop/minitornado/ioloop.py` rather than manual poll from a timer

Some EINTR error handling is needed presumably.

* http://stackoverflow.com/questions/1674162/how-to-handle-eintr-interrupted-system-call


Search Refs
~~~~~~~~~~~~~

* http://gerg.ca/blog/post/2013/zmq-child-process/
* https://github.com/zeromq/pyzmq/issues/348
* https://gist.github.com/minrk/5258909


"""
import logging, time, errno
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

