#!/usr/bin/env python
"""
CPLResponder
=============

The ChromaPhotonList responder is a pyzmq implemented 
ZMQ "worker" that connects to a broker and polls 
for messages.





Testing CPLResponder
======================

The responder acts as a dummy 
for a client connection.

Usage Steps
------------

Start the three below processes in separate 
Terminal windows in order to see whats happening in all of them.

* start broker on N (current `zmq-` default)::

    delta:~ blyth$ ssh N
    [blyth@belle7 ~]$ zmq_broker.sh

* start CPL responder on any node that can access backend port (5002)::

    cpl_responder.sh # NB .sh for environment setup

* send a test CPL object::

    delta:~ blyth$ czrt.sh 


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



TODO: 

* check if this also happens with bare C ZMQ, seemingly not with zmq_broker.c (which uses zmq_proxy call) 

* investigate `pyzmq/zmq/eventloop/minitornado/ioloop.py` rather than manual poll from a timer

To reproduce simply launch `cpl_responder.sh` and resize the terminal window
(happens with Terminal.app and iTerm.app).

Some EINTR error handling is needed presumably.

* http://stackoverflow.com/questions/1674162/how-to-handle-eintr-interrupted-system-call



Usage Examples
----------------

*  `env/chroma/echoserver/echoserver.py`



"""
import os, logging
log = logging.getLogger(__name__)

from env.zmqroot.responder import ZMQRootResponder
from cpl import examine_cpl, random_cpl, save_cpl, load_cpl


class CPLResponder(ZMQRootResponder):
    def __init__(self, config ):
        ZMQRootResponder.__init__(self, config)

    def reply(self, obj):
        log.info("CPLResponder reply") 

        if self.config.dump:
            examine_cpl( obj )

        if self.config.random:
            obj = random_cpl()

        return obj 



def check_cpl_responder():
    class Config(object):
        mode = 'connect'
        endpoint = os.environ['ZMQ_BROKER_URL_BACKEND']
        timeout = None # milliseconds  (None means dont timeout, just block)
        sleep = 0.5  # seconds
        dump = True
        random = False
    pass
    cfg = Config()
    responder = CPLResponder(cfg)
    log.info("polling: %s " % repr(responder))
    responder.poll()

def main():
    logging.basicConfig(level=logging.INFO)
    check_cpl_responder()

if __name__ == '__main__':
    main()



