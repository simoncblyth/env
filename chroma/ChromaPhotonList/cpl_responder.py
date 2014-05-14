#!/usr/bin/env python
"""
CPLResponder
=============

The ChromaPhotonList responder is a pyzmq implemented 
ZMQ "worker" that connects to a broker and polls 
for messages. The main of this script provides a dummy standin 
for a real "ChromaWorker" (to be implemented) 
based on the existing ChromaServer. 


Testing CPLResponder
======================

To test the ZMQ client/broker/worker network topology.


Usage Steps
------------

Start the three below processes in separate 
Terminal windows in order to see whats happening in all of them.

* start **broker** on N (current `zmq-` default)::

    delta:~ blyth$ ssh N
    [blyth@belle7 ~]$ zmq_broker.sh

* start **worker** CPL responder on any node that can access the broker backend port (5002)::

    cpl_responder.sh # NB .sh for environment setup

* run **client** by sending a test CPL object::

    delta:~ blyth$ czrt.sh 



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



