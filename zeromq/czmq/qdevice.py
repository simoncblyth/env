#!/usr/bin/env python
"""

https://learning-0mq-with-pyzmq.readthedocs.org/en/latest/pyzmq/pyzmqdevices/queue.html

http://zeromq.github.io/pyzmq/devices.html

http://zeromq.github.io/pyzmq/api/zmq.devices.html#zmq.device

   replace devices for proxy 

"""
import logging
log = logging.getLogger(__name__)
import time
import zmq

from zmq.devices.basedevice import ProcessDevice

import random

def endpoint(host,port):
    return "tcp://%s:%d" % (host,int(port))

def queue(cfg):
    queuedevice = ProcessDevice(zmq.QUEUE, zmq.XREP, zmq.XREQ)
    frontend = cfg.frontend
    backend = cfg.backend
    log.info("queue bind_in frontend %s and bind_out backend %s " % (frontend, backend)) 
    queuedevice.bind_in(frontend)
    queuedevice.bind_out(backend)
    #queuedevice.setsockopt_in(zmq.HWM, 1)
    #queuedevice.setsockopt_out(zmq.HWM, 1)
    queuedevice.start()
    time.sleep (2)  
    
def proxy(cfg):
    context = zmq.Context()
    frontend = cfg.frontend
    backend = cfg.backend
    log.info("proxy bind frontend %s and backend %s " % (frontend, backend)) 

    front = context.socket(zmq.XREP)
    back = context.socket(zmq.XREQ)

    front.bind(frontend)
    back.bind(backend)

    proxy_ = zmq.proxy( front, back )


def server(cfg):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    backend = cfg.backend
    log.info("server connecting REP socket to backend  %s " % backend )
    socket.connect(backend)
    server_id = random.randrange(1,10005)
    while True:
        message = socket.recv()
        log.info("server received %s " % message )
        socket.send("Response from %s" % server_id)

def client(cfg):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    frontend = cfg.frontend
    client_id = cfg.opts.client_id
    log.info("client %s connecting REQ socket to frontend %s " % (client_id, frontend) )
    socket.connect(frontend)
    for request in range (1,5):
        print "Sending request #%s" % request
        socket.send ("Request fron client: %s" % client_id)
        message = socket.recv()
        print "Received reply ", request, "[", message, "]"


class Config(object):
    def __init__(self, doc):
        opts, args = self.parse_arguments(doc)
        self.opts = opts
        self.args = args

    frontend = property(lambda self:endpoint(self.opts.frontend_host,self.opts.frontend_port))
    backend  = property(lambda self:endpoint(self.opts.backend_host,self.opts.backend_port))
    mode = property(lambda self:self.args[0])

    def parse_arguments(self,doc):
        from optparse import OptionParser
        parser = OptionParser(doc)
        parser.add_option("--level", default="INFO")
        parser.add_option("--frontend-port", default="5001")
        parser.add_option("--backend-port",  default="5002")
        parser.add_option("--frontend-host", default="127.0.0.1")
        parser.add_option("--backend-host",  default="127.0.0.1")
        parser.add_option("--client-id",     default="1")
        (opts, args) = parser.parse_args()
        logging.basicConfig(level=getattr(logging,opts.level.upper()))
        return opts, args


def main():
   cfg = Config(__doc__)
   mode = cfg.mode
   if mode == "server":
       server(cfg)
   elif mode == "client":
       client(cfg)
   elif mode== "queue":
       queue(cfg)
   elif mode== "proxy":
       proxy(cfg)
   else:
       assert 0 

if __name__ == '__main__':
   main()



