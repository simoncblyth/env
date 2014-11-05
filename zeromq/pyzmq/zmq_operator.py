#!/usr/bin/env python
"""

https://learning-0mq-with-pyzmq.readthedocs.org/en/latest/pyzmq/pyzmqdevices/queue.html

http://zeromq.github.io/pyzmq/devices.html

http://zeromq.github.io/pyzmq/api/zmq.devices.html#zmq.device

   replace devices for proxy 

"""
import logging, os
log = logging.getLogger(__name__)
import time
import zmq
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import io
from npyresponder import NPYContext 

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
    
def broker(cfg):
    context = zmq.Context()
    frontend = cfg.frontend
    backend = cfg.backend
    log.info("proxy bind frontend %s and backend %s " % (frontend, backend)) 

    front = context.socket(zmq.ROUTER)
    back = context.socket(zmq.DEALER)

    front.bind(frontend)
    back.bind(backend)

    proxy_ = zmq.proxy( front, back )


def server(cfg):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    log.info("connect REP socket to backend %s " % cfg.backend )
    socket.connect(cfg.backend)
    server_id = random.randrange(1,10005)
    while True:
        msg = socket.recv()
        log.info("recv %s " % msg )
        socket.send("send %s" % server_id)

def worker(cfg):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    backend = cfg.backend
    log.info("server connecting REP socket to backend  %s " % backend )
    socket.connect(backend)
    server_id = random.randrange(1,10005)
    while True:
        message = socket.recv()
        log.info("server received %s " % message )
        socket.send("Response from %s" % server_id)

def mirror(cfg):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    backend = cfg.backend
    log.info("server connecting REP socket to backend  %s " % backend )
    socket.connect(backend)
    server_id = random.randrange(1,10005)
    while True:
        message = socket.recv()
        log.info("server received %s " % message )
        #socket.send("Response from %s" % server_id)
        socket.send(message)

def npymirror(cfg, copy=False):
    """
    Responding with an array with 
    dimensions other than the first changed 
    will almost certainly fail back at client receiver
    """
    ctx = NPYContext()
    socket = ctx.socket(zmq.REP)
    backend = cfg.backend
    log.info("npymirror server connecting REP socket to backend  %s  copy %s " % (backend,copy) )
    socket.connect(backend)
    server_id = random.randrange(1,10005)
    while True:
        request = socket.recv_npy(copy=copy)
        log.info("npymirror received request dtype %s shape %s" % (request.dtype, request.shape) )
        log.info("npymirror received request \n%s" % request )

        qsh = list(request.shape)
        shape = [qsh[0]*2] + qsh[1:] 
        response = np.ones( shape, dtype=request.dtype )
        #response = request

        socket.send_npy( response )



def npymirror0(cfg, copy=False):
    """
    http://zeromq.github.io/pyzmq/api/zmq.html
    """
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    backend = cfg.backend
    log.info("server connecting REP socket to backend  %s  copy %s " % (backend,copy) )
    socket.connect(backend)
    server_id = random.randrange(1,10005)
    while True:
        # forming npy array from NPY serialized bytes received
        if copy:
            msg = socket.recv(copy=True)  # bytes
            buf = buffer(msg)
        else:
            frame = socket.recv(copy=False)    
            buf = frame.buffer            # memoryview object 
        pass
        stream = io.BytesIO(buf)
        request = np.load(stream)

        log.info("npymirror received request dtype %s shape %s" % (request.dtype, request.shape) )
        log.info("npymirror received request \n%s" % request )

        # changing dimensions other than the first will almost 
        # certainly fail back at client receiver
        #
        qsh = list(request.shape)
        shape = [qsh[0]*2] + qsh[1:] 
        response = np.ones( shape, dtype=request.dtype )
        #response = request

        outstream = io.BytesIO()
        np.save( outstream, response )
        outstream.seek(0)
        outbuf = buffer(outstream.read())
        socket.send( outbuf )




def client(cfg):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    client_id = cfg.opts.client_id
    log.info("connect REQ socket %s to frontend %s " % (client_id, cfg.frontend) )
    socket.connect(cfg.frontend)
    for request in range (1,5):
        print "send req #%s" % request
        socket.send ("req from : %s" % client_id)
        message = socket.recv()
        print "recv rep ", request, "[", message, "]"


class Config(object):
    def __init__(self, doc):
        opts, args = self.parse_arguments(doc)
        self.opts = opts
        self.args = args

    #frontend = property(lambda self:endpoint(self.opts.frontend_host,self.opts.frontend_port))
    #backend  = property(lambda self:endpoint(self.opts.backend_host,self.opts.backend_port))
    frontend = property(lambda self:self.opts.frontend)
    backend  = property(lambda self:self.opts.backend)
    mode = property(lambda self:self.args[0])

    def parse_arguments(self,doc):
        from optparse import OptionParser
        parser = OptionParser(doc)
        parser.add_option("--level", default="INFO")
        parser.add_option("--frontend", default=os.environ.get('FRONTEND',None))
        parser.add_option("--backend",  default=os.environ.get('BACKEND',None))
        parser.add_option("--client-id",     default="1")
        (opts, args) = parser.parse_args()
        logging.basicConfig(level=getattr(logging,opts.level.upper()))
        return opts, args


def main():
   cfg = Config(__doc__)
   mode = cfg.mode
   if   mode == "server": server(cfg)
   elif mode == "client": client(cfg)
   elif mode == "worker": worker(cfg)
   elif mode == "queue": queue(cfg)
   elif mode == "broker": broker(cfg)
   elif mode == "mirror": mirror(cfg)
   elif mode == "npymirror": npymirror(cfg)
   else:
       assert 0, mode

if __name__ == '__main__':
   main()



