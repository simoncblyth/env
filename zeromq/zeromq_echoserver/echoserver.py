#!/usr/bin/env python
"""

"""
import os, time, zmq, logging

#from chroma.rootimport import ROOT
#if ROOT.gSystem.Load("librootmq" ) < 0:ROOT.gSystem.Exit(10)


log = logging.getLogger(__name__)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s")
    print zmq.pyzmq_version()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    config = os.environ['ECHO_SERVER_CONFIG']
    socket.bind( config )
    log.info("bound to %s hwm %s " % ( config, socket.hwm ))

    copy = False
    while True:
        msg = socket.recv(copy=copy)

        if copy == False: assert type(msg) == zmq.backend.cython.message.Frame
        if copy == True: assert type(msg) == str

        log.info("recv message of length %s " % len(msg))
        time.sleep(1)
        socket.send(msg)



