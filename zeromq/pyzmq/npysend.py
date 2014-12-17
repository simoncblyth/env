#!/usr/bin/env python
"""
NPYSEND
=========

A quick way to feed the worker some numpy arrays to 
chew on for use while testing.  Avoids the need to 
run NuWa or mocknuwa. 

Usage::

    delta:~ blyth$ npysend.sh 
    delta:~ blyth$ npysend.sh --tag 1 --type cerenkov
    delta:~ blyth$ npysend.sh --tag 1 --type scintillation 
    delta:~ blyth$ npysend.sh --tag 1 --type photon

Both broker and worker must be running for a response to be provided, 
start those in two terminal windows first::

    delta:~ blyth$ czmq_broker.sh 
    delta:~ blyth$ g4daechroma.sh 

"""
import logging, os
import numpy as np
import zmq 
log = logging.getLogger(__name__)

from npycontext import NPYContext 


class Export(dict):
    tmplmap = {
            'photon':"DAE_PATH_TEMPLATE_NPY",
          'cerenkov':"DAECERENKOV_PATH_TEMPLATE",
   'scintillation':"DAESCINTILLATION_PATH_TEMPLATE",
          }

    def __init__(self):
        dict.__init__(self)
        self.update(self.tmplmap)

    def cerenkov(self, tag):
        return self.load(tag, "cerenkov")

    def scintillation(self, tag):
        return self.load(tag, "scintillation")

    def photon(self, tag):
        return self.load(tag, "photon")

    def load(self, tag, typ ): 
        tmplname = self.get(typ, None)
        assert tmplname, "typ %s unhandled" % typ
        tmpl = os.environ.get(tmplname, None)
        assert tmpl, "envvar %s missing " % tmplname
        log.info("tmplname %s tmpl %s tag %s " % (tmplname, tmpl, tag))
        path = tmpl % tag
        a = np.load(path)
        log.info("loaded %s %s " % (path, str(a.shape) ))    
        return a 



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    dtype = "cerenkov"
    dtag = "1"
    parser.add_argument("--type", default=dtype, help="Type of file to load",type=str)
    parser.add_argument("-l","--level", default="INFO", help="INFO/DEBUG/WARN/..")  
    parser.add_argument("--endpoint", default=os.environ['ZMQ_BROKER_URL_FRONTEND'], help="broker url")  
    parser.add_argument("--tag", default=dtag ) 
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    return args


def main():
    config = parse_args()

    exp = Export()
    request = exp.load(config.tag,  config.type )
    context = NPYContext()
    socket = context.socket(zmq.REQ)

    log.info("connect to endpoint %s " % config.endpoint ) 
    socket.connect(config.endpoint)
    log.info("send_npy")
    socket.send_npy(request)
    log.info("recv_npy")
    response = socket.recv_npy(copy=False)
    log.info("response %s\n%s " % (str(response.shape), repr(response)))
    pass

if __name__ == '__main__':
    main()

