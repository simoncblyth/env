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

class NPYProcessor(dict):
    tmplmap = {
            'photon':"DAE_PATH_TEMPLATE_NPY",
          'cerenkov':"DAECERENKOV_PATH_TEMPLATE",
   'scintillation':"DAESCINTILLATION_PATH_TEMPLATE",
            'test':"DAETEST_PATH_TEMPLATE",
          }

    def __init__(self, config):
        self.config = config
        dict.__init__(self)
        self.update(self.tmplmap)

    def cerenkov(self, tag):
        return self.load(tag, "cerenkov")

    def scintillation(self, tag):
        return self.load(tag, "scintillation")

    def photon(self, tag):
        return self.load(tag, "photon")

    def process(self, request):
        context = NPYContext()
        socket = context.socket(zmq.REQ)

        log.info("connect to endpoint %s " % self.config.endpoint ) 
        socket.connect(self.config.endpoint)
        log.info("send_npy")
        socket.send_npy(request)
        log.info("recv_npy")
        response = socket.recv_npy(copy=False)
        log.info("response %s\n%s " % (str(response.shape), repr(response)))

        return response


    def path(self, tag, typ):
        tmplname = self.get(typ, None)
        assert tmplname, "typ %s unhandled" % typ
        tmpl = os.environ.get(tmplname, None)
        assert tmpl, "envvar %s missing " % tmplname
        log.info("tmplname %s tmpl %s tag %s " % (tmplname, tmpl, tag))
        path = tmpl % tag
        return path

    def load(self, tag, typ, sli ): 
        path = self.path(tag, typ)
        a = np.load(path)
        log.info("load %s %s " % (path, str(a.shape) ))    
        if not sli is None:
            chop = slice(*map(int,sli.split(":")))
            a = a[chop]
            log.info("sliced down to %s " % (str(a.shape) ))    
        pass 
        return a 

    def save(self, b, tag, typ):
        path = self.path(tag, typ)
        pdir = os.path.dirname(path)
        if not os.path.isdir(pdir):
           os.makedirs(pdir)

        log.info("save %s %s " % (path, str(b.shape) ))    
        np.save(path, b)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    d = {}
    d['tag'] = "1"
    d['slice'] = None
    d['inp'] = "cerenkov"
    d['out'] = "test"
    d['level'] = "INFO"
    d['endpoint'] = os.environ['ZMQ_BROKER_URL_FRONTEND']

    parser.add_argument("--tag", default=d['tag'] ) 
    parser.add_argument("--inp", default=d['inp'], help="Type of file to load",type=str)
    parser.add_argument("--out", default=d['out'], help="Type of file to save response into", type=str)
    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    parser.add_argument("--endpoint", default=d['endpoint'], help="broker url")  
    parser.add_argument("--slice", default=d['slice'], help="Colon delimited slice to apply to array, eg 0:1 for 1st item, or 0:2 for 1st two items")
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    np.set_printoptions(precision=3, suppress=True)
    return args

def main():
    config = parse_args()
    proc = NPYProcessor(config)
    request = proc.load(config.tag,  config.inp, config.slice )
    response = proc.process(request)
    proc.save(response, config.tag, config.out ) 

if __name__ == '__main__':
    main()

