#!/usr/bin/env python
"""
NPYSEND
=========

A quick way to send numpy arrays to broker frontend, which
then passes to worker. 
Avoids the need to run NuWa or mocknuwa. 

Usage::

    delta:~ blyth$ npysend.sh -icerenkov -otest -t1

    delta:~ blyth$ npysend.sh --tag 1 --inp handshake
    delta:~ blyth$ npysend.sh --inp scintillation --out opscintillation
    delta:~ blyth$ npysend.sh --inp cerenkov --out opcerenkov

    delta:~ blyth$ npysend.sh --inp opcerenkov --slice ::100


Can also access remote frontends::

    delta:~ blyth$ npysend.sh --zmqtunnelnode=G5 --inp handshake
    ## TODO: avoid leaking tunnels 


Both broker and worker must be running for a response to be provided, 
start those in two terminal windows first::

    delta:~ blyth$ czmq_broker.sh 
    delta:~ blyth$ g4daechroma.sh 


Issues
--------

Wrong python::

    delta:example blyth$ npysend.py --help
    Traceback (most recent call last):
      File "/Users/blyth/env/bin/npysend.py", line 2, in <module>
        from env.zeromq.pyzmq.npysend import main
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/env/zeromq/pyzmq/npysend.py", line 37, in <module>
        import zmq 
    ImportError: No module named zmq
    delta:example blyth$ 


Solution, use the bash wrapper which sets up the required environment::

    delta:example blyth$ npysend.sh --help
    usage: npysend.py [-h] [--ipython] [--copy] [--zmqtunnelnode ZMQTUNNELNODE]
                      [-t TAG] [-i INP] [-o OUT] [-l LEVEL] [--endpoint ENDPOINT]
                      [--slice SLICE] [--threads-per-block THREADS_PER_BLOCK]
                      [--steps-per-call STEPS_PER_CALL]

    optional arguments:
      -h, --help            show this help message and exit
      --ipython
      --copy                Copy frames into bytes during npysocket operation
                            (SLOWER)
      --zmqtunnelnode ZMQTUNNELNODE
                            Option handled in the invoking bash script, which
                            opens ssh tunnel to remote frontend
      -t TAG, --tag TAG
      -i INP, --inp INP     Either "handshake" or the type of file to load, eg
                            "cerenkov", "scintillation", "opscintillation"
      -o OUT, --out OUT     Type of file to save response into
      -l LEVEL, --level LEVEL
                            INFO/DEBUG/WARN/..
      --endpoint ENDPOINT   broker url
      --slice SLICE         Colon delimited slice to apply to array, eg 0:1 0:2
                            ::100
      --threads-per-block THREADS_PER_BLOCK
                            Kernel launch threads_per_block
      --steps-per-call STEPS_PER_CALL
                            Propagation steps within launch




"""
import logging, os, pprint
import numpy as np
import zmq 
log = logging.getLogger(__name__)

from npycontext import NPYContext 


class NPY(np.ndarray):pass

class NPYProcessor(object):
    def __init__(self, config):
        self.config = config

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
        socket.send_npy(request,copy=self.config.copy,ipython=self.config.ipython)
        response = socket.recv_npy(copy=self.config.copy, ipython=self.config.ipython)
        log.info("response %s\n%s " % (str(response.shape), repr(response)))

        meta = getattr(response, 'meta', [])
        for jsd in meta:
            print pprint.pformat(jsd)
        pass
        return response


    def tmplname(self, tag):
        return "DAE_%s_PATH_TEMPLATE" % tag.upper()

    def path(self, tag, typ):
        tmplname = self.tmplname(typ)
        tmpl = os.environ.get(tmplname, None)
        assert tmpl, "envvar %s missing " % tmplname
        log.info("tmplname %s tmpl %s tag %s " % (tmplname, tmpl, tag))
        path = tmpl % tag
        return path

    def load(self, tag, typ, sli ): 
        """
        ::

            In [8]: a[::10].shape
            Out[8]: (265265, 4, 4)

            In [10]: a[slice(None,None,10)].shape
            Out[10]: (265265, 4, 4)

        """
        path = self.path(tag, typ)
        a = np.load(path)
        log.info("load %s %s " % (path, str(a.shape) ))    
        if not sli is None:
            int_ = lambda _:int(_) if _ else None
            chop = slice(*map(int_,sli.split(":")))
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
    d['ipython'] = False
    d['copy'] = False
    d['tag'] = "1"
    d['slice'] = None
    d['inp'] = "cerenkov"
    d['out'] = "test"
    d['level'] = "INFO"
    d['endpoint'] = os.environ['ZMQ_BROKER_URL_FRONTEND']
    d['threads_per_block'] = None
    d['steps_per_call'] = None

    h = {}
    h['inp'] = "Either \"handshake\" or the type of file to load, eg \"cerenkov\", \"scintillation\", \"opscintillation\" "  

    parser.add_argument("--ipython", action="store_true", default=d['ipython'] ) 
    parser.add_argument("--copy", action="store_true", default=d['copy'], help="Copy frames into bytes during npysocket operation (SLOWER)" ) 
    parser.add_argument("--zmqtunnelnode", default=None, help="Option handled in the invoking bash script, which opens ssh tunnel to remote frontend" ) 
    parser.add_argument("-t","--tag", default=d['tag'] ) 
    parser.add_argument("-i","--inp", default=d['inp'], help=h['inp'],type=str)
    parser.add_argument("-o","--out", default=d['out'], help="Type of file to save response into", type=str)
    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    parser.add_argument("--endpoint", default=d['endpoint'], help="broker url")  
    parser.add_argument("--slice", default=d['slice'], help="Colon delimited slice to apply to array, eg 0:1 0:2 ::100")
    parser.add_argument("--threads-per-block", type=int, default=d['threads_per_block'], help="Kernel launch threads_per_block")
    parser.add_argument("--steps-per-call", type=int, default=d['steps_per_call'], help="Propagation steps within launch")
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    np.set_printoptions(precision=3, suppress=True)
    return args


def attach_metadata(request, config):
    """
    Mimic metadata settings from G4DAEChroma::GPUProcessing 
    """ 
    ctrl = {}

    if not config.inp is None:
        ctrl['type'] = config.inp
    if not config.tag is None:
        ctrl['evt'] = config.tag

    atts = ["threads_per_block", "steps_per_call"]
    for att in atts: 
        val = getattr(config, att, None)
        if val is None:continue 
        ctrl[att] = val 
    pass
    metadata = {}
    metadata['ctrl'] = ctrl 
    request.meta = [metadata] 


def main():
    config = parse_args()
    proc = NPYProcessor(config)

    if config.inp == "handshake":
        request = None
    else:
        request = proc.load(config.tag,  config.inp, config.slice ).view(NPY)
        attach_metadata(request, config)
    pass

    response = proc.process(request)
    proc.save(response, config.tag, config.out ) 

if __name__ == '__main__':
    main()

