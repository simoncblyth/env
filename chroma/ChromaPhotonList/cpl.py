#!/usr/bin/env python
"""

Usage::

    cpl-
    cpl-export  # _LIB envvar 


#. TODO: lookinto rootmap with pyroot to automate finding libs ?

"""
import os, logging
import numpy as np

try:
    from env.root.import_ROOT import ROOT     # avoids sys.argv kidnap
except ImportError:
    ROOT = None


if not ROOT is None and not hasattr(ROOT, 'ChromaPhotonList'):
    if ROOT.gSystem.Load(os.environ["CHROMAPHOTONLIST_LIB"]) < 0:ROOT.gSystem.Exit(10)

log = logging.getLogger(__name__)


cpl_atts = 'pmtid polx poly polz px py pz t wavelength x y z'.split()

def examine_cpl(obj):
    print repr(obj)
    print obj.__class__

    atts = cpl_atts
    vecs = dict(map(lambda att:[att,getattr(obj,att)], atts ))
    sizs = map(lambda att:vecs[att].size(), atts)
    print sizs
    assert len(set(sizs)) == 1
    size = sizs[0]
    for i in range(size)[:10]:
        vals = map(lambda att:vecs[att][i], atts)
        d = dict(zip(atts,vals))
        print d

def random_cpl(n=100):
     cpl = ROOT.ChromaPhotonList()
     for _ in range(n):
         cpl.AddPhoton( *np.random.random(11) )
     return cpl 

def save_cpl( path, key, obj, compress=1 ):
    log.info("save_cpl to %s with key %s " % (path,key) )
    if ROOT is None:
        log.warn("save_cpl requires ROOT " )
        return

    title = key  
    f = ROOT.TFile( path, 'RECREATE', title, compress ) 
    if f.IsZombie():
        log.warn("save_cpl: failed open for writing path %s " % path )
        return None
    pass
    obj.Write(key)
    f.Close()
    return 0

def load_cpl( path, key ):
    log.info("load_cpl from %s " % path )
    if ROOT is None:
        log.warn("load_cpl required ROOT " )
        return

    if not os.path.exists(path):
        log.warn("path %s does not exist " % path )
        return None
    pass
    f = ROOT.TFile( path, 'READ' ) 
    if f.IsZombie():
        log.warn("path %s exists but open failed  " % path )
        return None
    pass
    obj = f.Get(key)
    f.Close()
    return obj

def check_creation():
    cpl = random_cpl()
    cpl.Details()
    cpl.Print()
    examine_cpl(cpl)
    return cpl

def check_save_load():
    cpl = random_cpl(10)
    path = "/tmp/cpl.root"
    save_cpl( path,cpl )
    obj = load_cpl(path)
    print obj
    obj.Details() 

if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    cpl = check_creation()





