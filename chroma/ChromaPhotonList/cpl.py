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
     import numpy as np
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


def create_cpl_from_photons_very_slowly( photons ):
    """
    Hmm how to copy numpy arrays into ROOT object in C/C++ 
    without laborious float-by-float python.
    To avoid that need to pass numpy array reference to C/C++ 

    * :doc:`/chroma/chroma_pyublas` pyublas/boost-python/boost-ublas example
    * http://root.cern.ch/phpBB3/viewtopic.php?t=4233 

    The slowness of this motivated looking into NPY serialization approaches.
    """
    cpl = ROOT.ChromaPhotonList()

    nphotons = len(photons.pos)
    x,y,z=photons.pos[:,0],photons.pos[:,1],photons.pos[:,2]
    px,py,pz=photons.dir[:,0],photons.dir[:,1],photons.dir[:,2]
    polx,poly,polz=photons.pol[:,0],photons.pol[:,1],photons.pol[:,2]
    t = photons.t 
    w = photons.wavelengths

    if not hasattr(photons, 'pmtid'):
       log.warn("filling in pmtid with dummy -1 ") 

    pmtid = getattr(photons,'pmtid', -np.ones( nphotons , np.int32 ))

    #cpl.FromArrays(x,y,z,px,py,pz,polx,poly,polz,t,w,pmtid,nphotons)  I WISH or better: cpl.FromPhotons(photons)
    for _ in range(nphotons):
        cpl.AddPhoton(x[_],y[_],z[_],
                      px[_],py[_],pz[_],
                      polx[_],poly[_],polz[_],
                      t[_],w[_],int(pmtid[_]))    # 
    return cpl 


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





