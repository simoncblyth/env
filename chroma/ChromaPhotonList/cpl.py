#!/usr/bin/env python
"""

Usage::

    chromaphotonlist-
    chromaphotonlist-export  # _LIB envvar 

"""
import os, logging
import numpy as np
from env.root.import_ROOT import ROOT     # avoids sys.argv kidnap
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


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    cpl = random_cpl()
    cpl.Details()
    cpl.Print()

    examine_cpl(cpl)




