#!/bin/env python
"""
Shortcut functions for use from ipython. Usage::

    In [2]: from env.g4dae import chroma_geometry, geometry, daenode

    In [3]: g = geometry()

    In [2]: dae = daenode()



    In [12]: slow = l.extra.properties['SLOWCOMPONENT']

    In [13]: fast = l.extra.properties['FASTCOMPONENT']

    In [21]: plt.bar(slow[:,0], slow[:,1])
    Out[21]: <Container object of 275 artists>

    In [22]: plt.show()




"""
import os
import numpy as np

from env.geant4.geometry.collada.g4daeview.daedirectconfig import DAEDirectConfig
from env.geant4.geometry.collada.g4daenode import DAENode
from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
from chroma.detector import Detector

from env.geant4.geometry.collada.g4daeview.daephotonsnpl import DAEPhotonsNPL as NPL
npl = lambda _:NPL.load(_)

pp = lambda _:np.load(os.environ['DAE_PHOTON_PATH_TEMPLATE'] % str(_))
hh = lambda _:np.load(os.environ['DAE_HIT_PATH_TEMPLATE'] % str(_))
cs = lambda _:np.load(os.environ['DAE_CERENKOV_PATH_TEMPLATE'] % str(_))
ss = lambda _:np.load(os.environ['DAE_SCINTILLATION_PATH_TEMPLATE'] % str(_))
cp = lambda _:np.load(os.environ['DAE_OPCERENKOV_PATH_TEMPLATE'] % str(_))
sp = lambda _:np.load(os.environ['DAE_OPSCINTILLATION_PATH_TEMPLATE'] % str(_))
tt = lambda _:np.load(os.environ['DAE_TEST_PATH_TEMPLATE'] % str(_))


def config():
    cfg = DAEDirectConfig()
    cfg.parse(nocli=True)
    return cfg

def daenode():
    cfg = config()
    DAENode.init(cfg.path)
    return DAENode

def chroma_geometry():
    cfg = config()
    print cfg.chromacachepath
    cg = Detector.get(cfg.chromacachepath)
    return cg 

def geometry():
    cfg = config()
    g = DAEGeometry.get(cfg) 
    return g 

def gdls():
    dae = daenode()
    return dae.materialsearch("__dd__Materials__GdDopedLS")    
    
def ls():
    dae = daenode()
    return dae.materialsearch("__dd__Materials__LiquidScintillator")


if __name__ == '__main__':
   g = geometry()
   print g

