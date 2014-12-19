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
from env.geant4.geometry.collada.g4daeview.daedirectconfig import DAEDirectConfig
from env.geant4.geometry.collada.g4daenode import DAENode
from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
from chroma.detector import Detector

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

