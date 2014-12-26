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
import os, logging
log = logging.getLogger(__name__)
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.geant4.geometry.collada.g4daeview.daedirectconfig import DAEDirectConfig
from env.geant4.geometry.collada.g4daenode import DAENode
from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
from chroma.detector import Detector

from env.geant4.geometry.collada.g4daeview.daephotonsnpl import DAEPhotonsNPL as NPL
npl = lambda _:NPL.load(_)

pp = lambda _:np.load(os.environ['DAE_PHOTON_PATH_TEMPLATE'] % str(_))
hh = lambda _:np.load(os.environ['DAE_HIT_PATH_TEMPLATE'] % str(_))
tt = lambda _:np.load(os.environ['DAE_TEST_PATH_TEMPLATE'] % str(_))


stc = lambda _:np.load(os.environ['DAE_CERENKOV_PATH_TEMPLATE'] % str(_))
sts = lambda _:np.load(os.environ['DAE_SCINTILLATION_PATH_TEMPLATE'] % str(_))

chc = lambda _:np.load(os.environ['DAE_OPCERENKOV_PATH_TEMPLATE'] % str(_))
chs = lambda _:np.load(os.environ['DAE_OPSCINTILLATION_PATH_TEMPLATE'] % str(_))

g4c = lambda _:np.load(os.environ['DAE_GOPCERENKOV_PATH_TEMPLATE'] % str(_))
g4s = lambda _:np.load(os.environ['DAE_GOPSCINTILLATION_PATH_TEMPLATE'] % str(_))


def genconsistency(evt):
    genconsistency_cerenkov(evt)
    genconsistency_scintillation(evt)    


def genconsistency_cerenkov(evt):
    _g4c = g4c(evt)
    _chc = chc(evt)
    _stc = stc(evt)

    n = _stc[:,0,3].view(np.int32).sum()

    log.info("g4c : GOPCERENKOV    %s " % str(_g4c.shape))
    log.info("chc :  OPCERENKOV    %s " % str(_chc.shape))
    log.info("stc :    CERENKOV    %s ==> N %s " % (str(_stc.shape), n) )
    
    assert _g4c.shape[0] == n
    assert _chc.shape[0] == n


def genconsistency_scintillation(evt):
    _g4s = g4s(evt)
    _chs = chs(evt)
    _sts = sts(evt)

    n = _sts[:,0,3].view(np.int32).sum()

    log.info("g4s : GOPSCINTILLATION    %s " % str(_g4s.shape))
    log.info("chs :  OPSCINTILLATION    %s " % str(_chs.shape))
    log.info("sts :    SCINTILLATION    %s ==> N %s " % (str(_sts.shape), n) )
    
    assert _g4s.shape[0] == n
    assert _chs.shape[0] == n




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


def cerenkov_wavelength(cg, cs, csi=0, nrand=1000):

    rands = np.random.random(nrand)
 
    materialIndex = cs[csi,0,2].view(np.int32)
    BetaInverse = cs[csi,4,0]
    maxSin2 = cs[csi,5,0]

    material = cg.unique_materials[materialIndex]
    ri = material.refractive_index
    w0 = ri[:,0][0]
    w1 = ri[:,0][-1]

    print "materialIndex %s BetaInverse %s maxSin2 %s material %s " % (materialIndex, BetaInverse, maxSin2, material.name)
    print "w0 %s w1 %s " % (w0, w1) 

    n = 0
    while n < nrand:
        u = rands[n]
        n += 1

        wavelength = w0 + (w1-w0)*u
        sampledRI = np.interp( wavelength , ri[:,0], ri[:,1] )
        cosTheta = BetaInverse/sampledRI
        sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta)
        sin2Theta_over_maxSin2 = sin2Theta/maxSin2 

        u = rands[n]
        n += 1
        print "wavelength %s sampledRI %s cosTheta %s sin2Theta %s sin2Theta/maxSin2 %s  %s " % (wavelength, sampledRI, cosTheta, sin2Theta, sin2Theta_over_maxSin2, u)
        if not u > sin2Theta_over_maxSin2:
            break
        pass
    pass

    print "n %s wavelength %s " % (n, wavelength)
 




def chroma_refractive_index(cg):
    for im, cm in enumerate(cg.unique_materials):
        wlri = cm.refractive_index
        wl = wlri[:,0]
        ri = wlri[:,1]
        print "[%2d] %25s %10s     wl %7.2f : %7.2f     %10.3f : %10.3f " % ( im, cm.name[17:-9], str(wlri.shape), wl.min(), wl.max(), ri.min(), ri.max() )  




def cf_xy(a,b):
    ir,ic = 0,0    
    plt.subplot(2,ir+1,ic+1)
    plt.title("x") 
    plt.hist((a[:,ir,ic],b[:,ir,ic]), bins=100,histtype="step" )
    ir,ic = 0,1    
    plt.subplot(2,ir+1,ic+1)
    plt.title("y") 
    plt.hist((a[:,ir,ic],b[:,ir,ic]), bins=100,histtype="step" )
    plt.show()

def cf_time(a,b, **kwa):
    ir, ic = 0, 3 

    cfg = dict(bins=100,histtype="step", range=(0,100),title="time")
    cfg.update(kwa)

    plt.title(cfg.pop("title"))
    plt.hist((a[:,ir,ic],b[:,ir,ic]), **cfg)
    plt.show()

def cf_wavelength(a,b, **kwa):
    ir, ic = 1, 3 
    cfg = dict(bins=100,histtype="step",title="wavelength")
    cfg.update(kwa)

    plt.title(cfg.pop("title"))
    plt.hist((a[:,ir,ic],b[:,ir,ic]), **cfg)
    plt.show()


def cf_xyz(a,b,r=0, **kwa):
    nr, nc = 3, 1 

    cfg = dict(bins=100,histtype="step")
    cfg.update(kwa)

    pl = 0 
    for c, t in enumerate(["x","y","z"]):
        pl += 1
        plt.subplot(nr,nc,pl)
        plt.title(t) 
        plt.hist((a[:,r,c],b[:,r,c]), **cfg )
    pass
    plt.show()



def cf_3xyz(a,b, **kwa):
    cfg = dict(bins=100,histtype="step")
    cfg.update(kwa)

    nr, nc, pl = 3, 3, 0 
    for r, rt in enumerate(["pos","dir","pol"]):
        for c, ct in enumerate(["x","y","z"]):
            pl += 1
            plt.subplot(nr,nc,pl)
            plt.title("%s %s " % (rt, ct)) 
            plt.hist((a[:,r,c],b[:,r,c]), **cfg )
        pass 
    pass
    plt.show()


def cf_3xyzw(a,b, **kwa):
    cfg = dict(bins=100,histtype="step")
    cfg.update(kwa)

    nr, nc, pl = 3, 4, 0 
    for r, rt in enumerate(["pos","dir","pol"]):
        for c, ct in enumerate(["x","y","z","w"]):
            pl += 1
            plt.subplot(nr,nc,pl)
            plt.title("%s %s " % (rt, ct)) 

            if (r,c) == (0,3):
                kwa['range'] = (0,100)
            else:
                kwa['range'] = None
            pass

            plt.hist((a[:,r,c],b[:,r,c]), **cfg)
        pass 
    pass
    plt.show()





if __name__ == '__main__':
   g = geometry()
   print g

