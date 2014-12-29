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

ppp = lambda _:np.load(os.environ['DAE_PHOTON_PATH_TEMPLATE'] % str(_))
hhh = lambda _:np.load(os.environ['DAE_HIT_PATH_TEMPLATE'] % str(_))
ttt = lambda _:np.load(os.environ['DAE_TEST_PATH_TEMPLATE'] % str(_))

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



class CerenkovPhoton(np.ndarray):
    """
    see DsChromaG4Cerenkov.cc
    ::
 
        In [86]: g4c_ = CerenkovPhoton.get(1)

        In [89]: g4c_.cpid
        Out[89]: CerenkovPhoton([     1,      2,      3, ..., 612839, 612840, 612841], dtype=int32)

        In [88]: g4c_.csid
        Out[88]: CerenkovPhoton([   1,    1,    1, ..., 7836, 7836, 7836], dtype=int32)

    """
    @classmethod
    def get(cls, tag):
        return g4c(tag).view(cls)

    cpid = property(lambda self:self[:,3,0].view(np.int32)) # 1-based CerenkovPhoton index 
    csid = property(lambda self:self[:,3,1].view(np.int32)) # 1-based CerenkovStep index 


    

class CerenkovStep(np.ndarray):
    """
    ::

        cg = chroma_geometry()

        In [45]: cs = stc(1).view(CerenkovStep)   # view array as CerenkovStep

        In [46]: cs.plot_refractive_index(cg)   

        cs.plot(mm, 'refractive_index')

        In [104]: wi = water_indices(cg)

        In [105]: wi
        Out[105]: [22, 24, 27, 28]

        In [106]: np.unique(cs.materialIndex)
        Out[106]: CerenkovStep([ 0,  3,  5, 10, 24, 27, 28], dtype=int32)

        ## kludge usage of accidental? clustering of the waters at high materialIndex

        In [108]: cs[cs.materialIndex < 22].shape
        Out[108]: (6487, 6, 4)

        In [109]: cs[cs.materialIndex > 22].shape
        Out[109]: (1349, 6, 4)

        ## indices of water and non-water steps

        In [121]: cs[cs.materialIndex > 22].csid
        Out[121]: CerenkovStep([   1,    2,    3, ..., 7834, 7835, 7836], dtype=int32)

        In [122]: cs[cs.materialIndex < 22].csid
        Out[122]: CerenkovStep([ 115,  116,  117, ..., 6599, 6600, 6601], dtype=int32)



    Want to work out the CerenkovPhoton indices of photons from water steps::

            In [156]: ws.csid
            Out[156]: CerenkovStep([   1,    2,    3, ..., 7834, 7835, 7836], dtype=int32)    
             

    """
    @classmethod
    def get(cls, tag):
        return stc(tag).view(cls)

    csid = property(lambda self:-self[:,0,0].view(np.int32))
    parentId = property(lambda self:self[:,0,1].view(np.int32))
    materialIndex = property(lambda self:self[:,0,2].view(np.int32))
    numPhotons = property(lambda self:self[:,0,3].view(np.int32))  

    code = property(lambda self:self[:,3,0].view(np.int32))  
    BetaInverse = property(lambda self:self[:,4,0])
    maxSin2 = property(lambda self:self[:,5,0])
    bialkaliIndex = property(lambda self:self[:,5,3].view(np.int32))  

    materialIndices = property(lambda self:np.unique(self.materialIndex))

    def materials(self, cg):
        return [cg.unique_materials[materialIndex] for materialIndex in self.materialIndices]

    def plot_refractive_index(self, cg):
        """
        Water starts at 200nm
        """
        mm = self.materials(cg)
        qplot(mm, 'refractive_index')






# from chroma.gpu.GPUGeometry
def interp_material_property(wavelengths, prop):
    # note that it is essential that the material properties be
    # interpolated linearly. this fact is used in the propagation
    # code to guarantee that probabilities still sum to one.
    return np.interp(wavelengths, prop[:,0], prop[:,1]).astype(np.float32)

def standardize( prop, standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)):
    """
    mimic what the chroma.geometry machinery does to properties on copying to GPU
    """
    vals = interp_material_property(standard_wavelengths,  prop )
    return np.vstack([standard_wavelengths, vals]).T


def qplot(materials, standard=False, qty='refractive_index'):
    """
    :param materials: list of chroma material instances
    :param standard:  when True apply chroma wavelength standardization and interpolation
    :param qty: name of quantity 
    """
    title = qty
    for m in materials:
        q = getattr(m, qty, None)
        if q is None:continue
        if standard:
            q = standardize(q)
            title += " standardized " 
        pass
        plt.plot( q[:,0], q[:,1], label=m.name[17:-9])
        pass
    pass
    plt.title(title)
    plt.legend()
    plt.show()  


def water_indices(cg):
    return filter(lambda _:cg.unique_materials[_].name.find('Water')>-1,range(len(cg.unique_materials)))


def cerenkov_wavelength(cg, cs, csi=0, nrand=100000, standard=False):
    """
    ::

         cg = chroma_geometry()
         cs = stc(1)

    Rapidly descending distrib with wavelength (Cerenkov blue light)
    starting from the low edge of the ri property of the material.
    What you get is majorly dependent on the ri range of the material
    so if diffent materials have different ranges, artifacts are inevitable

    Scintillator RINDEX start at 80nm, waters at 200nm

    ::

        In [56]: cerenkov_wavelength(cg, cs, 0)
        materialIndex 24 BetaInverse 1.00001 maxSin2 0.482422 material __dd__Materials__IwsWater0xc288f98 
        w0 199.975 w1 799.898 

        In [57]: cerenkov_wavelength(cg, cs, 1)
        materialIndex 24 BetaInverse 1.00001 maxSin2 0.482422 material __dd__Materials__IwsWater0xc288f98 
        w0 199.975 w1 799.898 

        In [58]: cerenkov_wavelength(cg, cs, 1000)
        materialIndex 0 BetaInverse 1.41302 maxSin2 0.0550548 material __dd__Materials__LiquidScintillator0xc2308d0 
        w0 79.9898 w1 799.898 


    """
    materialIndex = cs.materialIndex[csi]
    BetaInverse = cs.BetaInverse[csi]
    maxSin2 = cs.maxSin2[csi]

    material = cg.unique_materials[materialIndex]
    ri = material.refractive_index

    if standard:
        ri = standardize(ri) 

    w0 = ri[0,0]
    w1 = ri[-1,0]

    print "materialIndex %s BetaInverse %s maxSin2 %s material %s " % (materialIndex, BetaInverse, maxSin2, material.name)
    print "w0 %s w1 %s " % (w0, w1) 

    u1 = np.random.random(nrand)
    u2 = np.random.random(nrand)
 
    iw = (1./w1)*u1 + (1./w0)*(1.-u1)  # uniform in 1/w 
    w = 1./iw

    sampledRI = np.interp( w, ri[:,0], ri[:,1] )
    cosTheta = BetaInverse/sampledRI
    sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta)
    sin2Theta_over_maxSin2 = sin2Theta/maxSin2 
    ws = w[np.where( u2 <= sin2Theta_over_maxSin2)]

    plt.hist(ws, bins=100)
    plt.show() 




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

