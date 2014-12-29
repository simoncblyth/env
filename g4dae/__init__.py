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

from env.g4dae.types import CerenkovStep, G4CerenkovPhoton, ChCerenkovPhoton
from env.g4dae.types import ScintillationStep, G4ScintillationPhoton, ChScintillationPhoton

cg = None
def cg_get():
    global cg
    if cg is None:
        cg = chroma_geometry()
    return cg 




def genconsistency(evt):
    genconsistency_cerenkov(evt)
    genconsistency_scintillation(evt)    


def genconsistency_cerenkov(evt):
    g4c = G4CerenkovPhoton.get(evt)
    chc = ChCerenkovPhoton.get(evt)
    stc = CerenkovStep.get(evt)
    n = stc.totPhotons

    log.info("g4c : GOPCERENKOV    %s " % str(g4c.shape))
    log.info("chc :  OPCERENKOV    %s " % str(chc.shape))
    log.info("stc :    CERENKOV    %s ==> N %s " % (str(stc.shape), n) )
    
    assert g4c.shape[0] == n
    assert chc.shape[0] == n


def genconsistency_scintillation(evt):
    g4s = G4ScintillationPhoton.get(evt)
    chs = ChScintillationPhoton.get(evt)
    sts = ScintillationStep.get(evt)
    n = sts.totPhotons

    log.info("g4s : GOPSCINTILLATION    %s " % str(g4s.shape))
    log.info("chs :  OPSCINTILLATION    %s " % str(chs.shape))
    log.info("sts :    SCINTILLATION    %s ==> N %s " % (str(sts.shape), n) )
    
    assert g4s.shape[0] == n
    assert chs.shape[0] == n




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





def g4_cerenkov_wavelength(tag, **kwa):
    """
    """
    cg = cg_get()
    pass
    g4c = G4CerenkovPhoton.get(tag)
    base = os.path.expandvars('$STATIC_BASE/env/g4dae') 
    path =  os.path.join(base, "g4_cerenkov_wavelength.png")
    cat = "aux0"
    val = "wavelength"
    title = "G4/Detsim Generated Cerenkov Wavelength by material" 
    catplot(g4c, cg, cat=cat, val=val, path=path, title=title, log=True, histtype='step', stacked=False)


def catplot(a, **kwa):
    """
    Category plot, eg Geant4 Generated Cerenkov Wavelength categorized by material  
    """
    cg = cg_get()
    a4inches = np.array((11.69, 8.28)) 
    cfg = dict(bins=100,cat='aux0',val='wavelength',ics=None, reverse=True, path=None, figsize=a4inches*0.8, title=None)
    cfg.update(kwa)

    plt.figure(figsize=cfg.pop('figsize'))    

    title = cfg.pop('title')
    if title is None:
        title = " %s (%s categories)" % (cfg['val'], cfg['cat'])

    plt.title(title)

    cat = cfg.pop('cat')
    val = cfg.pop('val')
    ics = cfg.pop('ics')
    reverse = cfg.pop('reverse')
    path = cfg.pop('path')

    catprop = getattr(a, cat)
    valprop = getattr(a, val)
    if ics is None:
        ics = np.unique(catprop)
    else:
        log.info("using argument ics")
    pass
    bc = np.bincount(catprop)

    print "ics:", ics
    cfg['label'] = "All [%d]" % bc.sum()
    plt.hist(valprop, **cfg)
    for ic in sorted(ics, key=lambda ic:bc[ic], reverse=reverse):
        material = cg.unique_materials[ic]
        cfg['label'] = "%20s  [%d]" % ( material.name[17:-9],bc[ic])
        print cfg
        plt.hist(valprop[catprop == ic], **cfg)
    pass
    plt.legend()

    if not path is None:
        log.info("saving to %s " % path)
        dirp = os.path.dirname(path)
        if not os.path.exists(dirp):
            os.makedirs(dirp)
        pass 
        plt.savefig(path)

    plt.show()


def cf_cerenkov(qty='wavelength', tag=1, **kwa):
    """
    ::

       cf_cerenkov('wavelength')
       cf_cerenkov('time')

    """
    g4c = G4CerenkovPhoton.get(tag)
    chc = ChCerenkovPhoton.get(tag)
    cf(qty, g4c, chc, **kwa)


def cf_scintillation(qty='wavelength', tag=1, **kwa):
    """
    ::

       cf_scintillation('wavelength')
       cf_scintillation('time')

    """
    g4s = G4ScintillationPhoton.get(tag)
    chs = ChScintillationPhoton.get(tag)
    cf(qty, g4s, chs, **kwa)


def plot_refractive_index(tag=1, **kwa):
    """
    G4/Detsim
       Scintillators start at 80nm, waters at 200nm

    Chroma Standard interpolated 
        Everything interpolated to start from 60nm

    """
    cs = CerenkovStep.get(tag)
    cg = cg_get()
    mm = cs.materials(cg)
    cfg = dict(qty='refractive_index')
    cfg.update(kwa)
    qplot(mm, **cfg)



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
    if standard:
        title += " standardized " 

    for m in materials:
        q = getattr(m, qty, None)
        if q is None:continue
        if standard:
            q = standardize(q)
        pass
        plt.plot( q[:,0], q[:,1], label=m.name[17:-9])
        pass
    pass
    plt.title(title)
    plt.legend()
    plt.show()  


def water_indices(cg):
    return filter(lambda _:cg.unique_materials[_].name.find('Water')>-1,range(len(cg.unique_materials)))


def cerenkov_wavelength(cs, csi=0, nrand=100000, standard=False):
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




def chroma_refractive_index():
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




def cf(qty, a, b, **kwa):
    av = getattr(a, qty)
    bv = getattr(b, qty)

    cfg = dict(bins=100,histtype="step",title=qty)
    cfg.update(kwa)

    plt.title(cfg.pop("title"))

    cfg.update(label=getattr(a,'label','b'), color='b')
    plt.hist(av, **cfg)

    cfg.update(label=getattr(b,'label','r'), color='r')
    plt.hist(bv, **cfg)

    plt.legend()
    plt.show()


def cf_wavelength(a,b, **kwa):
    cf('wavelength', a, b, **kwa)

def cf_time(a, b, **kwa):
    cfg = dict(range=(0,100))
    cfg.update(kwa)
    cf('time',a,b, **cfg)



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


def cf_3xyzw_cerenkov(tag=1, **kwa):
    g4c = G4CerenkovPhoton.get(tag)
    chc = ChCerenkovPhoton.get(tag)
    cf_3xyzw(g4c,chc, **kwa)


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

