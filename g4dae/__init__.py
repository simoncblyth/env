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

from env.g4dae.types import NPY
from env.g4dae.types import CerenkovStep, G4CerenkovPhoton, ChCerenkovPhoton
from env.g4dae.types import ScintillationStep, G4ScintillationPhoton, ChScintillationPhoton

cg = None
def cg_get():
    global cg
    if cg is None:
        cg = chroma_geometry()
    return cg 

g = None
def g_get():
    global g
    if g is None:
        g = geometry()
    return g

dae = None
def dae_get():
    global dae
    dae = daenode()
    return dae

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

def get_gdls():
    dae = dae_get()
    return dae.materialsearch("__dd__Materials__GdDopedLS")    

def get_ls():
    dae = dae_get()
    return dae.materialsearch("__dd__Materials__LiquidScintillator")
 
def plt_gdls():
    gdls = get_gdls()
    props = gdls.extra.properties
    fast = props['FASTCOMPONENT']
    slow = props['SLOWCOMPONENT']

    plt.title( "GdLS  ln(SLOWCOMPONENT) vs wl   ")
    plt.plot(slow[:,0],np.log(slow[:,1]), 'r+')
    plt.show()

 



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








def g4_cerenkov_wavelength(tag, **kwa):
    """
    """
    cg = cg_get()
    pass
    g4c = G4CerenkovPhoton.get(tag)
    base = os.path.expandvars('$STATIC_BASE/env/g4dae') 
    path =  os.path.join(base, "g4_cerenkov_wavelength.png")
    cat = "cmat"
    val = "wavelength"
    title = "G4/Detsim Generated Cerenkov Wavelength by material" 

    catplot(g4c, cat=cat, val=val, path=path, title=title, log=True, histtype='step', stacked=False)




pdgcode = {11:"e",13:"mu",22:"gamma"}
scntcode = {1:"fast",2:"slow"}
def catname(cat,ic):
    if cat == 'cmat':
        cg = cg_get()
        material = cg.unique_materials[ic]
        name = material.name[17:-9]
    elif cat == 'pdg':
        name = pdgcode.get(ic, ic) 
    elif cat == 'scnt':
        
        name = scntcode.get(ic, ic) 
    else:
        name = "%s:%s" % (cat,ic)
    pass
    return name 


def catplot(a, **kwa):
    """
    Category plot, eg Geant4 Generated Cerenkov Wavelength categorized by material  

    ::

         g4s, = NPY.mget(1,"gopscintillation")
         catplot(g4s, val='wavelength', cat='pdg' )

    """
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
        cfg['label'] = "%20s  [%d]" % (catname(cat,ic),bc[ic])
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
    g4c,chc = NPY.mget(tag, "gopcerenkov","opcerenkov")
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

def chroma_refractive_index():
    for im, cm in enumerate(cg.unique_materials):
        wlri = cm.refractive_index
        wl = wlri[:,0]
        ri = wlri[:,1]
        print "[%2d] %25s %10s     wl %7.2f : %7.2f     %10.3f : %10.3f " % ( im, cm.name[17:-9], str(wlri.shape), wl.min(), wl.max(), ri.min(), ri.max() )  


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




def cf(qty, *arys, **kwa):
    """
    Comparison histogram with legend  

    ::

        cf('wavelength', tag=1, typs="test gopcerenkov", legend=True)

        cf('3xyz', tag=1, typs="gopcerenkov opcerenkov test", legend=False)

        cf('3xyzw', tag=1, typs="opcerenkov gopcerenkov test", legend=False)


        cf('wavelength', g4s, chs, log=True)


    """
    if len(arys) == 0:
        tag  = kwa.pop('tag')
        typs = kwa.pop('typs')
        arys = NPY.mget(tag, typs) 
    pass

    cfg = dict(bins=100,histtype="step",title=qty, color="rbgcmyk", legend=True)
    cfg.update(kwa)

    if qty == '3xyz':
        qty = "posx posy posz dirx diry dirz polx poly polz"  
    elif qty == '3xyzw':
        qty = "posx posy posz time dirx diry dirz wavelength polx poly polz weight"  
    pass


    qtys = qty.split() if qty.find(' ')>-1 else [qty]
    nqty = len(qtys)

    if nqty == 1:    
        nr, nc = 1, 1
    elif nqty == 9:    
        nr, nc = 3, 3
    elif nqty == 12:    
        nr, nc = 3, 4
    elif nqty == 4:    
        nr, nc = 2, 2
    else:    
        nr, nc = nqty//2, 2    # guessing
      

    legend = cfg.pop("legend")
    color = list(cfg.pop("color"))
    title = cfg.pop("title")

    for pl,qty in enumerate(qtys):
        plt.subplot(nr,nc,pl+1)
        plt.title(qty) 
        for i, ary in enumerate(arys):
            col = color[i] 
            cfg.update(label=getattr(ary,'label',col), color=col)
            val = getattr(ary, qty)
            plt.hist(val, **cfg)
        pass
        if legend:
            plt.legend()
    pass
    plt.show()






if __name__ == '__main__':
   g = geometry()
   print g

