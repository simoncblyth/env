#!/bin/env python
import os
import numpy as np

from env.geant4.geometry.collada.g4daeview.daephotonsnpl import DAEPhotonsNPL as NPL
npl = lambda _:NPL.load(_)

ppp_ = lambda _:load_("photon",_)
hhh_ = lambda _:load_("hit",_)
ttt_ = lambda _:load_("test",_)
stc_ = lambda _:load_("cerenkov",_)
sts_ = lambda _:load_("scintillation",_)
chc_ = lambda _:load_("opcerenkov",_)
chs_ = lambda _:load_("opscintillation",_)
g4c_ = lambda _:load_("gopcerenkov",_)
g4s_ = lambda _:load_("gopscintillation",_)

path_ = lambda typ,tag:os.environ["DAE_%s_PATH_TEMPLATE" % typ.upper()] % str(tag)
load_ = lambda typ,tag:np.load(path_(typ,tag))     

global typs
typs = "photon hit test cerenkov scintillation opcerenkov opscintillation gopcerenkov gopscintillation".split()

global typmap
typmap = {}

class NPY(np.ndarray):
    @classmethod
    def get(cls, tag):
        """
        # viewing an ndarray as a subclass allows adding customizations 
          on top of the ndarray while using the same storage
        """
        a = load_(cls.typ, tag).view(cls)
        a.tag = tag
        return a

    label = property(lambda self:"%s.get(%s)" % (self.__class__.__name__, self.tag))

    @classmethod
    def mget(cls, tag, *typs):
        """
        Load multiple typed instances::

            chc, g4c, tst = NPY.mget(1,"opcerenkov","gopcerenkov","test")

        """
        if len(typs) == 1:
            typs = typs[0].split()

        klss = map(lambda _:typmap[_], typs)
        arys = map(lambda kls:kls.get(tag), klss)
        return arys



class Photon(NPY):
    posx = property(lambda self:self[:,0,0])
    posy = property(lambda self:self[:,0,1])
    posz = property(lambda self:self[:,0,2])
    time = property(lambda self:self[:,0,3])

    dirx = property(lambda self:self[:,1,0])
    diry = property(lambda self:self[:,1,1])
    dirz = property(lambda self:self[:,1,2])
    wavelength = property(lambda self:self[:,1,3])

    polx = property(lambda self:self[:,2,0])
    poly = property(lambda self:self[:,2,1])
    polz = property(lambda self:self[:,2,2])
    weight = property(lambda self:self[:,2,3])

    aux0 = property(lambda self:self[:,3,0].view(np.int32))
    aux1 = property(lambda self:self[:,3,1].view(np.int32))
    flgs = property(lambda self:self[:,3,2].view(np.uint32))
    pmt  = property(lambda self:self[:,3,3].view(np.int32))


class G4CerenkovPhoton(Photon):
    """
    see DsChromaG4Cerenkov.cc
    """
    typ = "gopcerenkov"
    cmat = property(lambda self:self[:,3,0].view(np.int32)) # chroma material index
    csid = property(lambda self:self[:,3,1].view(np.int32)) # 1-based CerenkovStep index 
typmap[G4CerenkovPhoton.typ] = G4CerenkovPhoton

class ChCerenkovPhoton(Photon):
    typ = "opcerenkov"
typmap[ChCerenkovPhoton.typ] = ChCerenkovPhoton

class G4ScintillationPhoton(Photon):
    typ = "gopscintillation"
typmap[G4ScintillationPhoton.typ] = G4ScintillationPhoton

class ChScintillationPhoton(Photon):
    typ = "opscintillation"
typmap[ChScintillationPhoton.typ] = ChScintillationPhoton

class TestPhoton(Photon):
    typ = "test"
typmap[TestPhoton.typ] = TestPhoton







class G4Step(NPY):
    sid = property(lambda self:self[:,0,0].view(np.int32))    # 0
    parentId = property(lambda self:self[:,0,1].view(np.int32))
    materialIndex = property(lambda self:self[:,0,2].view(np.int32))
    numPhotons = property(lambda self:self[:,0,3].view(np.int32))  

    code = property(lambda self:self[:,3,0].view(np.int32))   # 3 

    totPhotons = property(lambda self:int(self.numPhotons.sum()))
    materialIndices = property(lambda self:np.unique(self.materialIndex))

    def materials(self, _cg):
        """
        :param _cg: chroma geometry instance
        :return: list of chroma material instances relevant to this evt 
        """
        return [_cg.unique_materials[materialIndex] for materialIndex in self.materialIndices]


class ScintillationStep(G4Step):
    """
    see DsChromaG4Scintillation.cc
    """
    typ = "scintillation"

 
class CerenkovStep(G4Step):
    """
    see DsChromaG4Cerenkov.cc
    """
    typ = "cerenkov"
    BetaInverse = property(lambda self:self[:,4,0])
    maxSin2 = property(lambda self:self[:,5,0])
    bialkaliIndex = property(lambda self:self[:,5,3].view(np.int32))  



if __name__ == '__main__':
    pass



