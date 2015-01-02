#!/bin/env python
import os, datetime, logging
log = logging.getLogger(__name__)
import numpy as np

pro_ = lambda _:load_("prop",_)
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
    def from_array(cls, arr ):
        return arr.view(cls)

    shape2type = {
            (4,4):"photon",
            (6,4):"g4step",
                 }

    @classmethod
    def detect_type(cls, arr ):
        """
        distinguish opcerenkov from opscintillation ?
        """
        assert len(arr.shape) == 3 
        itemshape = arr.shape[1:]
        typ = cls.shape2type.get(itemshape, None) 
        if typ == "g4step":
            zzz = arr[0,0,0].view(np.int32)
            if zzz < 0:
                typ = "cerenkov"
            else:
                typ = "scintillation"
            pass
        pass
        return typ 

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
    def summary(cls, tag):
        for typ in typs:
            path = path_(typ,tag)
            if os.path.exists(path):
                mt = os.path.getmtime(path)
                dt = datetime.datetime.fromtimestamp(mt)
                msg = dt.strftime("%c")
            else:
                msg = "-"
            pass
            print "%20s : %60s : %s " % (typ, path, msg)

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
    posx         = property(lambda self:self[:,0,0])
    posy         = property(lambda self:self[:,0,1])
    posz         = property(lambda self:self[:,0,2])
    time         = property(lambda self:self[:,0,3])
    position     = property(lambda self:self[:,0,:3]) 

    dirx         = property(lambda self:self[:,1,0])
    diry         = property(lambda self:self[:,1,1])
    dirz         = property(lambda self:self[:,1,2])
    wavelength   = property(lambda self:self[:,1,3])
    direction    = property(lambda self:self[:,1,:3]) 

    polx         = property(lambda self:self[:,2,0])
    poly         = property(lambda self:self[:,2,1])
    polz         = property(lambda self:self[:,2,2])
    weight       = property(lambda self:self[:,2,3])
    polarization = property(lambda self:self[:,2,:3]) 

    aux0         = property(lambda self:self[:,3,0].view(np.int32))
    aux1         = property(lambda self:self[:,3,1].view(np.int32))
    aux2         = property(lambda self:self[:,3,2].view(np.int32))
    aux3         = property(lambda self:self[:,3,3].view(np.int32))

    photonid     = property(lambda self:self[:,3,0].view(np.int32)) 
    spare        = property(lambda self:self[:,3,1].view(np.int32)) 
    flgs         = property(lambda self:self[:,3,2].view(np.uint32))
    pmt          = property(lambda self:self[:,3,3].view(np.int32))

    history      = property(lambda self:self[:,3,2].view(np.uint32))   # cannot name "flags" as that shadows a necessary ndarray property
    pmtid        = property(lambda self:self[:,3,3].view(np.int32)) 

    hits         = property(lambda self:self[self.pmtid > 0]) 
    aborts       = property(lambda self:self[np.where(self.history & 1<<31)])


    last_hit_triangle = property(lambda self:self[:,3,0].view(np.int32)) 

    #def _get_last_hit_triangles(self):
    #    if self._last_hit_triangles is None:
    #        self._last_hit_triangles = np.empty(len(self), dtype=np.int32)
    #        self._last_hit_triangles.fill(-1)
    #    return self._last_hit_triangles
    #last_hit_triangles = property(_get_last_hit_triangles)
    #
    #def _get_history(self):
    #    if self._last_hit_triangles is None:
    #        self._last_hit_triangles = np.empty(len(self), dtype=np.int32)
    #        self._last_hit_triangles.fill(-1)
    #    return self._last_hit_triangles
    #last_hit_triangles = property(_get_last_hit_triangles)




    def dump(self, index):
        log.info("dump index %d " % index)
        print self[index]
        print "photonid: ", self.photonid[index]
        print "history: ",  self.history[index]
        print "pmtid: ",    self.pmtid[index]    # is this still last_hit_triangle index when not a hit ?


class G4CerenkovPhoton(Photon):
    """DsChromaG4Cerenkov.cc"""
    typ = "gopcerenkov"
    cmat = property(lambda self:self[:,3,0].view(np.int32)) # chroma material index
    sid = property(lambda self:self[:,3,1].view(np.int32)) 
    pass
typmap[G4CerenkovPhoton.typ] = G4CerenkovPhoton

class G4ScintillationPhoton(Photon):
    """DsChromaG4Scintillation.cc"""
    typ = "gopscintillation"
    cmat = property(lambda self:self[:,3,0].view(np.int32)) # chroma material index
    sid = property(lambda self:self[:,3,1].view(np.int32)) 
    pdg = property(lambda self:self[:,3,2].view(np.int32)) 
    scnt = property(lambda self:self[:,3,3].view(np.int32)) 
    pass
typmap[G4ScintillationPhoton.typ] = G4ScintillationPhoton

class ChCerenkovPhoton(Photon):
    typ = "opcerenkov"
    pass
typmap[ChCerenkovPhoton.typ] = ChCerenkovPhoton

class ChScintillationPhoton(Photon):
    typ = "opscintillation"
    pass
typmap[ChScintillationPhoton.typ] = ChScintillationPhoton

class TestPhoton(Photon):
    typ = "test"
    pass
typmap[TestPhoton.typ] = TestPhoton





class Prop(NPY):
    """
    See test_ScintillationIntegral from gdct-
    """
    typ = "prop"
    flat = property(lambda self:self[:,0])  # unform draw from 0 to max ScintillationIntegral 
    wavelength = property(lambda self:1./self[:,1])# take reciprocal to give wavelength
    pass
typmap[Prop.typ] = Prop


class G4Step(NPY):
    typ = "g4step"
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
    pass
typmap[G4Step.typ] = G4Step


class ScintillationStep(G4Step):
    """
    see DsChromaG4Scintillation.cc
    """
    typ = "scintillation"
    pass
typmap[ScintillationStep.typ] = ScintillationStep

 
class CerenkovStep(G4Step):
    """
    see DsChromaG4Cerenkov.cc
    """
    typ = "cerenkov"
    BetaInverse = property(lambda self:self[:,4,0])
    maxSin2 = property(lambda self:self[:,5,0])
    bialkaliIndex = property(lambda self:self[:,5,3].view(np.int32))  
    pass
typmap[CerenkovStep.typ] = CerenkovStep






class VBOPhoton(Photon):
    """
    Extend Photon with VBO creation capabilites
    """
    numquad = 6 
    force_attribute_zero = "position_weight"
    @classmethod
    def from_array(cls, arr, max_slots=None):
        if arr is None:return None 
        assert max_slots
        a = arr.view(cls)
        a.max_slots = max_slots
        a._data = None
        a._ccolor = None
        a._indices = None
        return a 

    @classmethod
    def from_vbo_propagated(cls, vbo ):
        r = np.zeros( (len(vbo),4,4), dtype=np.float32 )  
        r[:,0,:4] = vbo['position_time'] 
        r[:,1,:4] = vbo['direction_wavelength'] 
        r[:,2,:4] = vbo['polarization_weight'] 
        r[:,3,:4] = vbo['last_hit_triangle'].view(r.dtype) # must view as target type to avoid coercion of int32 data into float32
        return r.view(cls) 

    def _get_data(self):
        if self._data is None:
            self._data = self.create_data(self.max_slots)
        return self._data
    data = property(_get_data)         

    def create_data(self, max_slots):
        """
        :return: numpy named constituent array with numquad*quads structure 

        Trying to replace DAEPhotonsData

        The start data is splayed out into the slots, leaving loadsa free slots
        this very sparse data structure with loadsa empty space limits 
        the number of photons that can me managed, but its for visualization  
        anyhow so do not need more than 100k or so. 

        Caution sensitivity to data structure naming:

        #. using 'position' would use traditional glVertexPointer furnishing gl_Vertex to shader
        #. using smth else eg 'position_weight' uses generic attribute , 
           which requires force_attribute_zero for anythinh to appear

        Fake attribute testing

        #. replace self.time, self.flags and self.last_hit_triangle 
           to check getting different types (especially integer) 
           attributes into shaders 

        r012_ = lambda _:np.random.randint(3, size=nvert ).astype(_)  # 0,1,2 randomly 
        r124_ = lambda _:np.array([1,2,4],dtype=_)[np.random.randint(3, size=nvert )]

        max_uint32 = (1 << 32) - 1 
        max_int32 =  (1 << 31) - 1 

        """
        if len(self) == 0:return None

        nphotons = len(self)
        nvert = nphotons*max_slots

        dtype = np.dtype([ 
            ('position_time'   ,          np.float32, 4 ), 
            ('direction_wavelength',      np.float32, 4 ), 
            ('polarization_weight',       np.float32, 4 ), 
            ('ccolor',                    np.float32, 4 ), 
            ('flags',                     np.uint32,  4 ), 
            ('last_hit_triangle',         np.int32,   4 ), 
          ])
        assert len(dtype) == self.numquad

        log.info( "create_data nphotons %d max_slots %d nvert %d (with slot scaleups) " % (nphotons,max_slots,nvert) )
        data = np.zeros(nvert, dtype )

        def pack31_( name, a, b ):
            data[name][::max_slots,:3] = a
            data[name][::max_slots,3] = b
        def pack1_( name, a):
            data[name][::max_slots,0] = a
        def pack4_( name, a):
            data[name][::max_slots] = a

        pack31_( 'position_time',        self.position ,    self.time )
        pack31_( 'direction_wavelength', self.direction,    self.wavelength )
        pack31_( 'polarization_weight',  self.polarization, self.weight  )
        pack1_(  'flags',                self.history )            # flags is used already by numpy 
        pack1_(  'last_hit_triangle',    self.last_hit_triangle )
        pack4_(  'ccolor',               self.ccolor) 

        return data


    def _get_ccolor(self):
        """
        #. initialize to red, reset by CUDA kernel
        """
        if self._ccolor is None:
            self._ccolor = np.tile( [1.,0.,0,1.], (len(self),1)).astype(np.float32)    
        return self._ccolor
    ccolor = property(_get_ccolor, doc=_get_ccolor.__doc__)

    def _get_indices(self):
        """
        List of indices
        """ 
        if self._indices is None:
            self._indices = np.arange( len(self), dtype=np.uint32)  
        return self._indices 
    indices = property(_get_indices, doc=_get_indices.__doc__)




if __name__ == '__main__':
    pass



