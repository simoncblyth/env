#!/usr/bin/env python
"""
For interactive inspection::

    ipython daechromaphotonlistbase.py -i

"""
import logging
import numpy as np
log = logging.getLogger(__name__)

xyz_ = lambda x,y,z,dtype:np.column_stack((np.array(x,dtype=dtype),np.array(y,dtype=dtype),np.array(z,dtype=dtype))) 

from env.graphics.color.wav2RGB import wav2RGB

class Photons(object):
    """
    Copy of chroma.event.Photons as this is too fundamental to require chroma for
    but want to be compatible with instances of this provided by chroma.
    """
    @classmethod
    def from_cpl(cls, cpl ):
        pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        wavelengths = np.array(cpl.wavelength, dtype=np.float32)
        t = np.array(cpl.t, dtype=np.float32)
        pass
        return cls(pos,dir,pol,wavelengths,t)   # huh pmtid ignored ?

    def sort(self, order):
        self.pos = self.pos[order]
        self.dir = self.dir[order] 
        self.pol = self.pol[order] 
        self.wavelengths = self.wavelengths[order]
        self.t = self.t[order]
        #self.pmtid = self.pmtid[order]

    def __init__(self, pos, dir, pol, wavelengths, t=None, last_hit_triangles=None, flags=None, weights=None ):
        '''Create a new list of n photons.

            pos: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Position 3-vectors (mm)

            dir: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Direction 3-vectors (normalized)

            pol: numpy.ndarray(dtype=numpy.float32, shape=(n,3))
               Polarization direction 3-vectors (normalized)

            wavelengths: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon wavelengths (nm)

            t: numpy.ndarray(dtype=numpy.float32, shape=n)
               Photon times (ns)

            last_hit_triangles: numpy.ndarray(dtype=numpy.int32, shape=n)
               ID number of last intersected triangle.  -1 if no triangle hit in last step
               If set to None, a default array filled with -1 is created

            flags: numpy.ndarray(dtype=numpy.uint32, shape=n)
               Bit-field indicating the physics interaction history of the photon.  See 
               history bit constants in chroma.event for definition.

            weights: numpy.ndarray(dtype=numpy.float32, shape=n)
               Survival probability for each photon.  Used by 
               photon propagation code when computing likelihood functions.
        '''
        self.pos = np.asarray(pos, dtype=np.float32)
        self.dir = np.asarray(dir, dtype=np.float32)
        self.pol = np.asarray(pol, dtype=np.float32)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)

        if t is None:
            self.t = np.zeros(len(pos), dtype=np.float32)
        else:
            self.t = np.asarray(t, dtype=np.float32)

        if last_hit_triangles is None:
            self.last_hit_triangles = np.empty(len(pos), dtype=np.int32)
            self.last_hit_triangles.fill(-1)
        else:
            self.last_hit_triangles = np.asarray(last_hit_triangles,
                                                 dtype=np.int32)

        if flags is None:
            self.flags = np.zeros(len(pos), dtype=np.uint32)
        else:
            self.flags = np.asarray(flags, dtype=np.uint32)

        if weights is None:
            self.weights = np.ones(len(pos), dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)

    def __add__(self, other):
        '''Concatenate two Photons objects into one list of photons.

           other: chroma.event.Photons
              List of photons to add to self.

           Returns: new instance of chroma.event.Photons containing the photons in self and other.
        '''
        pos = np.concatenate((self.pos, other.pos))
        dir = np.concatenate((self.dir, other.dir))
        pol = np.concatenate((self.pol, other.pol))
        wavelengths = np.concatenate((self.wavelengths, other.wavelengths))
        t = np.concatenate((self.t, other.t))
        last_hit_triangles = np.concatenate((self.last_hit_triangles, other.last_hit_triangles))
        flags = np.concatenate((self.flags, other.flags))
        weights = np.concatenate((self.weights, other.weights))
        return Photons(pos, dir, pol, wavelengths, t,
                       last_hit_triangles, flags, weights)

    def __len__(self):
        '''Returns the number of photons in self.'''
        return len(self.pos)

    def __getitem__(self, key):
        return Photons(self.pos[key], self.dir[key], self.pol[key],
                       self.wavelengths[key], self.t[key],
                       self.last_hit_triangles[key], self.flags[key],
                       self.weights[key])

    def reduced(self, reduction_factor=1.0):
        '''Return a new Photons object with approximately
        len(self)*reduction_factor photons.  Photons are selected
        randomly.'''
        n = len(self)
        choice = np.random.permutation(n)[:int(n*reduction_factor)]
        return self[choice]



class DAEPhotons(object):
    atts = 'pos dir pol wavelengths t pmtid color weights flags last_hit_triangle'.split()
    """
    #. Q: how to get smth drawable and sharable between Chroma and OpenGL ? 
          and avoid all the copying 
    """

    @classmethod
    def from_cpl(cls, cpl):
        photons = Photons.from_cpl(cpl)
        pmtid = np.array(cpl.pmtid, dtype=np.int32)
        order = np.argsort(photons.t)
        photons.sort(order)        
        pmtid = pmtid[order]
        return cls(photons, pmtid)

    def __init__(self, photons, pmtid=None ):
        self.photons = photons
        self.pmtid = pmtid
        self.nphotons = len(self.photons)
        self.vertices = self.photons.pos      # allows to be treated like a DAEMesh
        self._color = None

    def wavelengths2rgb(self):
        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.photons.wavelengths):
            color[i] = wav2RGB(wl)
        pass 
        return color

    def _get_color(self):
        if self._color is None:
            self._color = self.wavelengths2rgb()
        return self._color
    color = property(_get_color)

    def dump(self):
        for att in cls.atts:
            print "%s\n%s" % (att,getattr(self.photons,att) if hasattr(self.photons,att) else "-")


def check_load(path):

    cpl = load_cpl(path,'CPL')

    print "g4 stack action track order"
    dpho = DAEPhotons.from_cpl(cpl, timesort=False)
    dpho.dump()

    print "timesorted order"
    dpho = DAEPhotons.from_cpl(cpl, timesort=True)
    dpho.dump()



if __name__ == '__main__':
    import os
    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl

    path = os.environ['DAE_PATH_TEMPLATE'] % {'arg':"1"} 
    check_load(path)




