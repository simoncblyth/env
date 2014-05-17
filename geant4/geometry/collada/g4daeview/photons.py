#!/usr/bin/env python
"""
**WARNING** : This class `photons.Photons` duplicates `chroma.event.Photons` (in my forked chroma)

Duplication as wish to provide a fallback for installations without Chroma

"""
import logging
import numpy as np
log = logging.getLogger(__name__)

xyz_ = lambda x,y,z,dtype:np.column_stack((np.array(x,dtype=dtype),np.array(y,dtype=dtype),np.array(z,dtype=dtype))) 


# Photon history bits (see photon.h for source)
NO_HIT           = 0x1 << 0
BULK_ABSORB      = 0x1 << 1
SURFACE_DETECT   = 0x1 << 2
SURFACE_ABSORB   = 0x1 << 3
RAYLEIGH_SCATTER = 0x1 << 4
REFLECT_DIFFUSE  = 0x1 << 5
REFLECT_SPECULAR = 0x1 << 6
SURFACE_REEMIT   = 0x1 << 7
SURFACE_TRANSMIT = 0x1 << 8
BULK_REEMIT      = 0x1 << 9
NAN_ABORT        = 0x1 << 31

PHOTON_FLAGS =  {
            'NO_HIT':NO_HIT,
       'BULK_ABSORB':BULK_ABSORB,
    'SURFACE_DETECT':SURFACE_DETECT,
    'SURFACE_ABSORB':SURFACE_ABSORB,
  'RAYLEIGH_SCATTER':RAYLEIGH_SCATTER,
   'REFLECT_DIFFUSE':REFLECT_DIFFUSE,
  'REFLECT_SPECULAR':REFLECT_SPECULAR,
    'SURFACE_REEMIT':SURFACE_REEMIT,
  'SURFACE_TRANSMIT':SURFACE_TRANSMIT,
       'BULK_REEMIT':BULK_REEMIT,
         'NAN_ABORT':NAN_ABORT,
            }   

def arg2mask_( argl ):
    """ 
    Convert comma delimited strings like the below into OR mask integer:

    #. "BULK_ABSORB,SURFACE_DETECT" 
    #. "SURFACE_REEMIT" 
    """
    mask = 0 
    for arg in argl.split(","):
        if arg in PHOTON_FLAGS:
           mask |= PHOTON_FLAGS[arg]
        pass 
    return mask





class Photons(object):
    """
    Same as chroma.event.Photons but with a few additions:

    #. from_cpl classmethod 
    #. timesorting
    #. dump
    """
    @classmethod
    def from_cpl(cls, cpl, extend=True ):
        """
        :param cpl: ChromaPhotonList instance, as obtained from file or MQ
        :param extend: when True add pmtid attribute and sort by time
        """ 
        pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        wavelengths = np.array(cpl.wavelength, dtype=np.float32)
        t = np.array(cpl.t, dtype=np.float32)
        pass
        obj = cls(pos,dir,pol,wavelengths,t)   
       
        if extend:
            order = np.argsort(obj.t)
            pmtid = np.array(cpl.pmtid, dtype=np.int32)
            obj.sort(order)        
            obj.pmtid = pmtid[order]
        pass
        return obj

    atts = "pos dir pol wavelengths t last_hit_triangles flags weights".split()

    def dump(self):
        for att in self.atts:
            print "%s\n%s" % ( att, getattr(self, att))

    def sort(self, order):
        self.pos = self.pos[order]
        self.dir = self.dir[order] 
        self.pol = self.pol[order] 
        self.wavelengths = self.wavelengths[order]
        self.t = self.t[order]

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


def check_load(path):
    cpl = load_cpl(path,'CPL')
    pho = Photons.from_cpl(cpl)
    pho.dump()


if __name__ == '__main__':
    import os
    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl

    path = os.environ['DAE_PATH_TEMPLATE'] % {'arg':"1"} 
    check_load(path)




