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


class DAEChromaPhotonListBase(object):
    atts = 'pos dir pol wavelengths t pmtid color weights flags last_hit_triangle'.split()
    def __init__(self, cpl, timesort=True, chroma=False):
        self.cpl = cpl
        nphotons = cpl.x.size()
        cpl.Print()
        self.nphotons = nphotons
        pass
        self.color = None
        self.last_hit_triangles = None
        self.flags = None
        self.weights = None 
        pass
        self.copy_from_cpl(cpl, timesort=timesort)
        self.color_setup(nphotons)
        if chroma:
            self.chroma_setup(nphotons)
        pass

    vertices = property(lambda self:self.pos )  # allow to be treated like a solid

    def copy_from_cpl(self, cpl, timesort):
        """
        Q:

        #. how to get smth drawable and sharable between Chroma and OpenGL

        """
        pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        wavelengths = np.array(cpl.wavelength, dtype=np.float32)
        t = np.array(cpl.t, dtype=np.float32)
        pmtid = np.array(cpl.pmtid, dtype=np.int32)

        if not timesort:
            self.pos = pos 
            self.dir = dir 
            self.pol = pol 
            self.wavelengths = wavelengths
            self.t = t
            self.pmtid = pmtid
        else:
            order = np.argsort(t)
            self.pos = pos[order]
            self.dir = dir[order] 
            self.pol = pol[order] 
            self.wavelengths = wavelengths[order]
            self.t = t[order]
            self.pmtid = pmtid[order]
        pass

    def color_setup(self, nphotons):
        """
        #. hmm, a more numpy way ?
        """
        color = np.zeros(nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.wavelengths):
            color[i] = wav2RGB(wl)
        pass
        self.color = color
    def chroma_setup(self, nphotons, last_hit_triangles=None, flags=None, weights=None):
        """
        Allows DAEChromaPhotonListBase instances to act like chroma.event.Photons instances
        """
        if last_hit_triangles is None:
            self.last_hit_triangles = np.empty(nphotons, dtype=np.int32)
            self.last_hit_triangles.fill(-1)
        else:
            self.last_hit_triangles = np.asarray(last_hit_triangles,
                                                 dtype=np.int32)

        if flags is None:
            self.flags = np.zeros(nphotons, dtype=np.uint32)
        else:
            self.flags = np.asarray(flags, dtype=np.uint32)

        if weights is None:
            self.weights = np.ones(nphotons, dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        '''Returns the number of photons in self.'''
        return len(self.pos)

    def dump(self):
        self.dump_(self)

    @classmethod
    def dump_(cls,obj):
        for att in cls.atts:
            print "%s\n%s" % (att,getattr(obj,att) if hasattr(obj,att) else "-")


def check_load(path):

    cpl = load_cpl(path,'CPL')

    print "g4 stack action track order"
    dcpl = DAEChromaPhotonListBase(cpl, timesort=False)
    dcpl.dump()

    print "timesorted order"
    dcpl = DAEChromaPhotonListBase(cpl, timesort=True)
    dcpl.dump()




if __name__ == '__main__':
    import os
    import ROOT
    ROOT.gSystem.Load("$LOCAL_BASE/env/chroma/ChromaPhotonList/lib/libChromaPhotonList")
    from env.chroma.ChromaPhotonList.cpl import load_cpl, save_cpl, random_cpl
    path = os.environ['DAE_PATH_TEMPLATE'] % {'arg':"1"} 







