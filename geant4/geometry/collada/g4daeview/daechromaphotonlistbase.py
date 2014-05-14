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
    def __init__(self, cpl, timesort=True ):
        self.cpl = cpl
        self.nphotons = cpl.x.size()
        cpl.Print()
        self.copy_from_cpl(cpl, timesort=timesort)

    vertices = property(lambda self:self.pos )  # allow to be treated like a solid

    def copy_from_cpl(self, cpl, timesort):
        """
        Q:

        #. how to get smth drawable and sharable between Chroma and OpenGL

        """
        pos = xyz_(cpl.x,cpl.y,cpl.z,np.float32)
        dir = xyz_(cpl.px,cpl.py,cpl.pz,np.float32)
        pol = xyz_(cpl.polx,cpl.poly,cpl.polz,np.float32)
        wavelength = np.array(cpl.wavelength, dtype=np.float32)
        t = np.array(cpl.t, dtype=np.float32)
        pmtid = np.array(cpl.pmtid, dtype=np.int32)



        if not timesort:
            self.pos = pos 
            self.dir = dir 
            self.pol = pol 
            self.wavelength = wavelength
            self.t = t
            self.pmtid = pmtid
        else:
            order = np.argsort(t)
            self.pos = pos[order]
            self.dir = dir[order] 
            self.pol = pol[order] 
            self.wavelength = wavelength[order]
            self.t = t[order]
            self.pmtid = pmtid[order]
        pass

        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        # hmm, a more numpy way of doing this ?
        for i,wl in enumerate(self.wavelength):
            color[i] = wav2RGB(wl)
        pass
        self.color = color


    def dump(self):
        print "pos\n",self.pos
        print "dir\n",self.dir
        print "pol\n",self.pol
        print "wavelength\n",self.wavelength
        print "t\n",self.t
        print "pmtid\n",self.pmtid
        print "color\n",self.color
      


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







