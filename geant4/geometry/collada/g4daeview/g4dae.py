#!/usr/bin/env python
import os
import numpy as np
 
ph = lambda _:np.load(os.environ['DAE_PATH_TEMPLATE'] % _)
hh = lambda _:np.load(os.environ['DAEHIT_PATH_TEMPLATE'] % _)

class Tag(object):
    """
    For tag "hh1" loads photon and hit lists 
    created by running eg:: 
    
        mocknuwa.sh 1 hh
       
    See ~/g4dae.ipynb notebook that uses this. 
    Access with::

        ipython notebook --profile g4dae  # or ipython-nb

    """ 
    def __init__(self, tag):
        self.tag = tag
        self.p = ph(tag)   # photonlist
        self.h = hh(tag)   # hitlist 

    gx = property(lambda self:self.p[:,0,0])  # global pos, time
    gy = property(lambda self:self.p[:,0,1])
    gz = property(lambda self:self.p[:,0,2])
    gt = property(lambda self:self.p[:,0,3])
    
    dx = property(lambda self:self.p[:,1,0])  # global direction, wavelength 
    dy = property(lambda self:self.p[:,1,1])
    dz = property(lambda self:self.p[:,1,2])
    dw = property(lambda self:self.p[:,1,3])
    
    px = property(lambda self:self.p[:,2,0])  # global pol, weight
    py = property(lambda self:self.p[:,2,1])
    pz = property(lambda self:self.p[:,2,2])
    pw = property(lambda self:self.p[:,2,3])
    
    pid = property(lambda self:self.p[:,3,0].view(np.int32))
    slot = property(lambda self:self.p[:,3,1].view(np.int32))  # distinguish vbo from non-vbo by this
    flag = property(lambda self:self.p[:,3,2].view(np.uint32))
    pmt  = property(lambda self:self.p[:,3,3].view(np.int32)) 
    
    lx = property(lambda self:self.h[:,3,0])   # PMT local frame hit positions
    ly = property(lambda self:self.h[:,3,1])
    lz = property(lambda self:self.h[:,3,2])





def main():
    import sys
    np.set_printoptions(precision=3,suppress=True)

    name = sys.argv[1]
    a = ph(name)
    print a

    if len(sys.argv) > 2 and sys.argv[2] == '-i': 
        import IPython
        IPython.embed() 


if __name__ == '__main__':
    main()



