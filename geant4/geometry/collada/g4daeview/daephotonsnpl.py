#!/usr/bin/env python

import os, logging
import numpy as np
log = logging.getLogger(__name__)

class DAEPhotonsNPL(np.ndarray):
    """
    Usage::

        from env.geant4.geometry.collada.g4daeview.daephotonsnpl import DAEPhotonsNPL as NPL
        ## done automatically in ipython --profile=g4dae

    ::

        DAEPhotonsNPL([[[ -16520.   , -802110.   ,   -9610.   ,       2.404],
                [        nan,         nan,         nan,     550.   ],
                [      0.   ,       0.   ,       1.   ,       1.   ],
                [      0.   ,       0.   ,      -0.   ,       0.   ]]], dtype=float32)


    """
    @classmethod
    def from_array(cls, arr ):
        return arr.view(cls)

    @classmethod
    def from_vbo_propagated(cls, vbo ):
        r = np.zeros( (len(vbo),4,4), dtype=np.float32 )  
        r[:,0,:4] = vbo['position_time'] 
        r[:,1,:4] = vbo['direction_wavelength'] 
        r[:,2,:4] = vbo['polarization_weight'] 
        r[:,3,:4] = vbo['last_hit_triangle'].view(r.dtype) # must view as target type to avoid coercion of int32 data into float32
        return r.view(cls) 

    @classmethod
    def load(cls, tag):
        path = os.environ['DAE_PATH_TEMPLATE'] % tag 
        log.info("loading path %s for tag %s " % (path,tag) )
        a = np.load(path)
        return a.view(cls)

    position     = property(lambda self:self[:,0,:3]) 
    time         = property(lambda self:self[:,0, 3])

    direction    = property(lambda self:self[:,1,:3]) 
    wavelength   = property(lambda self:self[:,1, 3]) 

    polarization = property(lambda self:self[:,2,:3]) 
    weight       = property(lambda self:self[:,2, 3]) 

    photonid     = property(lambda self:self[:,3,0].view(np.int32)) 
    spare        = property(lambda self:self[:,3,1].view(np.int32)) 
    history      = property(lambda self:self[:,3,2].view(np.uint32))   # cannot name "flags" as that shadows a necessary ndarray property
    pmtid        = property(lambda self:self[:,3,3].view(np.int32)) 

    hits         = property(lambda self:self[self.pmtid > 0]) 
    aborts       = property(lambda self:self[np.where(self.history & 1<<31)])

    def dump(self, index):
        log.info("dump index %d " % index)
        print self[index]
        print "photonid: ", self.photonid[index]
        print "history: ",  self.history[index]
        print "pmtid: ",    self.pmtid[index]    # is this still last_hit_triangle index when not a hit ?




if __name__ == '__main__':
    pass
