#!/usr/bin/env python
"""
"""
import logging, math
log = logging.getLogger(__name__)
import numpy as np

class DummyView(object):
    def __init__(self, j):
        self.j = j 
    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.j )
    eye_look_up = property(lambda self:self.j*np.arange(1,10))

def oscillate( frame, low, high, speed ):
    return low + (high-low)*(math.sin(frame*math.pi*speed)+1.)/2. 

def sawtooth( frame, low, high, speed ):
    p = int(1./speed)
    return low + (high-low)* float(frame%(p+1))/float(p) 


class DAEInterpolateView(object):
    def __init__(self, views):
        assert len(views) > 1
        self.views = views
        self.cycle = 0
        self.define_pair(0,1)
        self.f = 0.

    nviews = property(lambda self:len(self.views))  # could add a view during operation ?

    def define_pair(self, i , j ):
        self.i = i  
        self.j = j
        log.info("define_pair cycle %s i %s j %s : %s %s " % (self.cycle, i,j,  self.views[i], self.views[j])  )
        self.a = self.views[i].eye_look_up
        self.b = self.views[j].eye_look_up 
        self.cycle += 1

    def next_cycle(self):
        i = (self.i + 1) % (self.nviews - 1)           
        j = (self.j + 1) % (self.nviews - 1)           
        self.define_pair(i,j )

    def __call__(self, count, speed):
        """
        For i = 0, as f reaches 1, 
        the interpolation A->B is complete and reach v1. 

             A->B    v0->v1
             0  1

        The sawtooth is about to plunge f back to 0, so 
        next cycle changes A and B to become v1->v2

        i = 1
             A->B    v1->v2 
             1  2  
   
        """
        f = sawtooth( count, 0., 1., speed )
        bump = np.allclose(f,1.)
        #log.info("count %s f %s speed %s bump %s " % ( count, f, speed, bump ))
        self.f = f
        if bump:self.next_cycle()
            

    def __repr__(self):
        return "%s f %s A %s B %s nviews %s " % ( self.__class__.__name__, f, self.views[self.i], self.views[self.j], self.nviews) 

    def _get_eye_look_up(self):
        """Linear interpolation of two 9 element arrays """
        return (1.-self.f)*self.a + self.f*self.b

    eye_look_up = property(_get_eye_look_up)     


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    views = map(DummyView, np.arange(1,10))
    view = DAEInterpolateView( views) 

    for frame in range(1000):
        view(frame, 0.1)        
        #print view
        




    

